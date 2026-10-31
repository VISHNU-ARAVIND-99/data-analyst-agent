import sys
import io
import traceback
import uuid
import re
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread-safe rendering
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None
import seaborn as sns
import duckdb
from config import CHARTS_DIR

# Compatibility shims for LLMs that call .to_pandas() or .to_df() on DataFrames or DuckDB relations
if not hasattr(pd.DataFrame, "to_pandas"):
    pd.DataFrame.to_pandas = lambda self: self
if not hasattr(pd.DataFrame, "to_df"):
    pd.DataFrame.to_df = lambda self: self
if hasattr(duckdb, "DuckDBPyRelation") and not hasattr(duckdb.DuckDBPyRelation, "to_pandas"):
    duckdb.DuckDBPyRelation.to_pandas = lambda self: self.df()

class PythonSandbox:
    """Safely executes Python analytical scripts, supports multi-table DuckDB & Pandas, intercepts charts and stdout."""

    # Patterns that indicate the LLM tried to read files instead of using injected DataFrames
    FILE_IO_PATTERNS = [
        (r'pd\.read_csv\s*\(', 'pd.read_csv()'),
        (r'pd\.read_excel\s*\(', 'pd.read_excel()'),
        (r'pd\.read_parquet\s*\(', 'pd.read_parquet()'),
        (r'pd\.read_json\s*\(', 'pd.read_json()'),
        (r'open\s*\(.*["\'].*\.csv', 'open() with csv file'),
    ]

    class DuckDBProxy:
        """Transparent proxy for DuckDB ensuring connect() returns default connection with all registered tables."""
        def __init__(self, duckdb_mod):
            self._duckdb = duckdb_mod

        def connect(self, *args, **kwargs):
            return self._duckdb.default_connection()

        def __getattr__(self, name):
            return getattr(self._duckdb, name)

    @staticmethod
    def extract_code(text: str) -> str:
        """Extract code from markdown code blocks or return clean text, sanitizing module-level return statements."""
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text, re.IGNORECASE)
        code = matches[0].strip() if matches else text.strip()

        # Sanitize top-level unindented 'return <expr>' to 'print(<expr>)' to avoid SyntaxError: 'return' outside function
        clean_lines = []
        for line in code.splitlines():
            m = re.match(r"^return\s+(.+)$", line)
            if m:
                clean_lines.append(f"print({m.group(1).strip()})")
            elif line.strip() == "return":
                continue
            else:
                clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    @staticmethod
    def pre_check_code(code: str, available_names: list, real_columns: dict = None) -> dict:
        """
        Static analysis before execution.
        Detects syntax errors, file reads, and hallucinated SQL column names.
        real_columns: dict of {table_name: [col1, col2, ...]} for column validation.
        """
        import ast

        # 1. AST syntax check
        try:
            ast.parse(code)
        except SyntaxError as syn_err:
            return {
                "ok": False,
                "error": f"SyntaxError on line {syn_err.lineno}: {syn_err.msg}"
            }

        # 2. File I/O checks
        for pattern, label in PythonSandbox.FILE_IO_PATTERNS:
            if re.search(pattern, code):
                names_str = ', '.join(available_names)
                return {
                    "ok": False,
                    "error": (
                        f"Generated code uses '{label}' — data is already loaded in memory. "
                        f"DO NOT read files. Use the pre-loaded DataFrames directly: {names_str}. "
                        f"For DuckDB: duckdb.query('SELECT ... FROM {available_names[0] if available_names else 'df'}').df()"
                    )
                }

        # 3. SQL column hallucination check — scan duckdb.query() strings for unknown column names
        if real_columns:
            all_real_cols = set()
            for cols in real_columns.values():
                all_real_cols.update(c.lower() for c in cols)

            # Extract SQL strings from duckdb.query("...") calls
            sql_strings = re.findall(r'duckdb\.(?:query|sql|execute)\s*\(\s*[f]?["\']([^"\']+)["\']', code)
            sql_strings += re.findall(r'duckdb\.(?:query|sql|execute)\s*\(\s*[f]?"""([\s\S]*?)"""', code)
            for sql in sql_strings:
                # Collect AS-aliases so we don't flag them as unknown columns
                aliases = set(m.lower() for m in re.findall(r'\bAS\s+(\w+)', sql, re.IGNORECASE))

                sql_keywords = {
                    'select','from','where','group','by','order','having','join','on',
                    'left','right','inner','outer','full','cross','as','and','or','not',
                    'in','is','null','count','sum','avg','min','max','mean','median',
                    'distinct','limit','offset','asc','desc','case','when','then',
                    'else','end','coalesce','cast','round','floor','ceil','abs','length',
                    'upper','lower','trim','substr','replace','strftime','date_trunc',
                    'int','float','double','varchar','string','boolean','date','timestamp',
                    'between','like','ilike','exists','all','any','union','intersect',
                    'except','with','over','partition','row_number','rank','dense_rank',
                    'lag','lead','first_value','last_value','ntile','percent_rank',
                    'stddev','variance','corr','percentile_cont','percentile_disc',
                    'true','false','interval','extract','epoch','year','month','day',
                }
                tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql)
                for tok in tokens:
                    tl = tok.lower()
                    if tl in sql_keywords:
                        continue
                    if tl in aliases:          # skip AS-created alias names
                        continue
                    if tl in {n.lower() for n in available_names}:
                        continue
                    if tl not in all_real_cols:
                        col_list = {k: list(v) for k, v in real_columns.items()}
                        return {
                            "ok": False,
                            "error": (
                                f"BinderException (pre-check): SQL references column '{tok}' which does NOT exist.\n"
                                f"Real columns: {col_list}\n"
                                f"ONLY use column names from the list above."
                            )
                        }

            # 4. Pandas column access hallucination check
            # Catch: df['salary_lpa'], df["salary_lpa"], groupby(['col']), dropna(subset=['col']), etc.
            # Extract all string literals that look like column name accesses
            pandas_col_refs = set()

            # df['col'] or df["col"] patterns
            pandas_col_refs.update(re.findall(r'\bdf\s*\[\s*["\']([^"\']+)["\']', code))
            # Also named DataFrames: india_job_market_2024_2026['col']
            for tname in available_names:
                safe = tname.replace(" ", "_").replace("-", "_").lower()
                pandas_col_refs.update(re.findall(rf'\b{re.escape(safe)}\s*\[\s*["\']([^"\']+)["\']', code))

            # .groupby('col') or .groupby(['col', 'col2'])
            pandas_col_refs.update(re.findall(r'\.groupby\s*\(\s*["\']([^"\']+)["\']', code))
            pandas_col_refs.update(re.findall(r'\.groupby\s*\(\s*\[[^\]]*["\']([^"\']+)["\']', code))

            # .dropna(subset=['col']), .sort_values('col'), .rename(columns={'old':'new'})
            pandas_col_refs.update(re.findall(r'subset\s*=\s*\[[^\]]*["\']([^"\']+)["\']', code))
            pandas_col_refs.update(re.findall(r'\.sort_values\s*\(\s*["\']([^"\']+)["\']', code))

            for col_ref in pandas_col_refs:
                if col_ref.lower() not in all_real_cols:
                    col_list = {k: list(v) for k, v in real_columns.items()}
                    return {
                        "ok": False,
                        "error": (
                            f"KeyError (pre-check): Code references column '{col_ref}' which does NOT exist.\n"
                            f"Real columns: {col_list}\n"
                            f"ONLY use column names from the list above."
                        )
                    }

        return {"ok": True, "error": None}

    @classmethod
    def execute(cls, code: str, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Execute python code in an isolated environment with access to data.
        `data` can be a single DataFrame or a dictionary of named DataFrames.
        """
        clean_code = cls.extract_code(code)

        # --- Pre-execution static analysis: catch syntax/file-read/hallucinated-column errors ---
        if isinstance(data, dict):
            available_names = list(data.keys())
            real_columns = {name: list(df.columns) for name, df in data.items()}
        else:
            available_names = ["df"]
            real_columns = {"df": list(data.columns)}

        check = cls.pre_check_code(clean_code, available_names, real_columns)
        if not check["ok"]:
            return {
                "success": False,
                "code": clean_code,
                "stdout": "",
                "chart_paths": [],
                "error": check["error"],
                "diagnostic": f"Pre-check caught hallucinated column. Real columns: {real_columns}",
                "traceback": f"Pre-execution check failed: {check['error']}"
            }

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        # Setup isolated environment scope
        default_conn = duckdb.default_connection()
        exec_scope = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "duckdb": cls.DuckDBProxy(duckdb),
            "con": default_conn,
            "conn": default_conn
        }

        # Inject DataFrames into execution scope and register into DuckDB
        if isinstance(data, dict):
            exec_scope["tables"] = data
            for name, df_obj in data.items():
                safe_name = str(name).replace(" ", "_").replace("-", "_").lower()
                exec_scope[safe_name] = df_obj.copy()
                try:
                    duckdb.register(safe_name, df_obj)
                except Exception:
                    pass
            # Set default df to the first table for convenience
            if data:
                exec_scope["df"] = list(data.values())[0].copy()
        else:
            exec_scope["df"] = data.copy()
            try:
                duckdb.register("df", data)
            except Exception:
                pass

        generated_charts: List[str] = []

        try:
            plt.close('all')
            sys.stdout = stdout_capture

            # Execute code with unified global/local scope to ensure clean function & comprehension access
            exec(clean_code, exec_scope, exec_scope)

            # Capture any generated matplotlib figures and extract plotted data
            fig_nums = plt.get_fignums()
            extracted_plot_data = []
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                chart_filename = f"chart_{uuid.uuid4().hex[:8]}.png"
                chart_path = CHARTS_DIR / chart_filename
                fig.savefig(chart_path, bbox_inches='tight', dpi=200)
                generated_charts.append(str(chart_path))

                # Extract plotted values from axes so synthesis has actual numbers even without print()
                for ax in fig.get_axes():
                    title = ax.get_title() or "Chart"
                    xlabel = ax.get_xlabel() or "Categories"
                    ylabel = ax.get_ylabel() or "Values"
                    x_ticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
                    if ax.patches:
                        entries = []
                        for idx, patch in enumerate(ax.patches):
                            val = round(float(patch.get_height()), 2)
                            lbl = x_ticks[idx % len(x_ticks)] if (x_ticks and idx < len(x_ticks)) else f"Item {idx+1}"
                            entries.append(f"{lbl}: {val}")
                        if entries:
                            extracted_plot_data.append(f"[{title} ({xlabel} vs {ylabel})]: " + ", ".join(entries[:25]))
                    for line in ax.get_lines():
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        lbl = line.get_label() or "Trend"
                        if len(x_data) > 0 and len(y_data) > 0:
                            points = [f"{x}: {round(float(y), 2)}" for x, y in zip(x_data[:10], y_data[:10])]
                            extracted_plot_data.append(f"[{title} - {lbl}]: " + ", ".join(points))
            plt.close('all')

            sys.stdout = old_stdout
            stdout_text = stdout_capture.getvalue()

            # Auto-capture any computed DataFrames or Series if stdout is empty
            if not stdout_text.strip():
                auto_captured = []
                for var_name, var_val in exec_scope.items():
                    if var_name.startswith("_") or var_name in ["pd", "np", "plt", "sns", "duckdb", "con", "conn", "tables", "df"]:
                        continue
                    if isinstance(var_val, pd.DataFrame) and not var_val.empty:
                        auto_captured.append(f"[Auto-captured DataFrame `{var_name}`]:\n" + var_val.head(25).to_string())
                    elif isinstance(var_val, pd.Series) and not var_val.empty:
                        auto_captured.append(f"[Auto-captured Series `{var_name}`]:\n" + var_val.head(25).to_string())
                    elif isinstance(var_val, (int, float, np.number)):
                        auto_captured.append(f"[Auto-captured `{var_name}`]: {round(float(var_val), 2)}")

                if auto_captured:
                    stdout_text = "\n\n".join(auto_captured)
                elif extracted_plot_data:
                    stdout_text = "\n".join(extracted_plot_data)
            elif extracted_plot_data:
                stdout_text += "\n\n" + "\n".join(extracted_plot_data)

            return {
                "success": True,
                "code": clean_code,
                "stdout": stdout_text,
                "chart_paths": generated_charts,
                "error": None,
                "diagnostic": None,
                "traceback": None
            }

        except Exception as e:
            sys.stdout = old_stdout
            plt.close('all')
            tb = traceback.format_exc()
            err_type = type(e).__name__
            formatted_error = f"{err_type}: {str(e)}"

            # Generate smart diagnostic hints for the self-healing loop
            diagnostic_hints = []
            if isinstance(e, KeyError):
                missing_key = str(e)
                df_infos = []
                for var_name, var_val in exec_scope.items():
                    if isinstance(var_val, pd.DataFrame) and not var_name.startswith("_") and var_name not in ("tables",):
                        df_infos.append(f"  \u2022 `{var_name}` ACTUAL columns: {list(var_val.columns)}")
                diagnostic_hints.append(
                    f"KeyError accessing {missing_key}. That column does NOT exist.\n"
                    f"ACTUAL columns in all DataFrames in scope:\n" + "\n".join(df_infos) + "\n"
                    f"ONLY use columns from the lists above. Do NOT use column names from previous queries."
                )
                if "salary" in str(missing_key).lower():
                    diagnostic_hints.append(
                        "HINT: You referenced a salary column that may not be needed for this question. "
                        "Re-read the original question and only use columns relevant to it."
                    )
            elif isinstance(e, (AttributeError, TypeError)):
                diagnostic_hints.append(f"Type/Attribute error: {e}. Check if you called a Series method on a DataFrame or vice versa.")
            elif "Catalog Error" in str(e) or "Binder Error" in str(e):
                # Dump the actual columns from every registered DataFrame so the retry LLM has ground truth
                col_dump = []
                for var_name, var_val in exec_scope.items():
                    if isinstance(var_val, pd.DataFrame) and not var_name.startswith("_") and var_name not in ("tables",):
                        col_dump.append(f"  • `{var_name}` real columns: {list(var_val.columns)}")
                col_info = "\n".join(col_dump) if col_dump else "  (no DataFrames found in scope)"
                diagnostic_hints.append(
                    f"DuckDB SQL Binder/Catalog Error — a column or table name in your SQL does NOT exist.\n"
                    f"Available tables in DuckDB: {available_names}\n"
                    f"ACTUAL columns in those DataFrames:\n{col_info}\n"
                    f"ONLY use column names from the lists above. Do NOT invent column names like 'company_id'."
                )


            diagnostic_str = "\n".join(diagnostic_hints)

            return {
                "success": False,
                "code": clean_code,
                "stdout": stdout_capture.getvalue(),
                "chart_paths": generated_charts,
                "error": formatted_error,
                "diagnostic": diagnostic_str,
                "traceback": tb
            }
        finally:
            sys.stdout = old_stdout
            plt.close('all')

