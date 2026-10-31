import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class DataProfiler:
    """Profiles datasets deterministically with pandas to extract stats, anomalies, and schema."""

    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform safe, automated initial cleaning transformations on a single dataframe.
        Returns: (cleaned_df, list_of_applied_transformations)
        """
        cleaned = df.copy()
        transformations = []

        # 1. Clean column names (strip whitespace)
        initial_cols = list(cleaned.columns)
        cleaned.columns = [str(c).strip().replace(" ", "_").lower() for c in cleaned.columns]
        if list(cleaned.columns) != initial_cols:
            transformations.append("Standardized column names (lowercased, spaces to underscores).")

        # 2. Check for duplicate rows
        dup_count = cleaned.duplicated().sum()
        if dup_count > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            transformations.append(f"Removed {dup_count} exact duplicate rows.")

        # 3. Detect and convert date columns
        for col in cleaned.columns:
            if cleaned[col].dtype == 'object':
                sample_vals = cleaned[col].dropna().astype(str).head(20)
                if len(sample_vals) > 0:
                    date_match_count = sum(bool(re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', v)) for v in sample_vals)
                    if date_match_count >= len(sample_vals) * 0.8:
                        try:
                            cleaned[col] = pd.to_datetime(cleaned[col], errors='coerce')
                            transformations.append(f"Converted '{col}' to datetime.")
                        except Exception:
                            pass

        # 4. Strip currency and percentage symbols from numeric strings
        for col in cleaned.columns:
            if cleaned[col].dtype == 'object':
                sample_vals = cleaned[col].dropna().astype(str).head(20)
                if len(sample_vals) > 0:
                    has_currency = any(re.search(r'[\$€£₹]', v) for v in sample_vals)
                    has_pct = any('%' in v for v in sample_vals)
                    if has_currency or has_pct:
                        try:
                            cleaned_col = (
                                cleaned[col].astype(str)
                                .str.replace(r'[\$€£₹,]', '', regex=True)
                                .str.replace('%', '', regex=False)
                                .str.strip()
                            )
                            numeric_vals = pd.to_numeric(cleaned_col, errors='coerce')
                            if numeric_vals.notnull().sum() >= len(sample_vals) * 0.8:
                                cleaned[col] = numeric_vals
                                transformations.append(f"Sanitized currency/percent format in '{col}' to numeric float.")
                        except Exception:
                            pass

        return cleaned, transformations

    @classmethod
    def clean_tables(cls, tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        """Clean all tables in a multi-table dictionary."""
        cleaned_tables = {}
        all_transforms = {}
        for name, df in tables.items():
            clean_name = name.lower().replace(".csv", "").replace(".xlsx", "").replace(" ", "_").strip()
            c_df, transforms = cls.clean_dataset(df)
            cleaned_tables[clean_name] = c_df
            all_transforms[clean_name] = transforms
        return cleaned_tables, all_transforms

    @staticmethod
    def detect_relationships(tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Auto-detect foreign key relationships between tables by matching ID column names and values.
        """
        relationships = []
        table_names = list(tables.keys())

        seen_links = set()
        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                t1_name, t2_name = table_names[i], table_names[j]
                df1, df2 = tables[t1_name], tables[t2_name]

                # Find candidate key columns (prioritize columns with 'id', 'key', 'code', or exact match)
                common_cols = [c for c in df1.columns if c in df2.columns]
                for col in common_cols:
                    # Ignore common generic non-key columns
                    col_lower = str(col).lower()
                    if col_lower in ['date', 'time', 'status', 'type', 'created_at', 'updated_at', 'year', 'month', 'flag']:
                        continue

                    # Require that the column looks like an identifier OR is unique in at least one table
                    is_id_like = any(term in col_lower for term in ['id', 'key', 'code', 'num', 'no'])
                    is_t1_pk = bool(df1[col].is_unique and df1[col].notnull().all())
                    is_t2_pk = bool(df2[col].is_unique and df2[col].notnull().all())

                    if not (is_id_like or is_t1_pk or is_t2_pk):
                        continue

                    s1 = set(df1[col].dropna().unique())
                    s2 = set(df2[col].dropna().unique())
                    overlap = s1.intersection(s2)

                    if len(overlap) > 0:
                        if is_t1_pk and not is_t2_pk:
                            parent_table, child_table = t1_name, t2_name
                            cardinality = "1:N (One-to-Many)"
                            orphans = len(s2 - s1)
                        elif is_t2_pk and not is_t1_pk:
                            parent_table, child_table = t2_name, t1_name
                            cardinality = "1:N (One-to-Many)"
                            orphans = len(s1 - s2)
                        elif is_t1_pk and is_t2_pk:
                            parent_table, child_table = t1_name, t2_name
                            cardinality = "1:1 (One-to-One)"
                            orphans = len(s1.symmetric_difference(s2))
                        elif is_id_like:
                            parent_table, child_table = t1_name, t2_name
                            cardinality = "N:M (Many-to-Many)"
                            orphans = 0
                        else:
                            continue

                        link_sig = (parent_table, child_table, col)
                        if link_sig not in seen_links:
                            seen_links.add(link_sig)
                            relationships.append({
                                "key_column": col,
                                "parent_table": parent_table,
                                "child_table": child_table,
                                "cardinality": cardinality,
                                "shared_values_count": len(overlap),
                                "orphan_records_count": orphans
                            })

        return relationships

    @classmethod
    def profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute full statistical profile and data quality assessment for a single dataframe."""
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        total_nulls = int(df.isnull().sum().sum())
        completeness_pct = round((1 - (total_nulls / total_cells if total_cells > 0 else 0)) * 100, 1)

        dup_count = int(df.duplicated().sum())
        dup_penalty = min(20, int((dup_count / n_rows if n_rows > 0 else 0) * 100))
        quality_score = max(0, min(100, int(completeness_pct * 0.8 - dup_penalty + 20)))

        columns_summary = {}
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        anomalies = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / n_rows * 100) if n_rows > 0 else 0, 1)
            unique_count = int(df[col].nunique(dropna=True))

            col_info = {
                "dtype": dtype,
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_count": unique_count,
            }

            if pd.api.types.is_bool_dtype(df[col]):
                categorical_cols.append(col)
                top_vals = df[col].value_counts(dropna=True).head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in top_vals.items()}

            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0:
                    try:
                        q25 = float(series.quantile(0.25))
                        q75 = float(series.quantile(0.75))
                        iqr = q75 - q25
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())
                    except Exception:
                        outliers_count = 0

                    col_info.update({
                        "min": float(series.min()) if len(series) > 0 else 0.0,
                        "max": float(series.max()) if len(series) > 0 else 0.0,
                        "mean": round(float(series.mean()), 2) if len(series) > 0 else 0.0,
                        "median": round(float(series.median()), 2) if len(series) > 0 else 0.0,
                        "outliers_count": outliers_count,
                    })

                    if outliers_count > 0:
                        pct_out = round(outliers_count / len(series) * 100, 1)
                        if pct_out > 2.0:
                            anomalies.append(f"Column '{col}' has {outliers_count} ({pct_out}%) statistical outliers outside 1.5x IQR.")

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                series = df[col].dropna()
                if len(series) > 0:
                    col_info.update({
                        "min_date": str(series.min()),
                        "max_date": str(series.max()),
                        "span_days": int((series.max() - series.min()).days)
                    })

            else:
                categorical_cols.append(col)
                top_vals = df[col].value_counts(dropna=True).head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in top_vals.items()}

            columns_summary[col] = col_info

        top_correlations = []
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        c1, c2 = numeric_cols[i], numeric_cols[j]
                        val = corr_matrix.loc[c1, c2]
                        if not np.isnan(val) and abs(val) >= 0.5:
                            top_correlations.append({
                                "col1": c1,
                                "col2": c2,
                                "correlation": round(float(val), 2)
                            })
                top_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            except Exception:
                pass

        return {
            "shape": {"rows": n_rows, "columns": n_cols},
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2),
            "quality_score": quality_score,
            "completeness_pct": completeness_pct,
            "duplicates": dup_count,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "columns": columns_summary,
            "anomalies": anomalies,
            "top_correlations": top_correlations[:5]
        }

    @classmethod
    def profile_relational(cls, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Profile multiple relational tables and their join relationships."""
        table_profiles = {}
        total_rows = 0
        total_cols = 0

        for name, df in tables.items():
            prof = cls.profile(df)
            table_profiles[name] = prof
            total_rows += prof["shape"]["rows"]
            total_cols += prof["shape"]["columns"]

        relationships = cls.detect_relationships(tables)

        # Average quality score across tables
        avg_quality = int(np.mean([p["quality_score"] for p in table_profiles.values()]))

        return {
            "is_multi_table": len(tables) > 1,
            "table_count": len(tables),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "quality_score": avg_quality,
            "table_profiles": table_profiles,
            "relationships": relationships
        }

    @classmethod
    def generate_context_card(cls, profile: Dict[str, Any], sample_df: pd.DataFrame) -> str:
        """Create a compact summary prompt string for a single table."""
        shape = profile["shape"]
        card = [
            f"DATASET METADATA:",
            f"- Rows: {shape['rows']:,} | Columns: {shape['columns']}",
            f"- Data Quality Score: {profile['quality_score']}/100 (Completeness: {profile['completeness_pct']}%)",
            f"- Numeric Columns ({len(profile['numeric_columns'])}): {', '.join(profile['numeric_columns'])}",
            f"- Categorical Columns ({len(profile['categorical_columns'])}): {', '.join(profile['categorical_columns'])}",
        ]
        if profile["datetime_columns"]:
            card.append(f"- Datetime Columns: {', '.join(profile['datetime_columns'])}")

        if profile["top_correlations"]:
            corrs = [f"{c['col1']} & {c['col2']} (r={c['correlation']})" for c in profile['top_correlations'][:3]]
            card.append(f"- Strong Correlations: {'; '.join(corrs)}")

        if profile["anomalies"]:
            card.append(f"- Detected Anomalies: {'; '.join(profile['anomalies'][:3])}")

        card.append("\nFIRST 3 ROWS PREVIEW:")
        card.append(sample_df.head(3).to_string(index=False))

        return "\n".join(card)

    @classmethod
    def generate_schema_sample(cls, tables: Dict[str, pd.DataFrame]) -> str:
        """
        Build a concise, crystal-clear schema catalog with exact column names,
        data types, and representative sample non-null values.
        Prevents LLM column hallucination and KeyError failures.
        """
        lines = []
        for tname, df in tables.items():
            lines.append(f"TABLE `{tname}` ({len(df):,} rows, {len(df.columns)} columns):")
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                non_nulls = df[col].dropna()
                if len(non_nulls) > 0:
                    samples = non_nulls.head(2).tolist()
                    formatted_samples = ", ".join(repr(s) if isinstance(s, str) else str(round(s, 2) if isinstance(s, float) else s) for s in samples)
                    sample_str = f" [sample: {formatted_samples}]"
                else:
                    sample_str = " [all null]"
                lines.append(f"  - {col} ({dtype_str}){sample_str}")
            lines.append("")
        return "\n".join(lines).strip()

    @classmethod
    def generate_relational_context_card(cls, rel_profile: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
        """Create a compact, token-efficient relational schema card with ALL columns visible."""
        lines = [
            f"RELATIONAL SCHEMA & MULTI-TABLE DATABASE:",
            f"- Total Tables: {rel_profile['table_count']}",
            f"- Total Aggregate Records: {rel_profile['total_rows']:,}",
            f"- Quality Score: {rel_profile['quality_score']}/100",
            "\nTABLE SCHEMAS & AVAILABLE COLUMNS (Use ONLY these exact names):"
        ]

        for name, prof in rel_profile["table_profiles"].items():
            shape = prof["shape"]
            num_cols = prof.get("numeric_columns", [])
            cat_cols = prof.get("categorical_columns", [])
            dt_cols = prof.get("datetime_columns", [])

            lines.append(f"\n• Table `{name}` ({shape['rows']:,} rows, {shape['columns']} cols):")
            if num_cols:
                lines.append(f"   - Metrics/Numeric: {', '.join(num_cols)}")
            if cat_cols:
                cat_samples = []
                for c in cat_cols:
                    top_vals = prof.get("columns", {}).get(c, {}).get("top_categories", {})
                    if top_vals:
                        sample_str = "/".join(list(top_vals.keys())[:3])
                        cat_samples.append(f"{c} ('{sample_str}')")
                    else:
                        cat_samples.append(c)
                lines.append(f"   - Dimensions/Categorical: {', '.join(cat_samples)}")
            if dt_cols:
                lines.append(f"   - Datetime: {', '.join(dt_cols)}")

        if rel_profile.get("relationships"):
            lines.append("\nDETECTED FOREIGN KEY RELATIONSHIPS (JOIN KEYS):")
            for r in rel_profile["relationships"]:
                lines.append(
                    f"• `{r['parent_table']}.{r['key_column']}` ──[{r['cardinality']}]──> `{r['child_table']}.{r['key_column']}`"
                )
        else:
            lines.append("\n(No auto-detected foreign keys based on matching column names)")

        return "\n".join(lines)

