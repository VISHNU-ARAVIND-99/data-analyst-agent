import io
import json
from pathlib import Path
from typing import List
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CHARTS_DIR, EXPORTS_DIR, MODEL_NAME
from core.llm_client import test_connection
from core.profiler import DataProfiler
from core.agent import DataAnalystAgent
from core.reporter import ExecutivePDFReport

app = FastAPI(title="Autonomous Data Analyst Agent API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Session State
app_state = {
    "tables_raw": {},
    "tables_clean": {},
    "dataset_name": None,
    "profile": None,
    "transformations": {},
    "briefing": None,
    "chat_history": [],
    "all_charts": [],
    "session_notes": []   # Rolling list of key discoveries (max 6)
}

agent = DataAnalystAgent()

# Ensure directories exist
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static and charts directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Data Analyst Agent API is running. UI index.html not found."}

@app.get("/api/status")
def get_status():
    status = test_connection()
    return {
        "status": "online" if status["connected"] else "offline",
        "model": MODEL_NAME,
        "ollama": status,
        "active_dataset": app_state["dataset_name"],
        "has_data": bool(app_state["tables_clean"])
    }

def sse_keepalive_wrapper(inner_generator_fn, interval_seconds: float = 5.0):
    """
    Wraps an SSE generator function to emit ': keepalive\n\n' pings every `interval_seconds`
    whenever the generator is blocked (e.g. during long local LLM CPU prompt evaluations).
    Prevents browser, proxy, and HTTP client connection timeouts.
    """
    import queue
    import threading

    q = queue.Queue()
    done = threading.Event()

    def worker():
        try:
            for item in inner_generator_fn():
                q.put(("item", item))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while not done.is_set() or not q.empty():
        try:
            msg_type, payload = q.get(timeout=interval_seconds)
            if msg_type == "item":
                yield payload
            elif msg_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'detail': payload})}\n\n"
        except queue.Empty:
            if not done.is_set():
                yield ": keepalive\n\n"

def stream_profiling_and_briefing(new_tables: dict, source_name: str, names: list):
    """
    Generator yielding real-time SSE events during dataset onboarding & profiling:
    1. Ingestion & File Parsing
    2. Automated Cleaning
    3. Statistical Profiling
    4. Relational Key & DuckDB Catalog Discovery
    5. Local Qwen 2.5:7B Proactive Executive Briefing Synthesis
    """
    import time
    t_start = time.time()

    # Step 1: Ingestion & File Parsing
    yield {
        "type": "onboarding_step",
        "step_idx": 1,
        "step_name": "1. Ingestion & File Parsing",
        "status": "running",
        "detail": f"Parsing {len(new_tables)} dataset table(s)..."
    }
    table_names = list(new_tables.keys())
    raw_rows = sum(len(df) for df in new_tables.values())
    raw_cols = sum(len(df.columns) for df in new_tables.values())
    yield {
        "type": "log",
        "level": "INGEST",
        "msg": f"Ingested {len(new_tables)} table(s) ({', '.join(table_names)}) totaling {raw_rows:,} records across {raw_cols} columns."
    }
    yield {
        "type": "onboarding_step",
        "step_idx": 1,
        "step_name": "1. Ingestion & File Parsing",
        "status": "completed",
        "detail": f"Parsed {len(new_tables)} table(s) successfully."
    }

    # Step 2: Automated Data Cleaning
    yield {
        "type": "onboarding_step",
        "step_idx": 2,
        "step_name": "2. Automated Data Cleaning",
        "status": "running",
        "detail": "Standardizing column headers, formats & duplicates..."
    }
    t_clean_start = time.time()
    cleaned_tables, transforms = DataProfiler.clean_tables(new_tables)
    clean_duration = round(time.time() - t_clean_start, 3)
    total_transforms = sum(len(tr) for tr in transforms.values())
    
    yield {
        "type": "log",
        "level": "CLEAN",
        "msg": f"Applied {total_transforms} cleaning transformations in {clean_duration}s (snake_case headers, date conversions, trimmed spaces)."
    }
    yield {
        "type": "onboarding_step",
        "step_idx": 2,
        "step_name": "2. Automated Data Cleaning",
        "status": "completed",
        "detail": f"Cleaned in {clean_duration}s ({total_transforms} transforms)."
    }

    # Step 3: Statistical Profiling
    yield {
        "type": "onboarding_step",
        "step_idx": 3,
        "step_name": "3. Statistical Profiling",
        "status": "running",
        "detail": "Computing metrics, distributions & health index..."
    }
    t_prof_start = time.time()
    if len(cleaned_tables) > 1:
        profile = DataProfiler.profile_relational(cleaned_tables)
    else:
        first_name = list(cleaned_tables.keys())[0]
        single_prof = DataProfiler.profile(cleaned_tables[first_name])
        single_prof["is_multi_table"] = False
        profile = single_prof

    prof_duration = round(time.time() - t_prof_start, 3)
    q_score = profile.get("quality_score", 100)

    if profile.get("is_multi_table"):
        for tname, tprof in profile.get("table_profiles", {}).items():
            yield {
                "type": "log",
                "level": "PROFILE",
                "msg": f"Table '{tname}': {tprof['shape']['rows']:,} rows, {tprof['shape']['columns']} cols | Completeness: {tprof.get('completeness_pct', 100)}% | Quality: {tprof.get('quality_score', 95)}/100."
            }
    else:
        yield {
            "type": "log",
            "level": "PROFILE",
            "msg": f"Single table: {profile['shape']['rows']:,} rows, {profile['shape']['columns']} cols | Completeness: {profile.get('completeness_pct', 100)}% | Quality: {q_score}/100."
        }

    yield {
        "type": "onboarding_step",
        "step_idx": 3,
        "step_name": "3. Statistical Profiling",
        "status": "completed",
        "detail": f"Health index: {q_score}/100 in {prof_duration}s."
    }

    # Step 4: Relational Foreign Key Discovery & DuckDB Catalog
    yield {
        "type": "onboarding_step",
        "step_idx": 4,
        "step_name": "4. Relational Key Discovery",
        "status": "running",
        "detail": "Discovering primary/foreign keys & registering DuckDB..."
    }
    relationships = profile.get("relationships", [])
    if relationships:
        for r in relationships:
            yield {
                "type": "log",
                "level": "SCHEMA",
                "msg": f"Discovered Relational Join: {r['parent_table']}.{r['key_column']} -> {r['child_table']}.{r['key_column']} ({r['cardinality']}, {r['shared_values_count']} shared keys)."
            }
    else:
        yield {
            "type": "log",
            "level": "SCHEMA",
            "msg": "Single dataset mounted into DuckDB runtime environment."
        }

    yield {
        "type": "log",
        "level": "DUCKDB",
        "msg": f"Pre-registered {len(cleaned_tables)} table(s) in DuckDB in-memory session."
    }
    yield {
        "type": "onboarding_step",
        "step_idx": 4,
        "step_name": "4. Relational Key Discovery",
        "status": "completed",
        "detail": f"Mapped {len(relationships)} entity join(s)." if relationships else "Mounted in DuckDB."
    }

    # Save to global state
    app_state["tables_raw"] = new_tables
    app_state["tables_clean"] = cleaned_tables
    app_state["dataset_name"] = source_name
    app_state["profile"] = profile
    app_state["transformations"] = transforms
    app_state["chat_history"] = []
    app_state["all_charts"] = []
    app_state["session_notes"] = []

    # Step 5: Local Qwen 2.5:7B Proactive Executive Briefing Synthesis
    yield {
        "type": "onboarding_step",
        "step_idx": 5,
        "step_name": "5. Local LLM Executive Briefing",
        "status": "running",
        "detail": f"Local {MODEL_NAME} is auditing schema & formulating hypotheses..."
    }
    yield {
        "type": "log",
        "level": "LLM",
        "msg": f"Prompting local {MODEL_NAME} via Ollama to formulate executive briefing and clickable exploration hypotheses..."
    }

    t_brief_start = time.time()
    target_data = cleaned_tables if profile.get("is_multi_table") else list(cleaned_tables.values())[0]
    briefing = agent.generate_proactive_briefing(profile, target_data)
    app_state["briefing"] = briefing
    brief_duration = round(time.time() - t_brief_start, 1)

    yield {
        "type": "onboarding_step",
        "step_idx": 5,
        "step_name": "5. Local LLM Executive Briefing",
        "status": "completed",
        "detail": f"Executive briefing & hypotheses formulated in {brief_duration}s."
    }

    total_duration = round(time.time() - t_start, 1)
    yield {
        "type": "log",
        "level": "DONE",
        "msg": f"Dataset onboarding and profiling completed in {total_duration}s. System ready for analysis."
    }

    result_data = {
        "dataset_name": app_state["dataset_name"],
        "profile": profile,
        "briefing": briefing,
        "tables": list(cleaned_tables.keys()),
        "duration": total_duration
    }
    yield {
        "type": "onboarding_result",
        "payload": result_data
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    new_tables = {}
    names = []
    for f in files:
        names.append(f.filename)
        tbl_name = Path(f.filename).stem.lower().replace(" ", "_").replace("-", "_")
        content = await f.read()
        try:
            if f.filename.endswith(".csv"):
                new_tables[tbl_name] = pd.read_csv(io.BytesIO(content))
            else:
                new_tables[tbl_name] = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse {f.filename}: {e}")

    if not new_tables:
        raise HTTPException(status_code=400, detail="No valid data files provided.")

    source_name = ", ".join(names)
    last_payload = None
    for event in stream_profiling_and_briefing(new_tables, source_name, names):
        if event.get("type") == "onboarding_result":
            last_payload = event.get("payload")
    return last_payload


@app.post("/api/upload-stream")
async def upload_files_stream(files: List[UploadFile] = File(...)):
    new_tables = {}
    names = []
    for f in files:
        names.append(f.filename)
        tbl_name = Path(f.filename).stem.lower().replace(" ", "_").replace("-", "_")
        content = await f.read()
        try:
            if f.filename.endswith(".csv"):
                new_tables[tbl_name] = pd.read_csv(io.BytesIO(content))
            else:
                new_tables[tbl_name] = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse {f.filename}: {e}")

    if not new_tables:
        raise HTTPException(status_code=400, detail="No valid data files provided.")

    source_name = ", ".join(names)

    def event_generator():
        try:
            for event in stream_profiling_and_briefing(new_tables, source_name, names):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        sse_keepalive_wrapper(event_generator),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
def execute_query(req: QueryRequest):
    if not app_state["tables_clean"]:
        raise HTTPException(status_code=400, detail="No active dataset loaded.")

    target_data = app_state["tables_clean"] if app_state["profile"].get("is_multi_table") else list(app_state["tables_clean"].values())[0]
    
    history = [{"role": "user", "content": h["query"]} for h in app_state["chat_history"]]
    
    try:
        result = agent.answer_query(
            query=req.query,
            data=target_data,
            profile=app_state["profile"],
            history=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Map chart absolute paths to web URLs (/charts/filename.png)
    web_charts = []
    for cp in result.get("charts", []):
        fn = Path(cp).name
        web_charts.append(f"/charts/{fn}")
        if cp not in app_state["all_charts"]:
            app_state["all_charts"].append(cp)

    result_payload = {
        "success": result.get("success", False),
        "query": req.query,
        "code": result.get("code", ""),
        "stdout": result.get("stdout", ""),
        "charts": web_charts,
        "answer": result.get("answer", ""),
        "retries": result.get("retries", 0),
        "telemetry": result.get("telemetry", {})
    }

    app_state["chat_history"].append(result_payload)
    return result_payload

@app.post("/api/query-stream")
def execute_query_stream(req: QueryRequest):
    if not app_state["tables_clean"]:
        raise HTTPException(status_code=400, detail="No active dataset loaded.")

    target_data = app_state["tables_clean"] if app_state["profile"].get("is_multi_table") else list(app_state["tables_clean"].values())[0]
    # Build proper alternating user/assistant history (last 5 turns = 10 messages max)
    history = []
    for h in app_state["chat_history"][-5:]:
        history.append({"role": "user", "content": h["query"]})
        if h.get("answer"):
            history.append({"role": "assistant", "content": h["answer"][:300]})  # truncate to save tokens


    # Build formatted session memory from rolling discoveries log
    session_memory = None
    if app_state["session_notes"]:
        session_memory = "\n".join(
            f"  {i+1}. {note}" for i, note in enumerate(app_state["session_notes"])
        )

    def event_generator():
        try:
            for event in agent.answer_query_stream(
                query=req.query,
                data=target_data,
                profile=app_state["profile"],
                history=history,
                session_memory=session_memory
            ):
                if event.get("type") == "result":
                    res = event.get("payload", {})
                    web_charts = []
                    for cp in res.get("charts", []):
                        fn = Path(cp).name
                        web_charts.append(f"/charts/{fn}")
                        if cp not in app_state["all_charts"]:
                            app_state["all_charts"].append(cp)
                    res["charts"] = web_charts
                    app_state["chat_history"].append(res)

                    # Update rolling session notes with this query's key discovery
                    mem_note = res.get("memory_note", "")
                    if mem_note and res.get("success"):
                        entry = f"Q: '{req.query[:50]}' → {mem_note}"
                        app_state["session_notes"].append(entry)
                        # Keep rolling window of last 6 discoveries
                        app_state["session_notes"] = app_state["session_notes"][-6:]
                    # Cap chat history to last 20 entries to prevent unbounded growth
                    app_state["chat_history"] = app_state["chat_history"][-20:]

                    event["payload"] = res

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            err_event = {
                "type": "error",
                "detail": str(e)
            }
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(
        sse_keepalive_wrapper(event_generator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/chat/clear")
def clear_chat():
    app_state["chat_history"] = []
    app_state["all_charts"] = []
    app_state["session_notes"] = []
    return {"status": "ok", "message": "Chat history, charts and session memory cleared."}

@app.post("/api/session/reset")
def reset_session():
    app_state["tables_raw"] = {}
    app_state["tables_clean"] = {}
    app_state["dataset_name"] = None
    app_state["profile"] = None
    app_state["transformations"] = {}
    app_state["briefing"] = None
    app_state["chat_history"] = []
    app_state["all_charts"] = []
    app_state["session_notes"] = []
    return {"status": "ok", "message": "Session reset successfully."}


@app.get("/api/state")
def get_current_state():
    return {
        "dataset_name": app_state["dataset_name"],
        "has_data": bool(app_state["tables_clean"]),
        "profile": app_state["profile"],
        "briefing": app_state["briefing"],
        "tables": list(app_state["tables_clean"].keys()),
        "chat_history": [
            {
                "query": h["query"],
                "code": h.get("code", ""),
                "stdout": h.get("stdout", ""),
                "charts": [f"/charts/{Path(c).name}" for c in h.get("charts", [])],
                "answer": h.get("answer", ""),
                "retries": h.get("retries", 0),
                "telemetry": h.get("telemetry", {})
            }
            for h in app_state["chat_history"]
        ]
    }

@app.get("/api/table/{name}/preview")
def preview_table(name: str):
    if name not in app_state["tables_clean"]:
        raise HTTPException(status_code=404, detail=f"Table '{name}' not found.")
    df = app_state["tables_clean"][name]
    # Return columns, dtypes, and first 25 rows
    return {
        "table": name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "records": json.loads(df.head(25).to_json(orient="records", date_format="iso"))
    }

@app.get("/api/table/{name}/export")
def export_table(name: str):
    if name not in app_state["tables_clean"]:
        raise HTTPException(status_code=404, detail=f"Table '{name}' not found.")
    df = app_state["tables_clean"][name]
    csv_str = df.to_csv(index=False)
    return JSONResponse(
        content={"csv": csv_str, "filename": f"clean_{name}.csv"}
    )

@app.post("/api/export-pdf")
def export_pdf():
    if not app_state["tables_clean"]:
        raise HTTPException(status_code=400, detail="No active dataset.")
    
    report_filename = f"report_{Path(app_state['dataset_name']).stem[:20]}.pdf"
    insights_summary = [
        {"question": item.get("query", ""), "answer": item.get("answer", "")}
        for item in app_state["chat_history"]
    ]
    exec_headline = (app_state["briefing"] or {}).get(
        "executive_headline", "Data intelligence and strategic analytics summary."
    )

    report_path = ExecutivePDFReport.generate(
        filename=report_filename,
        dataset_name=app_state["dataset_name"],
        profile=app_state["profile"],
        executive_summary=exec_headline,
        chat_insights=insights_summary,
        charts=app_state["all_charts"],
        transformations=app_state["transformations"]
    )

    return FileResponse(
        str(report_path),
        media_type="application/pdf",
        filename=report_filename
    )
