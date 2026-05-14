import os
import json
import asyncio
import re
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent_runner import AgentRunner
from typing import Dict, Any
app = FastAPI()

# Ensure history directory exists
BASE_DIR = Path("/home/gujing/Graduation_Design/ours/vis")
HISTORY_DIR = BASE_DIR / "data" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

def history_path(conv_id: str) -> Path:
    if not conv_id or not SAFE_ID_RE.fullmatch(conv_id):
        raise HTTPException(status_code=400, detail="Invalid conversation id")
    return HISTORY_DIR / f"{conv_id}.json"

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.agent_runners: dict[WebSocket, AgentRunner] = {}
        self.feedback_runner = AgentRunner(self.noop_broadcast)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.agent_runners[websocket] = AgentRunner(self.make_sender(websocket))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.agent_runners.pop(websocket, None)

    async def noop_broadcast(self, message: dict):
        return

    def make_sender(self, websocket: WebSocket):
        async def send(message: dict):
            payload = json.dumps(message, ensure_ascii=False)
            try:
                await websocket.send_text(payload)
            except Exception:
                self.disconnect(websocket)
        return send

    def runner_for(self, websocket: WebSocket) -> AgentRunner:
        return self.agent_runners[websocket]

manager = ConnectionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/library")
def get_library(agent_type: str = "react"):
    def load_json(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        return {}

    cloud_tool = load_json("/home/gujing/Graduation_Design/ours/vis/data/cloud_tool_db.json")
    edge_tool = load_json("/home/gujing/Graduation_Design/ours/vis/data/edge_tool_db.json")
    
    if agent_type not in {"react", "llmcompiler"}:
        agent_type = "react"
    db_filename = "react_experience_db.json" if agent_type == "react" else "llm_compiler_experience_db.json"
    exp_db_path = f"/home/gujing/Graduation_Design/ours/vis/data/{db_filename}"
    exp_db_records = load_json(exp_db_path) if os.path.exists(exp_db_path) else []

    def get_recs(db):
        if not isinstance(db, dict): return {}
        return db

    exp_dict = {}
    if isinstance(exp_db_records, list):
        for rec in exp_db_records:
            q = rec.get("query", "Unknown Query")
            exp_dict[q] = rec

    return {
        "cloud_tool": get_recs(cloud_tool),
        "edge_tool": get_recs(edge_tool),
        "exp_db": exp_dict,
        "active_agent": agent_type
    }

@app.get("/api/history")
def get_all_history():
    history_list = []
    for path in HISTORY_DIR.glob("*.json"):
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                history_list.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Unknown Chat"),
                    "timestamp": data.get("timestamp")
                })
        except Exception:
            pass
    history_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history_list

@app.get("/api/history/{conv_id}")
def get_conversation(conv_id: str):
    path = history_path(conv_id)
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.post("/api/history/save")
def save_conversation(data: Dict[str, Any]):
    conv_id = data.get("id")
    path = history_path(conv_id)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"status": "success"}

@app.delete("/api/history/{conv_id}")
def delete_conversation(conv_id: str):
    path = history_path(conv_id)
    if path.exists():
        path.unlink()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.post("/api/feedback")
async def submit_feedback(data: Dict[str, Any]):
    print(f"--- Feedback Received: ID={data.get('msg_id')} | Status={data.get('status')} ---")
    await manager.feedback_runner.handle_feedback(data.get("status"), data.get("msg_id"))
    return {"status": "success"}

@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    runner = manager.runner_for(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON payload"})
                continue
            
            if payload.get("type") == "config":
                # Update config in agent runner
                await runner.update_config(payload.get("data", {}))
                
            elif payload.get("type") == "clear_memory":
                # Clear short-term memory
                await runner.clear_memory()
                
            elif payload.get("type") == "query":
                # User sent a message, start agent processing
                asyncio.create_task(runner.process_query(payload.get("text", "")))
                
            elif payload.get("type") == "load_conversation":
                # Restore backend state
                await runner.load_conversation(payload.get("data", {}))

            elif payload.get("type") == "feedback":
                await runner.handle_feedback(payload.get("status"), payload.get("msg_id"))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8011)
