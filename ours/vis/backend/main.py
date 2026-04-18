import sys
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_runner import AgentRunner, AGENT_BASE_DIR
from typing import Dict, Any, List, Optional
app = FastAPI()

# Ensure history directory exists
HISTORY_DIR = "/home/gujing/Graduation_Design/ours/vis/data/history"
os.makedirs(HISTORY_DIR, exist_ok=True)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.agent_runner = AgentRunner(self.broadcast)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/library")
def get_library():
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
    
    agent_type = manager.agent_runner.config_state["agent_type"]
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
    if os.path.exists(HISTORY_DIR):
        for filename in os.listdir(HISTORY_DIR):
            if filename.endswith(".json"):
                path = os.path.join(HISTORY_DIR, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        history_list.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Unknown Chat"),
                            "timestamp": data.get("timestamp")
                        })
                except:
                    pass
    history_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history_list

@app.get("/api/history/{conv_id}")
def get_conversation(conv_id: str):
    path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"error": "Not found"}

@app.post("/api/history/save")
def save_conversation(data: Dict[str, Any]):
    conv_id = data.get("id")
    if not conv_id:
        return {"error": "No ID provided"}
    path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"status": "success"}

@app.delete("/api/history/{conv_id}")
def delete_conversation(conv_id: str):
    path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return {"status": "success"}
    return {"error": "Not found"}

@app.post("/api/feedback")
async def submit_feedback(data: Dict[str, Any]):
    print(f"--- Feedback Received: ID={data.get('msg_id')} | Status={data.get('status')} ---")
    await manager.agent_runner.handle_feedback(data.get("status"), data.get("msg_id"))
    return {"status": "success"}

@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "config":
                # Update config in agent runner
                await manager.agent_runner.update_config(payload.get("data", {}))
                
            elif payload.get("type") == "clear_memory":
                # Clear short-term memory
                await manager.agent_runner.clear_memory()
                
            elif payload.get("type") == "query":
                # User sent a message, start agent processing
                asyncio.create_task(manager.agent_runner.process_query(payload.get("text", "")))
                
            elif payload.get("type") == "load_conversation":
                # Restore backend state
                await manager.agent_runner.load_conversation(payload.get("data", {}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
