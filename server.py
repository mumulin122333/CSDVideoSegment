import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from pydantic import BaseModel
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoadProjectRequest(BaseModel):
    csv_path: str
    folder_path: str

class ExportRequest(BaseModel):
    segments: List[Dict]

@app.get("/api/fs/ls")
async def list_directory(path: str = "/"):
    """Lists files and folders in a given path for the integrated file browser."""
    if not os.path.exists(path):
        # Fallback to home directory if path doesn't exist
        path = os.path.expanduser("~")
    
    try:
        items = []
        # Add parent directory option
        parent = os.path.dirname(path)
        if parent != path:
            items.append({"name": "..", "path": parent, "is_dir": True})
            
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name.startswith('.'): continue
                items.append({
                    "name": entry.name,
                    "path": entry.path,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else None
                })
        
        # Sort: directories first, then files
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        return {
            "current_path": os.path.abspath(path),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video")
async def get_video(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Video file not found at {path}")
    return FileResponse(path)

@app.post("/api/project/load")
async def load_project(req: LoadProjectRequest):
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file not found at {req.csv_path}")
    if not os.path.exists(req.folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found at {req.folder_path}")
    
    # Scan folder for videos
    video_extensions = ('.mp4', '.MP4', '.mov', '.avi', '.mkv')
    video_map = {}
    for root, _, files in os.walk(req.folder_path):
        for f in files:
            if f.endswith(video_extensions):
                vid_id = os.path.splitext(f)[0]
                video_map[vid_id] = os.path.join(root, f)
    
    try:
        df = pd.read_csv(req.csv_path)
        segments = []
        for _, row in df.iterrows():
            vid_id = str(row.get('video_id', ''))
            full_video_path = video_map.get(vid_id)
            
            segments.append({
                "phase": row.get('phase', 'Unknown'),
                "start_time": row.get('phase_start_time', '00:00:00'),
                "end_time": row.get('phase_end_time', '00:00:01'),
                "duration": row.get('duration_s', 0.0),
                "video_id": vid_id,
                "video_path": full_video_path,
                "instrument": row.get('instrument_sheath', 'none'),
                "num_instruments": row.get('num_instruments', 0)
            })
            
        return {
            "segments": segments,
            "csv_path": req.csv_path,
            "folder_path": req.folder_path,
            "video_mappings": video_map
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing CSV: {str(e)}")

@app.post("/api/export")
async def export_csv(req: ExportRequest):
    try:
        df = pd.DataFrame(req.segments)
        df_to_save = df.rename(columns={
            "phase": "phase",
            "start_time": "phase_start_time",
            "end_time": "phase_end_time",
            "duration": "duration_s",
            "instrument": "instrument_sheath"
        })
        
        if 'video_path' in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=['video_path'])
            
        output = io.StringIO()
        df_to_save.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=modified_results.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
