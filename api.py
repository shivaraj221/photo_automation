import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pipeline import process_passport_photo
from config import PASSPORT_CONFIG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("output", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Mount directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(status_code=204)

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/process")
async def process_photo(
    file: UploadFile = File(...),
    copies: int = Form(4),
    bg_color_hex: str = Form("#FFFFFF"),
    force_ai: bool = Form(False)
):
    temp_path = Path("temp") / file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        p_cfg = PASSPORT_CONFIG.copy()
        bg_rgb = tuple(int(bg_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        p_cfg["background_color"] = (bg_rgb[2], bg_rgb[1], bg_rgb[0])
        
        results = process_passport_photo(
            input_path=str(temp_path), 
            copies=copies, 
            force_ai=force_ai, 
            passport_config=p_cfg
        )
        
        # Results returns absolute paths. We need relative web paths.
        passport_path = Path(results["passport_path"]).name
        sheet_path = Path(results["sheet_path"]).name
        
        passport_url = f"/output/{passport_path}"
        sheet_url = f"/output/{sheet_path}"
        
        return {
            "success": True,
            "passport_url": passport_url,
            "sheet_url": sheet_url
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass
