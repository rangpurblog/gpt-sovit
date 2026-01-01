import os
import sys
import uuid
import json
import threading
import queue
from datetime import datetime

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "GPT_SoVITS"))

from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
os.makedirs("outputs", exist_ok=True)
os.makedirs("voices", exist_ok=True)
os.makedirs("jobs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

ADMIN_KEY = os.environ.get("ADMIN_KEY", "supersecretadmin")

def admin_auth(x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

# ============ JOB QUEUE SYSTEM ============

job_queue = queue.Queue()
tts_model = None

def get_tts_model():
    global tts_model
    if tts_model is None:
        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights
        
        GPT_MODEL = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        SOVITS_MODEL = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        
        if os.path.exists(GPT_MODEL) and os.path.exists(SOVITS_MODEL):
            change_gpt_weights(GPT_MODEL)
            change_sovits_weights(SOVITS_MODEL)
        
        tts_model = True
    return tts_model

def save_job(job_id, data):
    with open(f"jobs/{job_id}.json", "w") as f:
        json.dump(data, f)

def load_job(job_id):
    path = f"jobs/{job_id}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def worker():
    """Background worker - processes TTS jobs"""
    get_tts_model()
    from GPT_SoVITS.inference_webui import get_tts_wav
    import soundfile as sf
    
    while True:
        job_data = job_queue.get()
        job_id = job_data["job_id"]
        
        try:
            # Update status
            job = load_job(job_id)
            job["status"] = "processing"
            job["started_at"] = datetime.now().isoformat()
            save_job(job_id, job)
            
            # Language mapping
            lang_map = {
                "en": "English", "english": "English",
                "zh": "中文", "chinese": "中文",
                "ja": "日本語", "japanese": "日本語",
                "ko": "한국어", "korean": "한국어",
                "bn": "English"
            }
            tts_lang = lang_map.get(job_data["language"].lower(), "English")
            
            # Generate TTS
            result = get_tts_wav(
                ref_wav_path=job_data["ref_wav"],
                prompt_text="",
                prompt_language=tts_lang,
                text=job_data["text"],
                text_language=tts_lang,
            )
            
            sr, audio_data = next(result)
            sf.write(job_data["out_wav"], audio_data, sr)
            
            # Update status
            job = load_job(job_id)
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
            job["audio_url"] = job_data["out_wav"]
            save_job(job_id, job)
            
            print(f"✅ Job {job_id} completed")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"❌ Job {job_id} failed: {error_msg}")
            
            job = load_job(job_id)
            job["status"] = "failed"
            job["error"] = str(e)
            job["failed_at"] = datetime.now().isoformat()
            save_job(job_id, job)
        
        finally:
            job_queue.task_done()

# Start worker thread
threading.Thread(target=worker, daemon=True).start()

# ============ ENDPOINTS ============

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "GPT-SoVITS API",
        "queue_size": job_queue.qsize()
    }

@app.post("/clone-voice")
async def clone_voice(
    audio: UploadFile,
    user_id: str = Form(...),
    voice_name: str = Form(...)
):
    voice_name_clean = voice_name.lower().replace(" ", "_")
    voice_dir = f"voices/{user_id}/{voice_name_clean}"
    os.makedirs(voice_dir, exist_ok=True)
    
    ref_path = f"{voice_dir}/ref.wav"
    content = await audio.read()
    with open(ref_path, "wb") as f:
        f.write(content)
    
    meta_path = f"{voice_dir}/meta.json"
    with open(meta_path, "w") as f:
        json.dump({"voice_name": voice_name, "user_id": user_id, "public": False}, f)
    
    return {"status": "success", "message": "Voice cloned", "voice_path": ref_path}

@app.post("/tts")
async def tts_submit(
    user_id: str = Form(...),
    voice_name: str = Form(...),
    text: str = Form(...),
    language: str = Form("en")
):
    """Submit TTS job - returns immediately with job_id"""
    voice_name_clean = voice_name.lower().replace(" ", "_")
    ref_wav = f"voices/{user_id}/{voice_name_clean}/ref.wav"
    
    if not os.path.exists(ref_wav):
        raise HTTPException(status_code=404, detail="Voice not found")
    
    job_id = str(uuid.uuid4())
    out_dir = f"outputs/{user_id}/{voice_name_clean}"
    os.makedirs(out_dir, exist_ok=True)
    out_wav = f"{out_dir}/{job_id}.wav"
    
    job_data = {
        "job_id": job_id,
        "user_id": user_id,
        "voice_name": voice_name_clean,
        "text": text,
        "language": language,
        "ref_wav": ref_wav,
        "out_wav": out_wav,
        "status": "queued",
        "created_at": datetime.now().isoformat()
    }
    save_job(job_id, job_data)
    job_queue.put(job_data)
    
    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Job submitted. Poll /tts/status/{job_id} for progress."
    }

@app.get("/tts/status/{job_id}")
def tts_status(job_id: str):
    """Check TTS job status"""
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    resp = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at")
    }
    
    if job["status"] == "queued":
        resp["message"] = "Waiting in queue..."
        resp["queue_position"] = job_queue.qsize()
    elif job["status"] == "processing":
        resp["message"] = "Generating audio..."
        resp["started_at"] = job.get("started_at")
    elif job["status"] == "completed":
        resp["message"] = "Audio ready!"
        resp["audio_url"] = f"/outputs/{job['user_id']}/{job['voice_name']}/{job_id}.wav"
        resp["completed_at"] = job.get("completed_at")
    elif job["status"] == "failed":
        resp["message"] = "Generation failed"
        resp["error"] = job.get("error")
    
    return resp

@app.get("/voices/{user_id}")
def list_voices(user_id: str):
    voice_dir = f"voices/{user_id}"
    if not os.path.exists(voice_dir):
        return {"user_id": user_id, "voices": []}
    
    voices = []
    for v in os.listdir(voice_dir):
        if os.path.isdir(f"{voice_dir}/{v}"):
            voices.append({"voice_id": v, "voice_name": v})
    return {"user_id": user_id, "voices": voices}

@app.post("/delete-voice")
async def delete_voice(user_id: str = Form(...), voice_name: str = Form(...)):
    import shutil
    voice_dir = f"voices/{user_id}/{voice_name.lower().replace(' ', '_')}"
    if os.path.exists(voice_dir):
        shutil.rmtree(voice_dir)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Voice not found")

# Admin endpoints
@app.get("/admin/voices", dependencies=[Depends(admin_auth)])
def admin_list_voices():
    voices = []
    if os.path.exists("voices"):
        for u in os.listdir("voices"):
            upath = f"voices/{u}"
            if os.path.isdir(upath):
                for v in os.listdir(upath):
                    vpath = f"{upath}/{v}"
                    if os.path.isdir(vpath):
                        meta = {"public": False}
                        meta_path = f"{vpath}/meta.json"
                        if os.path.exists(meta_path):
                            with open(meta_path) as f:
                                meta = json.load(f)
                        voices.append({
                            "user_id": u, "voice_id": v, "voice_name": v,
                            "public": meta.get("public", False)
                        })
    return {"voices": voices}

@app.get("/admin/stats", dependencies=[Depends(admin_auth)])
def admin_stats():
    users, voices, public = set(), 0, 0
    if os.path.exists("voices"):
        for u in os.listdir("voices"):
            upath = f"voices/{u}"
            if os.path.isdir(upath):
                users.add(u)
                for v in os.listdir(upath):
                    if os.path.isdir(f"{upath}/{v}"):
                        voices += 1
                        meta_path = f"{upath}/{v}/meta.json"
                        if os.path.exists(meta_path):
                            with open(meta_path) as f:
                                if json.load(f).get("public"): public += 1
    return {"users": len(users), "voices": voices, "public_voices": public, "queue_size": job_queue.qsize()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
