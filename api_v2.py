import os
import sys
import uuid
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
os.makedirs("outputs", exist_ok=True)
os.makedirs("voices", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

ADMIN_KEY = os.environ.get("ADMIN_KEY", "supersecretadmin")

# Admin auth
def admin_auth(x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

# GPT-SoVITS inference
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
import soundfile as sf
import numpy as np

# Load default models (first time)
GPT_MODEL = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
SOVITS_MODEL = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"

change_gpt_weights(GPT_MODEL)
change_sovits_weights(SOVITS_MODEL)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "GPT-SoVITS Voice Clone API",
        "endpoints": [
            "POST /clone-voice",
            "POST /tts",
            "GET /voices/{user_id}",
            "POST /delete-voice",
            "GET /admin/stats",
            "GET /admin/voices"
        ]
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
    
    # Save metadata
    import json
    meta_path = f"{voice_dir}/meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "voice_name": voice_name,
            "user_id": user_id,
            "public": False
        }, f)
    
    return {
        "status": "success",
        "message": "Voice cloned",
        "voice_path": ref_path
    }


@app.post("/tts")
async def text_to_speech(
    user_id: str = Form(...),
    voice_name: str = Form(...),
    text: str = Form(...),
    language: str = Form("en")
):
    voice_name_clean = voice_name.lower().replace(" ", "_")
    ref_wav = f"voices/{user_id}/{voice_name_clean}/ref.wav"
    
    if not os.path.exists(ref_wav):
        raise HTTPException(status_code=404, detail=f"Voice not found: {ref_wav}")
    
    job_id = str(uuid.uuid4())
    out_dir = f"outputs/{user_id}/{voice_name_clean}"
    os.makedirs(out_dir, exist_ok=True)
    out_wav = f"{out_dir}/{job_id}.wav"
    
    try:
        # 🔥 FIXED: GPT-SoVITS uses full language names
        lang_map = {
            "en": "English",
            "english": "English",
            "zh": "中文",
            "chinese": "中文",
            "ja": "日本語",
            "japanese": "日本語",
            "ko": "한국어",
            "korean": "한국어",
            "yue": "粤语",
            "cantonese": "粤语",
            "bn": "English",  # Bengali fallback to English
        }
        tts_lang = lang_map.get(language.lower(), "English")
        
        from GPT_SoVITS.inference_webui import get_tts_wav
        import soundfile as sf
        
        result = get_tts_wav(
            ref_wav_path=ref_wav,
            prompt_text="",
            prompt_language=tts_lang,
            text=text,
            text_language=tts_lang,
        )
        
        sr, audio_data = next(result)
        sf.write(out_wav, audio_data, sr)
        
        return {
            "status": "success",
            "output": out_wav,
            "audio_url": f"/outputs/{user_id}/{voice_name_clean}/{job_id}.wav"
        }
        
    except Exception as e:
        import traceback
        print(f"TTS Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices/{user_id}")
def list_voices(user_id: str):
    voice_dir = f"voices/{user_id}"
    if not os.path.exists(voice_dir):
        return {"user_id": user_id, "voices": []}
    
    voices = []
    for v in os.listdir(voice_dir):
        vpath = f"{voice_dir}/{v}"
        if os.path.isdir(vpath):
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
    import json
    voices = []
    if os.path.exists("voices"):
        for u in os.listdir("voices"):
            upath = f"voices/{u}"
            if os.path.isdir(upath):
                for v in os.listdir(upath):
                    vpath = f"{upath}/{v}"
                    if os.path.isdir(vpath):
                        meta_path = f"{vpath}/meta.json"
                        meta = {"public": False}
                        if os.path.exists(meta_path):
                            with open(meta_path) as f:
                                meta = json.load(f)
                        voices.append({
                            "user_id": u,
                            "voice_id": v,
                            "voice_name": v,
                            "public": meta.get("public", False)
                        })
    return {"voices": voices}

@app.get("/admin/stats", dependencies=[Depends(admin_auth)])
def admin_stats():
    users, voices, public = set(), 0, 0
    import json
    if os.path.exists("voices"):
        for u in os.listdir("voices"):
            upath = f"voices/{u}"
            if os.path.isdir(upath):
                users.add(u)
                for v in os.listdir(upath):
                    vpath = f"{upath}/{v}"
                    if os.path.isdir(vpath):
                        voices += 1
                        meta_path = f"{vpath}/meta.json"
                        if os.path.exists(meta_path):
                            with open(meta_path) as f:
                                if json.load(f).get("public"):
                                    public += 1
    return {"users": len(users), "voices": voices, "public_voices": public}

if __name__ == "__main__":
    import uvicorn
    # টাইমআউট সেটিংস: কোনো লিমিট নেই, লম্বা টেক্সটের জন্য
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        timeout_keep_alive=0,           # Keep-alive timeout disabled
        timeout_graceful_shutdown=None, # No graceful shutdown timeout
        limit_concurrency=None,         # No concurrency limit
        backlog=2048                    # Increase backlog for better handling
    )
