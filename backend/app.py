# backend/app.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
from inference import load_model, predict_video, DEVICE
import uvicorn
import shutil

MODEL_PATH = os.environ.get('MODEL_PATH', 'checkpoints/best_model.pth')
ARCH = os.environ.get('ARCH', 'rnn')   # or '3d'
ARCH_3D = os.environ.get('ARCH_3D', 'r3d_18')
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, arch=ARCH, arch_3d=ARCH_3D)
        print("Model loaded from", MODEL_PATH)
    else:
        print("Warning: model path does not exist:", MODEL_PATH)
except Exception as e:
    print("Failed to load model:", e)

app = FastAPI()
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != 'video':
        raise HTTPException(status_code=400, detail="Upload a video file.")
    file_id = f"{uuid.uuid4().hex}_{file.filename}"
    dest = os.path.join(UPLOAD_DIR, file_id)
    with open(dest, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    try:
        res = predict_video(dest, model, num_frames=16)
        # Optionally remove file
        # os.remove(dest)
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
