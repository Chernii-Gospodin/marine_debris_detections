from fastapi import FastAPI, File, UploadFile, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fasterrcnn import TorchFasterrcnnModel
import os
import numpy as np
import cv2
from uuid import uuid4
import uvicorn
from fastapi import Form
import torch

model = TorchFasterrcnnModel()
app = FastAPI()

@app.post("/detect")
async def detect_trash(background_tasks: BackgroundTasks,
                       file: UploadFile = File(...),
                       treshold: float = Form(0.3)):

    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "File must be an image"})
    file_path = f"uploads/{file.filename}"
    photo = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(photo)
    img_tensor = model.image2tensors(file_path)
    if img_tensor.size()[0] == 4:
        img_tensor = img_tensor[:-1]

    boxes = model.predict(img_tensor)
    model.save_image(img_tensor, boxes, saved_path="uploads/predict.jpg", score_treshold=treshold)

    background_tasks.add_task(os.remove, file_path)
    background_tasks.add_task(os.remove, "uploads/predict.jpg")
    return FileResponse("uploads/predict.jpg")

@app.websocket("/thread")
async def process_thread(ws: WebSocket):
    await ws.accept()
    print("Connection set")    
    try:
        while True:
            data = await ws.receive_bytes()
            array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame_name = uuid4()
                cv2.imwrite(f"uploads/{frame_name}.jpg", frame)
                boxes = model.predict(torch.frombuffer(data))
                model.save_image(torch.from_file(f"uploads/{frame_name}.jpg"), boxes, saved_path=f"uploads/pred_{frame_name}.jpg")
                _, buffer = cv2.imencode('.jpg', cv2.imread(f"uploads/pred_{frame_name}.jpg"))
                os.remove(f"uploads/{frame_name}.jpg")
                os.remove(f"uploads/pred_{frame_name}.jpg")
                
                await ws.send_bytes(buffer.tobytes())
                
    except Exception as e:
        print(e)
        
if __name__ == "__main__":
    uvicorn.run(app, port=8000)