from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io


MODEL_PATH = "model/model.h5"
CLASS_NAMES = ['Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Orange',
               'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate',
               'Tomatoes', 'muskmelon']


app = FastAPI(title="Fruit Classifier API", version="1.0.0")
model = load_model(MODEL_PATH)  # load 1 lần khi khởi động


def preprocess_image(file_bytes: bytes, target_size=(160, 160)) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {e}")

    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img)              # shape (H, W, 3), dtype=uint8
    # Nếu model của bạn cần chuẩn hóa, mở comment dòng dưới:
    # arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict_class(image_batch: np.ndarray) -> str:
    preds = model.predict(image_batch)
    scores = tf.nn.sigmoid(preds).numpy()
    label_idx = int(np.argmax(scores, axis=-1)[0])
    return CLASS_NAMES[label_idx]


@app.get("/")
def root():
    return {"status": "ok", "message": "Fruit classifier is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ JPEG/PNG")

    file_bytes = await file.read()
    img_batch = preprocess_image(file_bytes, target_size=(160, 160))
    class_name = predict_class(img_batch)
    return JSONResponse({"class_name": class_name})
