from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict_logic import predict
import io

app = FastAPI()

# Allow CORS (adjust for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = io.BytesIO(await file.read())
    prediction = predict(image_bytes)
    return {"prediction": prediction}
