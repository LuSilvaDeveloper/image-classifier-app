from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .predict_logic import predict
import io
import os

app = FastAPI()

# Mount the CIFAR-10-images directory
app.mount("/images", StaticFiles(directory="CIFAR-10-images"), name="images")

# Allow CORS (adjust for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = io.BytesIO(await file.read())
        prediction = predict(image_bytes)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test-images")
async def get_test_images():
    test_dir = "CIFAR-10-images/test"
    images = []
    
    # Walk through each category directory
    for category in os.listdir(test_dir):
        category_path = os.path.join(test_dir, category)
        if os.path.isdir(category_path):
            for image in os.listdir(category_path):
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    images.append({
                        "path": f"/images/test/{category}/{image}",
                        "category": category,
                        "filename": image
                    })
    
    return {"images": images}
