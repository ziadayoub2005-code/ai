from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ai_engine import predict_image, load_model
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def read_root():
    return {"status": "Can2Coin AI Engine Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
    
    contents = await file.read()
    result = predict_image(contents)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
