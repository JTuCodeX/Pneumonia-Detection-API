from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from app.inference import predict_image

app = FastAPI(
    title="Pneumonia Detection API",
    description="FastAPI service for pneumonia detection from chest X-ray images.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type. Please upload an image."}
        )

    # Read file bytes
    image_bytes = await file.read()

    # Run inference
    try:
        result = predict_image(image_bytes)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
