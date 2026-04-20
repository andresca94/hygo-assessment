from __future__ import annotations

from fastapi import FastAPI, File, UploadFile

from ml.inference.predictor import AgeSafetyPredictor
from ml.inference.schemas import BatchResult, HealthResponse, ImageResult

app = FastAPI(title="Age Safety Inference Service", version="1.0.0")
predictor = AgeSafetyPredictor()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(**predictor.health())


@app.post("/v1/age-safety/check", response_model=ImageResult)
async def check_image(file: UploadFile = File(...)) -> ImageResult:
    payload = predictor.predict_image(await file.read())
    return ImageResult(**payload)


@app.post("/v1/age-safety/check-batch", response_model=BatchResult)
async def check_batch(files: list[UploadFile] = File(...)) -> BatchResult:
    items = [ImageResult(**predictor.predict_image(await upload.read())) for upload in files]
    return BatchResult(
        items=items,
        safe_count=sum(item.verdict == "safe" for item in items),
        flagged_count=sum(item.verdict == "flagged" for item in items),
        uncertain_count=sum(item.verdict == "uncertain" for item in items),
    )
