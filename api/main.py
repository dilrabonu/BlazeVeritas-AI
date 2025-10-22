from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image
import io, uuid

from .settings import settings
from .schemas import DetectRequest, DetectResponse, PlanRequest, CalibrationReport
from .inference import engine
from .rag.retrieve import generate_plan

app = FastAPI(title="BlazeVeritas AI — API")

# static files (for grad-cam image display)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<html>
  <head><title>BlazeVeritas API</title></head>
  <body style="font-family: system-ui; line-height:1.5; padding:24px;">
    <h2>BlazeVeritas AI — API is running </h2>
    <ul>
      <li><a href="/health">/health</a></li>
      <li><a href="/docs">/docs</a></li>
      <li><a href="/redoc">/redoc</a></li>
    </ul>
    <p>Use the Streamlit dashboard: <code>streamlit run app.py</code></p>
  </body>
</html>
"""

@app.post("/v1/detect", response_model=DetectResponse)
async def detect(req: DetectRequest = None, file: UploadFile = File(None)):
    if (req is None or req.url is None) and file is None:
        raise HTTPException(status_code=400, detail="Provide file or url")

    if file is not None:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    else:
        img = engine.fetch_image(req.url)

    label, prob, uncertainty, overlay = engine.predict(img)
    event_id = uuid.uuid4().hex
    outpath = settings.XAI_DIR / f"{event_id}_gradcam.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(outpath)

    grad_cam_url = f"reports/xai/{outpath.name}"
    return DetectResponse(
        label=label,
        prob=prob,
        uncertainty=uncertainty,
        grad_cam_url=grad_cam_url,
        event_id=event_id,
    )

@app.post("/v1/copilot/plan")
async def copilot_plan(body: PlanRequest):
    payload = body.model_dump()
    result = generate_plan(payload)
    return JSONResponse(result)

@app.post("/v1/metrics/calibration", response_model=CalibrationReport)
async def calibration_report(probs: list[float] = Body(...), labels: list[int] = Body(...)):
    import numpy as np
    ece, points = engine.reliability_curve(np.array(probs), np.array(labels), n_bins=10)
    return {"ece": ece, "points": points}
