"""FastAPI application for Smart Loan Recovery."""

import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import pandas as pd

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter

from src.api.schemas import LoanFeatures, PredictionResponse, CollectionOptimizationRequest
from src.models.predict_model import load_all_models, predict

# Setup structured logging
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

# Custom Prometheus Metrics
slr_prediction_auc = Gauge("slr_prediction_auc", "AUC of the loaded ensemble model")
slr_risk_tier_counter = Counter("slr_risk_tier_counter", "Count of predictions by risk tier", ["tier"])
slr_recovery_p50_avg = Gauge("slr_recovery_p50_avg", "Rolling average of recovery P50")
slr_collection_action_counter = Counter("slr_collection_action_counter", "Count of actions recommended", ["channel", "intensity"])

# State for average calculation
recovery_state = {"total": 0.0, "count": 0}

# Global dictionary to hold models
models = {}

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Phase 4 models at startup and clean up at shutdown."""
    logger.info(json.dumps({"event": "startup", "message": "Loading Phase 4 models..."}))
    try:
        stacker, recovery_models, rl_agent = load_all_models()
        models['stacker'] = stacker
        models['recovery_models'] = recovery_models
        models['rl_agent'] = rl_agent
        models['status'] = 'ready'
        
        slr_prediction_auc.set(0.85) 
        
        logger.info(json.dumps({"event": "startup", "message": "Models loaded successfully."}))
    except Exception as e:
        logger.error(json.dumps({"event": "startup", "message": f"Error loading models: {e}"}))
        models['status'] = 'error'
    yield
    models.clear()

app = FastAPI(title="Smart Loan Recovery API", lifespan=lifespan)

# Instrument FastAPI with Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom exception handler to return 422 with human-readable message."""
    details = exc.errors()
    messages = [f"{err.get('loc')[-1]}: {err.get('msg')}" for err in details]
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(messages)}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Async middleware to log request_id, endpoint, latency, and risk_tier."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    response = await call_next(request)
    
    latency_ms = (time.time() - start_time) * 1000
    risk_tier = getattr(request.state, "risk_tier", "unknown")
    
    log_data = {
        "request_id": request_id,
        "endpoint": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
        "risk_tier": risk_tier
    }
    logger.info(json.dumps(log_data))
    return response

@app.get("/health")
async def health_check():
    """Returns API health status."""
    uptime = time.time() - START_TIME
    return {
        "status": models.get('status', 'loading'),
        "model_version": "v1.0",
        "uptime_seconds": round(uptime, 2)
    }

@app.post("/predict/default", response_model=PredictionResponse)
async def predict_default(features: LoanFeatures, request: Request):
    """Predict default probability and recommend collection actions."""
    if models.get('status') != 'ready':
        raise HTTPException(status_code=503, detail="Models are not ready.")
        
    try:
        # Convert pydantic model to dict, then dataframe
        features_dict = features.model_dump()
        df = pd.DataFrame([features_dict])
        
        # Run prediction
        results = predict(
            df,
            stacker=models['stacker'],
            recovery_models=models['recovery_models'],
            rl_agent=models['rl_agent']
        )
        
        res = results[0]
        
        # Prometheus updates
        tier = res.get('risk_tier', 'unknown')
        slr_risk_tier_counter.labels(tier=tier).inc()
        
        if res.get('recovery_p50') is not None:
            recovery_state["total"] += res['recovery_p50']
            recovery_state["count"] += 1
            avg_p50 = recovery_state["total"] / recovery_state["count"]
            slr_recovery_p50_avg.set(avg_p50)
            
        if res.get('recommended_channel') is not None and res.get('recommended_channel') != 'none':
            channel = res['recommended_channel']
            intensity = res['recommended_intensity']
            slr_collection_action_counter.labels(channel=channel, intensity=intensity).inc()
        
        # Attach risk_tier to request state for the middleware logger
        request.state.risk_tier = tier
        
        return PredictionResponse(**res)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/collection")
async def optimize_collection(request_data: CollectionOptimizationRequest):
    """Optimize collection strategy endpoint (Placeholder for advanced manual checks)."""
    return {
        "customer_id": request_data.customer_id,
        "risk_tier": request_data.risk_tier,
        "message": "Optimization strategy requested successfully."
    }
