# FastAPI server 
# Exposes two endpoints:
# GET /              ---> health check (confirms server is running)
# POST / predict     ---> takes EEG signal, returns prediction
# WS /  ws           ---> Websocket for real time inference



import numpy as np
from fastapi  import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json

from src.serving.predict import predict, predict_from_raw
from config import PORT, HOST, LABELS

# Create FastAPI app
app = FastAPI(
    title = "NeuroSynth API",
    description = "Real-time EEG motor imagery classification",
    version = "1.0.0"
)


# CORS Middleware
# CORS = Cross - Origin Resource Sharing
# Without this, our React frontend (running on port 3000)
# would be blocked from talking to this server (port 8000)
# by the browser's security policy
# allow_origins = ["*"] means accept requests from ANY origin
# in production you'd restrict this to your specific frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)


# Request/Response models
# Pydantic models define the exact shape of data
# coming IN and going OUT of our API endpoints
# FastAPI uses these to automatically valudate requests
# and generate API documentation
class EEGReqest(BaseModel):
    """
    Shape of data the client send to /predict
    signal: a 2D array of shape (641, 64)
            641 timepoints * 64 EEG channe;s
            sent as a nested list (JSON doesn't have numpy)
            
    """
    signal = List[List[float]]
    
    
class PredictionResponse(BaseModel):
    """
    Shape of data we send back to the client.
    
    """
    prediction: str         # "Left fist" or "Right fist"
    confidence: float       # probability of predicted class
    label: int              # 0 or 1
    scores: List[float]     # raw model output scores
    
    

    
    


