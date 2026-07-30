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
class EEGRequest(BaseModel):
    """
    Shape of data the client send to /predict
    signal: a 2D array of shape (641, 64)
            641 timepoints * 64 EEG channe;s
            sent as a nested list (JSON doesn't have numpy)
            
    """
    signal: List[List[float]]
    
    
class PredictionResponse(BaseModel):
    """
    Shape of data we send back to the client.
    
    """
    prediction: str         # "Left fist" or "Right fist"
    confidence: float       # probability of predicted class
    label: int              # 0 or 1
    scores: List[float]     # raw model output scores
    
    

    
# ----------- ENDPOINT 1: health check
@app.get("/")
def health_check():
    """
    simple health check endpoint.
    returns a message confirming the server is running.
    
    used by deployment platforms( Railway) to verify
    the service is alive.
    
    """
    return {
        "status": "online",
        "message": "NeuroSynth API is running",
        "model": "EEG Transformer (`61.1%` accuracy)",
        "classes": list(LABELS.values())
        
    }
    
# ---------- ENDPOINT 2. REST prediction
@app.post("/predict", response_model = PredictionResponse)
def predict_endpoint(request: EEGRequest):
    """
    takes a preprocessed EEG signal and returns a prediction.
    
    Why REST (POST) and WebSocket?
    REST ---> simple, stateless,
             one request = one response
             good for single predictions on demand 
             
    WebSocket ---> persistent connection, streams results continuously 
                   good for real-time BCI where signals arrive constantly 
                   
                   
    This endpoint is useful for:
    - testing the model from curl or Postman
    - the streamlit demo (which uses REST, not WebSocket)
    - any client that doesn't need real-time streaming
    
    """
    
    
    # convert the nestes list --> numpy array
    signal = np.array(request.signal, dtype = np.float32)
    # signal shape: (641, 64)
    
    # run inference
    result = predict_from_raw(signal)
    return result


# -------- Endpoint 3: WebSocket real-time inference
@app.websocket("/ws")
async def websocket_endpoint(webscket: WebSocket):
    """
    WebSocket endpoint for real-time EEG inference.
    
    How WebSocket differs from REST:
    REST:           client sends requests --> server responds --> connection closes
    WebSocket:      connection stays OPEN --> client keeps sending signals
                    server keeps sending predictions back
                    no overhead of opening/ closing connection each time
                    
    
    This is  what enables sub - 120 msn end to end latency
    the persistent connection removes the connection setup
    overhead that REST would add for each prediction,
    
    
    Flow:
    1. Client connects to ws://localhost:8000/ws
    2. Client sends EEG signal as JSON string
    3. Server runs inference sends prediction back as JSON
    4. Repeat continuously util client disconnects
    
    """
    
    # accept the WebSocket connection
    await webscket.accept()
    print("WebSocket client connected!")
    
    try:
        while True:
            # wait for data from the client
            data = await webscket.receive_text()
            
            # parse the JSON string into a Python dict
            payload = json.loads(data)
            
            # extract the signal array
            signal = np.array(payload["signal"], dtype = np.float32)
            # signal shape: (641, 64)
            
            # run inference
            result = predict_from_raw(signal)
            
            # send prediction back to client as JSON
            await webscket.send_text(json.dumps(result))
            
            
            
    except WebSocketDisconnect:
        # client disconneted - this is normal, not an error
        print("WebSocket client disconnected")

    except Exception as e:
        # something went wrong - send error back to client
        await webscket.send_text(json.dumps({
            "error": str(e)
        }))    
    


