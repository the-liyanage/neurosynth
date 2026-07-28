# Inference pipeline for Neurosynth
# this file is the bridge between:
#   - raw EEG input( coming from a websocket connection)
#   - a trained model prediction (left or right fist)


# used by server.py - the model is loaded once at startup,
# then predict() is called for every incoming signal

import torch
import numpy as np

from config import(
    MODEL_PATH,
    LABELS,
    D_MODEL
)
from src.models.transformer import load_model


# load model once at module import
# we load the model here (at module level) rather than
# inside the predict() function - this is intentional
# loading a model takes ~ 1 second . if we loaded it inside predict(),
# every single prediction would take 1 extra second.
# loading once at startup means all predictions are fast

model, device = load_model(MODEL_PATH)

# Main prediction function
def predict(eeg_signal: np.ndarray) -> dict:
    """
    Takes a preprocessed EEG signal and returns a prediction.
    
    Args:
    eeg_signal : numpy array shape (1, 641, 64)
                - already preprocessed
                
    Returns:
    DICT WITH:
        prediction:     str - "left fist" or "right fist"
        confidence:     float - prob of predicted class (0 -1)
        label:          int - 0 or 1
        scores:         list - raw scores for both classes
    """
    