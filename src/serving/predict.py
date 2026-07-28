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
    
    # 1.0 convert nump --> PyTorch tensor
    # the model only understands PyTorch tensors, not numpy arrays
    # FloatTensor because our EEG values are decimals
    tensor = torch.tensor(
    eeg_signal,
    dtype=torch.float32,
    device=device
)
    # tensor shape: (1, 641, 64)
    
    
    # 2.0 run through model
    # torch.no_grad() tells PyTorch not to tracj gradients
    # we don't need gradients during inference - only during training
    # this saves memory and makes inderence faster
    with torch.no_grad():
        output = model(tensor)
    # output shape: (1, 2) -- two raw scores, one per class 
    
    
    # 3.0 convert raw scores ---> probabilities
    # softmax converts raw scores into probabilities that sum to 1
    # eg: [0.73, 0.27] means 73% confident it's left fist
    probabilities = torch.softmax(output, dim = 1)
    # probabilities shape: (1, 2)
    
    
    # 4.0 get the predicted class
    # argmax returns the index of the highest probability
    # 0 = left fist      1 = right fist
    predicted_label = probabilities.argmax(dim = 1).item()
    # .item() converts from a PyTorch tensor to a plain python int
    
    
    # confidence = the probability of the predicted class
    confidence = probabilities[0][predicted_label].item()
    
    
    # 5.0 build and return result
    return {
        "prediction":   LABELS[predicted_label],        # "left fist" or "right fist"
        "confidence":   round(confidence, 4),           # eg: 0.7325
        "label":        predicted_label,                # 0 or 1
        "scores":       output[0].tolist()              # raw scores eg: [1.23, -0.67]
    }
    
    
    
# helper: predict from raw numpy signal
def predict_from_raw(raw_signal: np.ndarray) -> dict:
    """
    convenience wrapper that handles the tensor conversion
    and batch dimenion automatically
    
    """
    
    
    # add batch dimension: (641, 64) ---> (1, 641, 64)
    # the model always expects a batch, even for one sample
    signal_with_batch = raw_signal[np.newaxis, :]
    
    return predict(signal_with_batch)