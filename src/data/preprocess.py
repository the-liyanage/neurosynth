import mne 
import numpy as np

from mne.datasets import eegbci
from mne.io import concatenate_raws
from tqdm import tqdm


from config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    SUBJECTS,
    RUNS,
    FREQ_LOW,
    FREG_HIGH,
    EPOCH_TMIN,
    EPPOCH_TMAX,
    EVENT_ID
)



# loading the data
def load_subject(subject, runs, data_dir):
    """
    Loads all EEG runs for one subject
    and concatenates them into one recording.
    
    """
    raw_fnames = eegbci.load_data(
        subjects = subject,
        runs = runs, 
        path = data_dir,
        verbose = False
    )
    
    raws = [
        mne.io.read_raw_edf(
        f,
        preload = True,
        verbose = False
        )
        for f in raw_fnames
    
    ]
    
    raw = concatenate_raws(raws)
    # tiny clearning step of file extentions to be the same
    eegbci.standardize(raw)
    
    
    # map of where every electrode sits on the scalp
    # loads a template of electrode positions
    montage = mne.channels.make_standard_montage("standard_1005")
    # every channel has a physical location
    raw.set_montage(montage, verbose = False)
    return raw
    
    
# applying common average reference (CAR)
def apply_car(raw):
    """
    removes noise shared across all electrodes simultaneously 
    - validated during EDA where we found high-inter channel 
    correlation due to volume conduction
    
    """
    raw_car = raw.copy()
    raw.set_eeg_reference("average", verbose = False)
    return raw


# applying bandpass filter
def apply_bandpass_filter(raw):
    """
    keeps only 8-30 Hz the Alpha + Beta motor imagery bands.
    validated during EDA with PSD plot showing real activity
    in this range.
    
    below 8 Hz --> movement/sweat noise
    above 30Hz --> muscle noise + power line interference
    """
    raw.filter(
        l_freq = FREQ_LOW,
        h_freq = FREG_HIGH,
        # a filter (mathematical way)
        method = "iir",
        verbose = False
    )
    return raw


# extract epochs
def extract_epochs(raw):
    """
    cuts the continuous signal into 4-second windows
    aligned to each T1/T2 event onset
    
    note : event_id passed to events_from_annotations to 
    avoid the T0/T1/T2 label-mapping bug found during
    development (TO was accidentally included before this fix)
    """
    events, _ = mne.events_from_annotations(
        raw,
        event_id = EVENT_ID,
        verbose = False
    )
    
    epochs = mne.Epochs(
        raw,
        events,
        event_id = EVENT_ID,
        tmin = EPOCH_TMIN,
        tmax = EPPOCH_TMAX,
        baseline = None,
        preload = True,
        verbose = False
    )
    
    X = epochs.get_data()
    y = epochs.events[:, 2] -1
    return X, y    
    
    
    
# normalize
def normalize(X):
    """
    z-score normalization per channel per epoch.
    removes amplitude differences between subject caused by
    skull thickness, hair, electrode contact quality.
    
    model learns signal PATTERNS not raw voltage values
    
    """
    mean = X.mean(axis = -1, keepdims = True)
    std  = X.std(axis = -1, keepdims = True)
    std[std == 0] = 1
    return (X - mean) /std
    
    
# preprocess one raw EEG signal
def preprocess_raw_signal(raw):
    """
    runs the full preprocessing pipeline on ONE raw
    EEG recording - used by the serving layer for
    real - time inference on incoming signals 
    
    Pipeline: CAR --> bandpass filter --> epoch ---> normalize
    ---> transpose (time first, for Transfirner input)
    """
    
    raw = apply_car(raw)
    raw = apply_bandpass_filter(raw)
    X, _ = extract_epochs(raw)
    X = normalize
    
    # transpose from (n_epochs, 64, 641)
    # to (n_epochs, 641, 64) for Transformer input
    X = X.transpose(0, 2, 1)
    return X


# full pipeline for training data

    
        
    
    