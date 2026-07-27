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
    
    