import mne 
import numpy as np

from mne.datasets import eegbci
from mne.io import concatenate_raws
from tqdm import tqdm


from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SUBJECTS,
    RUNS,
    FREQ_LOW,
    FREQ_HIGH,
    EPOCH_TMIN,
    EPOCH_TMAX,
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
    raw_car.set_eeg_reference("average", verbose = False)
    return raw_car


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
        h_freq = FREQ_HIGH,
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
        tmax = EPOCH_TMAX,
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
def run_preprocessing(subjects, runs, data_dir):
    """
    Complete EEG preprocessing pipeline.

    Loads all subjects,
    applies preprocessing steps,
    extracts epochs,
    normalizes data,
    and returns processed arrays.
    """

    all_X = []
    all_y = []
    all_subject_ids = []


    print("Starting preprocessing pipeline...\n")


    for subject in tqdm(
        SUBJECTS,
        desc="Processing subjects"
    ):

        print(f"\nSubject {subject}")


        # 1. Load EEG data
        raw = load_subject(
            subject,
            RUNS,
            RAW_DATA_DIR
        )


        # 2. Common Average Reference
        raw = apply_car(raw)


        # 3. Band-pass filtering
        raw = apply_bandpass_filter(raw)


        # 4. Extract epochs and labels
        X, y = extract_epochs(raw)


        # Check if subject produced enough data
        if len(X) < 5:
            print(
                f"Skipping subject {subject}: too few epochs"
            )
            continue


        # 5. Normalize EEG signals
        X = normalize(X)


        # Store results
        all_X.append(X)
        all_y.append(y)


        # IMPORTANT:
        # keep track of which subject produced each epoch

        subject_ids = np.full(
            len(y),
            subject
        )

        all_subject_ids.append(subject_ids)


        print(
            f"Subject {subject}: "
            f"{X.shape[0]} epochs"
        )



    # Combine all subjects together

    all_X = np.concatenate(
        all_X,
        axis=0
    )

    all_y = np.concatenate(
        all_y,
        axis=0
    )

    all_subject_ids = np.concatenate(
        all_subject_ids,
        axis=0
    )


    print("\nPreprocessing complete!")
    print("---------------------------")
    print(f"X shape: {all_X.shape}")
    print(f"y shape: {all_y.shape}")
    print(
        f"subject_ids shape: {all_subject_ids.shape}"
    )


    return (
        all_X,
        all_y,
        all_subject_ids
    )
    
        
if __name__ == "__main__":
    # Run the preprocessing pipeline
    X, y, subject_ids = run_preprocessing(
        SUBJECTS,
        RUNS,
        RAW_DATA_DIR
    )

    # Save processed arrays for training
    np.save(PROCESSED_DATA_DIR / "X.npy", X)
    np.save(PROCESSED_DATA_DIR / "y.npy", y)
    np.save(PROCESSED_DATA_DIR / "subject_ids.npy", subject_ids)

    print("\nProcessed data saved successfully!")