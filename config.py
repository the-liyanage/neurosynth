from pathlib import Path

# paths

# project root directory
ROOT_DIR = Path(__file__).parent

# data paths
RAW_DATA_DIR            = ROOT_DIR/ "data" / "raw"
PROCESSED_DATA_DIR      = ROOT_DIR/ "data" / "processed"


# artifact path
ARTIFACTS_DIR           = ROOT_DIR/ "artifacts"

MODEL_DIR               = ARTIFACTS_DIR/ "models"
CONFIG_DIR              = ARTIFACTS_DIR/ "configs"
METRICS_DIR             = ARTIFACTS_DIR / "metrics"
PLOTS_DIR               = ARTIFACTS_DIR / "plots"


# model files
MODEL_PATH              = MODEL_DIR/ "best_model.pth"
MODEL_CONFIG_PATH       = CONFIG_DIR/ "model_config.json"

# dataset files
X_PATH                  = PROCESSED_DATA_DIR/ "X.npy"
Y_PATH                  = PROCESSED_DATA_DIR/ "y.npy"
SUBJECT_PATH            = PROCESSED_DATA_DIR/ "subject_ids.npy"


# EEG Data
SUBJECTS                = list(range(1, 21))

RUNS                    = [4, 8, 12]

SFREQ                   = 160.0

N_CHANNELS              = 64



# Preprocessing
FREQ_LOW                = 8.0

FREQ_HIGH               = 30.0

EPOCH_TMIN              = 0.0

EPOCH_TMAX             = 4.0

EVENT_ID                = {
    "T1": 1,
    "T2": 2
}




# Model Architecture
IN_CHANNELS             = 64

CONV_OUT                = 32

D_MODEL                 = 96

NHEAD                   = 4

NUM_LAYERS              = 2

NUM_CLASSES             = 2

DIM_FEEDFORWARD         = 256

DROPOUT                 = 0.1




#Training 
BATCH_SIZE              = 16

NUM_EPOCHS              = 50

LEARNING_RATE           = 0.001

TEST_SIZE               =  0.2

RANDOM_STATE            = 42




# Serving
HOST                    = "0.0.0.0"

PORT                    = 8000

LABELS                  ={
    0 : "Left Fist",
    1 : "Right Fist"
}



for folder in [
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    CONFIG_DIR,
    METRICS_DIR,
    PLOTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
