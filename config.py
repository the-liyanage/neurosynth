from pathlib import Path

# paths
ROOT_DIR = Path(__file__).parent
RAW_DATA_DIR            = ROOT_DIR/ "data" / "raw"
PROCESSED_DATA_DIR      = ROOT_DIR/ "data" / "processed"


ARTIFACTS_DIR           = ROOT_DIR/ "artifacts"
MODEL_PATH              = ARTIFACTS_DIR/ "best_model.pth"
CONFIG_PATH             = ARTIFACTS_DIR/ "model_config.json"

