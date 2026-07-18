from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# DIRS

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
RESOURCES_DIR = PROJECT_ROOT / "resources"
RESULTS_DIR = PROJECT_ROOT / "results"

# CHECKPOINTS

BEST_MODEL_FILE = CHECKPOINTS_DIR / "best_model.keras"
FINAL_MODEL_FILE = CHECKPOINTS_DIR / "final_model.keras"

# DATA

IMAGES_DIR = DATA_DIR / "images"
FEATURES_FILE = DATA_DIR / "flickr8k_vgg16_14_14_512_features.npy"
TRAIN_TF_RECORD_FILE = DATA_DIR / "train.tfrecord"
VAL_TF_RECORD_FILE = DATA_DIR / "val.tfrecord"
TEST_TF_RECORD_FILE = DATA_DIR / "test.tfrecord"
LEFT_TF_RECORD_FILE = DATA_DIR / "leftovers.tfrecord"

# RESOURCES

SPLITS_FILE = RESOURCES_DIR / "splits.json"
VECTORIZER_CONFIG_FILE = RESOURCES_DIR / "vectorizer_config.json"

# RESULTS

FIGURES_DIR = RESULTS_DIR / "figures"
HISTORY_PNG_FILE = FIGURES_DIR / "history.png"
LR_PNG_FILE = FIGURES_DIR / "lr.png"
BLEU_SCORES_PNG_FILE = FIGURES_DIR / "bleu_scores.png"
TEST_PREDICTIONS_PNGS_DIR = FIGURES_DIR / "test_predictions"

HISTORY_DIR = RESULTS_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "history.json"

METRICS_DIR = RESULTS_DIR / "metrics"
BLEU_SCORES_FILE = METRICS_DIR / "bleu_scores.json"

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TEST_PREDICTIONS_FILE = PREDICTIONS_DIR / "test_predictions.json"