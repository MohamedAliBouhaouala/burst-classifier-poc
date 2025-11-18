SEED = 42
LABEL_MAP = {"b": 0, "mb": 1, "h": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
LABELS = ["b", "mb", "h"]
LABEL_IDX = {lab: i for i, lab in enumerate(LABELS)}

TRAINING = "TRAINING"
VALIDATION = "VALIDATION"
TEST = "TEST"
FULL = "FULL"
SPLIT_CHOICES = [TRAINING, VALIDATION, TEST, FULL]

REQUIRED_COLS = {"audio_file", "start_seconds", "end_seconds", "label"}
