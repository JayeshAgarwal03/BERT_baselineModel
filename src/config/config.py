# Model Configuration
MODEL_NAME = "bert-base-uncased"  # Other options: "t5-base", "facebook/bart-base"
NUM_LABELS = 3

# Training Configuration
EPOCHS = 6
LEARNING_RATE = 2e-5
BATCH_SIZE = 16

# Data Configuration
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Model Architecture
DROPOUT_RATE = 0.2

# Available model options for reference:
# - "bert-base-uncased": BERT model (default)
# - "t5-base": T5 encoder model
# - "facebook/bart-base": BART model
