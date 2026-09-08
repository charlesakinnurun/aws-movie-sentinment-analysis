"""
Configuration and hyperparameters for SentimentScope.
"""

import os
import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_POS_PATH = os.path.join("aclImdb_v1", "aclImdb", "train", "pos")
TRAIN_NEG_PATH = os.path.join("aclImdb_v1", "aclImdb", "train", "neg")
TEST_POS_PATH = os.path.join("aclImdb_v1", "aclImdb", "test", "pos")
TEST_NEG_PATH = os.path.join("aclImdb_v1", "aclImdb", "test", "neg")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
MAX_LENGTH = 128
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
config = {
    "vocabulary_size": tokenizer.vocab_size,  # e.g., ~30522 for bert-base-uncased
    "num_classes": 2,                         # binary classification (pos/neg)
    "d_embed": 128,
    "context_size": MAX_LENGTH,
    "layers_num": 4,
    "heads_num": 4,
    "head_size": 32,  # 4 heads * 32 = 128 -> matches d_embed
    "dropout_rate": 0.1,
    "use_bias": True,
}

EPOCHS = 3
LEARNING_RATE = 3e-4
