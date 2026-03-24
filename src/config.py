import os
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

TICKET_CSV = os.path.join(DATA_DIR, 'customer_support_tickets.csv')
GLOVE_FILE = os.path.join(DATA_DIR, 'glove.6B.300d.txt')

# Hyperparameters / Configurations
VEC_DIM = 300
VOCAB_LIMIT = 5000
RNG_SEED = 7

# Compute configuration
gpu_count = torch.cuda.device_count()
compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed():
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RNG_SEED)

set_seed()
