"""
Configuration file for the Intelligent File Type Classification System
"""

import os

# ==================== DIRECTORY PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
FRAGMENTS_DIR = os.path.join(DATA_DIR, "fragments")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")

# Model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_FOREST_DIR = os.path.join(MODELS_DIR, "random_forest")
CNN_DIR = os.path.join(MODELS_DIR, "cnn")
XGBOOST_DIR = os.path.join(MODELS_DIR, "xgboost")
RESNET_DIR = os.path.join(MODELS_DIR, "resnet")

# ==================== FRAGMENT SETTINGS ====================
FRAGMENT_SIZE = 4096  # bytes (verified from Train data - all fragments are 4096 bytes)
DEFAULT_HEADER_SIZE = 64  # bytes to remove from start
DEFAULT_FOOTER_SIZE = 64  # bytes to remove from end

# ==================== DATA SPLIT RATIOS ====================
TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
VALIDATION_RATIO = 0.15

# ==================== FILE TYPE SIGNATURES ====================
# Common file signatures (magic numbers) for header/footer detection
FILE_SIGNATURES = {
    'pdf': {
        'header': [b'%PDF-'],
        'footer': [b'%%EOF', b'%%EOF\n', b'%%EOF\r\n']
    },
    'jpg': {
        'header': [b'\xff\xd8\xff'],
        'footer': [b'\xff\xd9']
    },
    'png': {
        'header': [b'\x89PNG\r\n\x1a\n'],
        'footer': [b'IEND\xae\x42\x60\x82']
    },
    'gif': {
        'header': [b'GIF87a', b'GIF89a'],
        'footer': [b'\x00\x3b']
    },
    'bmp': {
        'header': [b'BM'],
        'footer': []  # BMP doesn't have a footer
    },
    'zip': {
        'header': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
        'footer': [b'PK\x05\x06']
    },
    'mp3': {
        'header': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
        'footer': [b'TAG']  # ID3v1 tag
    },
    'mp4': {
        'header': [b'\x00\x00\x00\x18ftyp', b'\x00\x00\x00\x1cftyp', 
                   b'\x00\x00\x00\x20ftyp'],
        'footer': []  # MP4 doesn't have a standard footer
    },
    'avi': {
        'header': [b'RIFF', b'AVI '],
        'footer': []
    },
    'wav': {
        'header': [b'RIFF', b'WAVE'],
        'footer': []
    },
    'doc': {
        'header': [b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'],
        'footer': []
    },
    'docx': {
        'header': [b'PK\x03\x04'],
        'footer': [b'PK\x05\x06']
    },
    'xlsx': {
        'header': [b'PK\x03\x04'],
        'footer': [b'PK\x05\x06']
    },
    'exe': {
        'header': [b'MZ'],
        'footer': []
    },
    'elf': {
        'header': [b'\x7fELF'],
        'footer': []
    }
}

# ==================== NOISE SETTINGS ====================
NOISE_LEVEL = 0.0  # Default noise level (0.0 = no noise, 0.9 = high noise)

# ==================== MODEL SETTINGS ====================

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42

# CNN
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_RANDOM_STATE = 42

# ResNet
RESNET_EPOCHS = 50
RESNET_BATCH_SIZE = 32
RESNET_LEARNING_RATE = 0.001

# ==================== GENERAL SETTINGS ====================
RANDOM_SEED = 42
VERBOSE = True

# Supported file extensions
SUPPORTED_EXTENSIONS = [
    'pdf', 'jpg', 'jpeg', 'png', 'gif', 'bmp',
    'mp3', 'mp4', 'avi', 'wav',
    'doc', 'docx', 'xlsx', 'txt',
    'zip', 'exe', 'elf'
]
