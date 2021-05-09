from training import *
from model import *
import argparse

# Define model architecture, pre-trained, and max epochs
MODEL = 'densenet121'
PRETRAIN = True
MAX_EPOCHS = 1

# Starting model training
print('Start Model Training')
retrain(MODEL, PRETRAIN, MAX_EPOCHS)