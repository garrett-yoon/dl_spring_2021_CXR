from training import *
from model import *
import argparse


MODEL = 'densenet121'
PRETRAIN = True
MAX_EPOCHS = 1

retrain(MODEL, PRETRAIN, MAX_EPOCHS)