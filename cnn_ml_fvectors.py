import os
import pathlib
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split

import numpy as np 
import pandas as pd 
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import time
import copy
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)
rootPath = pathlib.Path(os.getcwd())

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
import time
from xgboost import XGBClassifier
from model import make_model

# Edit these
ml_model = 'rf'
MODEL='densenet121'

# Create model
PRETRAIN=True
BATCH_SIZE = 32

input_size = 224
model_ft = make_model(MODEL, PRETRAIN)
model_no_fc = copy.deepcopy(model_ft)
if MODEL == 'densenet121':
    model_no_fc.classifier = torch.nn.Identity()
else:
    model_no_fc.fc = torch.nn.Identity()
model_ft.eval()
model_no_fc.eval()

# Transforms for data
ml_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data loader initialization
init_dataset = ImageFolder('../COVID-19_Radiography_Dataset', ml_transforms)
data_loader = DataLoader(dataset=init_dataset, batch_size =BATCH_SIZE, shuffle=True)

# Make arrays for outputs
logits = []
predictions = []
labels = []
bottleneck_features = []
for batchID, (X, Y) in enumerate(data_loader):

    Y = Y.cuda()
    X = X.cuda()

    bn_feat = model_no_fc(X)
    logit = model_ft(X)
    _, preds = torch.max(logit, 1)
    
    bottleneck_features.append(bn_feat.detach().cpu())
    logits.append(logit.detach().cpu())
    predictions.append(preds.cpu())
    labels.append(Y.detach().cpu())

batch_feat = torch.cat(bottleneck_features, dim=0)
batch_logits = torch.cat(logits, dim=0)
batch_pred = torch.cat(predictions, dim=0)
batch_lab = torch.cat(labels, dim = 0)

batch_pred = torch.reshape(batch_pred, (batch_pred.shape[0], 1))
batch_lab = torch.reshape(batch_lab, (batch_lab.shape[0],1))

# Create tensor with both predicted and true labels
batch_pl = torch.cat([batch_pred, batch_lab], dim=1)

idx_to_class = {v:k for k,v in init_dataset.class_to_idx.items()}

# Store as dataframe
pred_true = pd.DataFrame(batch_pl.numpy(), columns = ['Pred', 'True'])
pred_true = pred_true.replace(idx_to_class)

# Make feature and label vectors
y = pred_true['True']
X = batch_feat

# Random forest
if ml_model == 'rf':
	lr = LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100)
	lr = AdaBoostClassifier(n_estimators=100)
	clfs = [lr, rf]
	clfs_names = ['lr', 'rf']

	nest_score = {}
	for clf,name in zip(clfs, clfs_names):
	    nest_score[name] = cross_validate(clf, X, y, cv = 3, 
	                                   scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
	                                   verbose=True, n_jobs=-1)

	print(pd.DataFrame(nest_score['lr']).mean(axis=0))

# XGB
if ml_model == 'xgb':
	X_train, X_test, y_train, y_test = train_test_split(X, 
	                                                    y, 
	                                                    test_size=0.20, 
	                                                    random_state=0, 
	                                                    shuffle=True, 
	                                                    stratify=y)

	xgb = XGBClassifier(n_estimators=500, random_state=0)

	eval_set = [(X_train, y_train), (X_test, y_test)]
	eval_metric = ["merror"]

	xgb.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

	pred_test = xgb.predict(X_test)
	pred_train = xgb.predict(X_train)
	print('Train Accuracy: ', accuracy_score(y_train, pred_train))
	print('Test Accuraccy: ', accuracy_score(y_test, pred_test))
	print('Classification Report:')
	print(classification_report(y_test,pred_test, digits=4))