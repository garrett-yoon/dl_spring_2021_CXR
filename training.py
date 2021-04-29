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

from model import make_model

torch.manual_seed(42)
rootPath = pathlib.Path(os.getcwd())


# Early stopping object
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        # self.val_loss_min = val_loss
        pass

def train_model(model, 
                criterion, 
                optimizer, 
                scheduler,
                dataloaders, 
                dataset_sizes,
                early_stopping,
                num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_train_loss = []
    avg_val_loss = []
    train_acc = []
    val_acc = []
    first_start = time.time()
    for epochID in range(num_epochs):
        
        print('Epoch {}/{}'.format(epochID, num_epochs - 1))
        print('-' * 10)
        start_time = time.time()
        tsTime = time.strftime("%H%M%S")
        tsDate = time.strftime("%d%m%Y")
        tsStart = tsDate + '-' + tsTime
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for batchID, (X, Y) in enumerate(dataloaders[phase]):
                
                Y = Y.cuda()
                X = X.cuda()
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(X)
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, Y.long())
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_corrects += torch.sum(preds == Y.data)
            
            # Calculate epoch loss/acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Store training and validation loss/acc
            if phase =='train':
                avg_train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
            else:
                scheduler.step(running_loss) # Reduce learning rate if val loss plateaus
                avg_val_loss.append(epoch_loss) 
                val_acc.append(epoch_acc.item())
                early_stopping(epoch_loss, model) # Add one to early stopping if val loss plateaus
                
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # Store best model and validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
                
                
        time_elapsed = time.time() - start_time
        print('Time Taken for Epoch: %.2fs' % (time.time() - start_time))
        print()

    print('Best Validation Acc: {:4f}'.format(best_acc))
    print('-' * 10)
    model.load_state_dict(best_model_wts)
        
    acc_loss_dict = {'train_loss': avg_train_loss, 
                    'train_acc':train_acc,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc}


    
    return model, acc_loss_dict

def retrain(MODEL, PRETRAIN, MAX_EPOCHS = 50):
    #Training code

    MODEL = MODEL
    PRETRAIN= PRETRAIN

    # Create model
    model = make_model(MODEL, PRETRAIN)

    # Define learning rate
    learning_rate = 1e-3

    # Create optimizer, LR scheduler, and loss function
    # opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), 
    #                  eps=1e-08, weight_decay=0.01)
    opt = torch.optim.SGD(model.parameters(), 
                          lr=learning_rate, 
                          momentum=0.9, 
                          weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, factor = 0.1, patience = 1, mode = 'min', verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create early stopping object
    early_stopping = EarlyStopping(patience=3, verbose=True, path='checkpoint.pt', trace_func=print)

    # Create output directory of train/val metrics + model state dict
    tsTime = time.strftime("%H-%M-%S")
    tsDate = time.strftime("%m-%d-%Y")
    tsEnd = tsDate + '_' + tsTime
    outputdir = f'{tsEnd}/'
    # outputdir = '/content/drive/MyDrive/' + f'{tsEnd}/'

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # PARAMETERS FOR DATALOADER

    BATCH_SIZE = 64
    input_size=224

    # Create dataloaders

    data_transforms = {}

    data_transforms['train'] = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms['val'] = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    init_dataset = ImageFolder('../COVID-19_Radiography_Dataset', data_transforms['val'])

    labels = pd.Series(np.asarray(init_dataset.targets))
    train_labels, test_labels = train_test_split(labels, test_size=0.2, 
                                             stratify=labels, random_state=0)
    train_labels, val_labels = train_test_split(train_labels, test_size=0.25, 
                                            stratify=train_labels,  random_state=0)

    datasets = {}
    datasets['train'] = torch.utils.data.Subset(init_dataset, train_labels.index)
    datasets['val'] = torch.utils.data.Subset(init_dataset, val_labels.index)
    datasets['test'] = torch.utils.data.Subset(init_dataset, test_labels.index)


    datasets['train'].dataset.transform = data_transforms['train']

    train_loader = DataLoader(dataset=datasets['train'],batch_size =BATCH_SIZE,
                             shuffle=True)
    val_loader = DataLoader(dataset=datasets['val'],batch_size =BATCH_SIZE,
                             shuffle=False)
    test_loader = DataLoader(dataset=datasets['test'],batch_size =BATCH_SIZE,
                             shuffle=False)


    dataloaders = {'train': train_loader, 
               'val': val_loader, 
               'test': test_loader}
    dataset_sizes = {'train': len(datasets['train']), 
                 'val': len(datasets['val'] ), 
                 'test': len(datasets['test'])}

    # for k,v in datasets.items():
    #     print(k, len(v))

    # for x,v in dataloaders.items():
    #     print(x, len(v))

    # Train model
    model_ft, acc_loss_dict = train_model(model, 
                                          loss_fn, 
                                          opt, 
                                          scheduler, 
                                          dataloaders,
                                          dataset_sizes,
                                          early_stopping,
                                          num_epochs=MAX_EPOCHS)

    # Save hyperparameters
    hp = open(outputdir + "hyperparameters.txt","w") #write mode
    hp.write(f"lr = {learning_rate}\n") 
    hp.write(f"opt = {opt.__class__.__name__}\n")
    hp.write(f"model={MODEL}\n")
    hp.write(f"pretrain={PRETRAIN}\n")
    hp.close()

    items = ['train_acc', 'train_loss', 'val_acc', 'val_loss']

    print("Done Training Model")
    print('-' * 10)        

    # Save train/val loss and accuracy per epoch 
    for i in items:
        np.save(outputdir +  f'{i}_epoch', acc_loss_dict[i])

    # Save model dict
    torch.save(model.state_dict(), outputdir + f'model_dict_{tsEnd}.pt')

    print("Saved!")


