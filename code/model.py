import torch
import torchvision.models as models

# Create model

def make_model(MODEL, PRETRAIN):
    if MODEL == 'resnet18':
        model = models.resnet18(pretrained=PRETRAIN)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 4)
    if MODEL == 'resnet50':
        model = models.resnet50(pretrained=PRETRAIN)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 4)
    if MODEL == 'densenet121':
        model = models.densenet121(pretrained=PRETRAIN)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, 4)
        
    model.to('cuda')

    return model