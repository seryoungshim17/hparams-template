import torch
from torchmetrics.functional import f1, accuracy

def calculateAcc(y_pred, y_test, num_classes):    
    output = torch.argmax(y_pred, dim=1)
    return accuracy(output, y_test), f1(output, y_test, average='macro', num_classes=num_classes)