
#trainning code;
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import shutil
import time
import pandas as pd
import numpy as np
import PIL.Image
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from PIL import Image
from os import listdir
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, SequentialSampler 

import h5py
from models.equiv_cnn import *
from models.resnet_2_blocks import *
from models.cnn_model import *
from torch.autograd import Variable
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import os

import sys 
'''stdoutOrigin=sys.stdout 
sys.stdout = open("log.txt", "w")'''


# Data preparation
CHECKPOINT_DIR = 'checkpoint'
drive_base_path = './pcam/'
BATCH_SIZE =16
print("batchsize",BATCH_SIZE)
dataloader_params = {'batch_size': BATCH_SIZE, 'num_workers': 2,}

class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset_x = None
        self.dataset_y = None
        self.transform = transform
        ### Going to read the X part of the dataset - it's a different file
        with h5py.File(self.file_path + '_x.h5', 'r') as filex:
            self.dataset_x_len = len(filex['x'])

        ### Going to read the y part of the dataset - it's a different file
        with h5py.File(self.file_path + '_y.h5', 'r') as filey:
            self.dataset_y_len = len(filey['y'])

    def __len__(self):
        assert self.dataset_x_len == self.dataset_y_len # Since we are reading from different sources, validating we are good in terms of size both X, Y
        return self.dataset_x_len

    def __getitem__(self, index):
        imgs_path = self.file_path + '_x.h5'
        labels_path = self.file_path + '_y.h5'

        if self.dataset_x is None:
            self.dataset_x = h5py.File(imgs_path, 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(labels_path, 'r')['y']

        # get one pair of X, Y and return them, transform if needed
        image = self.dataset_x[index]
        label = self.dataset_y[index]

        if self.transform:
            image = self.transform(image)

        return (image, label)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_path = drive_base_path + 'camelyonpatch_level_2_split_train'
# val_path = drive_base_path + 'camelyonpatch_level_2_split_valid'
#test_path = drive_base_path + 'camelyonpatch_level_2_split_test'

train_dataset = H5Dataset(train_path, transform=train_transforms)
train_loader = DataLoader(train_dataset, **dataloader_params)



test_validation_split = 0.80
test_split=0.5

dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_validation_split * dataset_size))
split2= int(np.floor((dataset_size+split)/2))
# print(split)
# print(dataset_size)

train_indices, valid_indices,test_indices = indices[:split], indices[split:split2],indices[split2:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler  = SequentialSampler(test_indices)


val_dataset = H5Dataset(train_path, transform=test_transforms)
validation_loader = DataLoader(val_dataset, **dataloader_params, sampler=valid_sampler)

train_dataset = H5Dataset(train_path, transform=train_transforms)
train_loader = DataLoader(train_dataset, **dataloader_params, sampler=train_sampler)
#test_loader=validation_loader
test_dataset = H5Dataset(train_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, **dataloader_params,sampler=test_sampler)

# val_dataset = H5Dataset(val_path, transform=test_transforms)
# validation_loader = DataLoader(val_dataset, **dataloader_params,shuffle=False)


print("train_size:",len(train_loader)*BATCH_SIZE)
print("val_size:",len(validation_loader)*BATCH_SIZE)
print("test_size:",len(test_loader)*BATCH_SIZE)



# print("training")
# for i in range(len(train_loader)):
#     sample = train_loader[i]
#     print(i, sample['image'].size(), sample['label'].size())
#     if i == 5:
#         break

# print("validation")
# for i in range(len(dev_loader )):
#     sample = dev_loader[i]
#     print(i, sample['image'].size(), sample['label'].size())
#     if i == 5:
#         break
        
# print("testing")
# for i in range(len(test_loader )):
#     sample = test_loader[i]
#     print(i, sample['image'].size(), sample['label'].size())
#     if i == 5:
#         break
#helper functions;
def sigmoid(x):
    """This method calculates the sigmoid function"""
    return 1.0/(1.0 + np.exp(-x))

def training_accuracy(predicted, true, i, acc):
    """Taken from https://www.kaggle.com/krishanudb/cancer-detection-deep-learning-model-using-pytorch"""
    predicted = predicted.cpu() # Taking the predictions, why cpu and not device?
    true = true.cpu() # Taking the labels, why cpu and not device?
    
    predicted = (sigmoid(predicted.data.numpy()) > 0.5) # Using sigmoid above, if prediction > 0.5 it is 1
    true = true.data.numpy()
    accuracy = np.nan_to_num(np.sum(predicted == true) / true.shape[0]) # Accuracy is: (TP + TN)/(TP + TN + FN + FP)
    acc = acc * (i) / (i + 1) + accuracy / (i + 1)
    return acc


def dev_accuracy(predicted, target):
    """Taken from https://www.kaggle.com/krishanudb/cancer-detection-deep-learning-model-using-pytorch"""
    predicted = predicted.cpu()
    target = target.cpu()
    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    true = target.data.numpy()
    accuracy = np.nan_to_num(np.sum(predicted == true) / true.shape[0])
    return accuracy

def test_accuracy(predicted, target,y_true,y_score):
    """Taken from https://www.kaggle.com/krishanudb/cancer-detection-deep-learning-model-using-pytorch"""
    predicted = predicted.cpu()
    target = target.cpu()
    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    for v in predicted :
        y_score.append(v[0])
    true = target.data.numpy()
    for v in true:
        y_true.append(v[0])
    #print( np.sum(predicted == true))
    accuracy = np.nan_to_num(np.sum(predicted == true) / true.shape[0])
 
    return accuracy

def fetch_state(epoch, model, optimizer, dev_loss_min, dev_acc_max):
    """Returns the state dictionary for a model and optimizer"""
    state = {
        'epoch': epoch,
        'dev_loss_min': dev_loss_min,
        'dev_acc_max': dev_acc_max,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    }
    return state

def save_checkpoint(state, is_best = False, checkpoint = CHECKPOINT_DIR):
    """Taken from CS230 PyTorch Code Examples"""
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last_v2.pth.tar')
    if (not os.path.exists(checkpoint)):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if (is_best):
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_v2.pth.tar'))
        
def load_checkpoint(model, optimizer = None, checkpoint = "./checkpoint/best_v2.pth.tar"):
    """Taken from CS230 PyTorch Code Examples"""
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        print("File doesn't exist {}".format(checkpoint))
        checkpoint = None
        return
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # if optimizer:
    #     optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def update_scores(loss_arr, acc_arr, curr_avg_loss, curr_acc):
    """This method gets scores / metrics arrays and parameters, and updates them accordingly"""
    loss_arr.append(curr_avg_loss)
    acc_arr.append(curr_acc)
   


  #model hyperparameters and optimization
  # Model Instantiation
USE_GPU = torch.cuda.is_available()

model = eqcnn()
print(model)

if (USE_GPU):
    model.cuda()
    print("cuda is available runing on GPU")
else:
    print("cuda is not available runing on CPU")

# Hyperparameters + Log
lr = 0.5e-4
print("lr--",lr)
#wandb.config.learning_rate = lr

# Parameters
total_epochs = 0
num_epochs = 30
patience = 7
bad_epoch_count = 0
stop = False
train_loss_min = np.Inf
dev_loss_min = np.Inf
dev_acc_max = 0

# Optimizer + Loss Function
optimizer = optim.Adam(model.parameters(),lr=lr)
# optimizer = optim.SGD(model.parameters(), lr = lr)
criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy for binary classification - malignant / benign
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=1,verbose=True)
best_checkpoint = os.path.join(CHECKPOINT_DIR, 'best_v2.pth.tar') # For saving model

total_epochs = 0 # Used in training

# Initialize arrays for plot
train_loss_arr = []
train_acc_arr = []


dev_loss_arr = []
dev_acc_arr = []


#trainning of model



     # Loop over the dataset multiple times
total_num_epochs = total_epochs + num_epochs
for epoch in range(num_epochs):
    print('current learning rate = {} ;'.format (optimizer.param_groups[0]["lr"]))
    curr_epoch = total_epochs + epoch + 1
    # Keep track of training loss
    train_loss = []
    # Keep track of dev loss
    dev_loss = []
    dev_acc1 = []
    acc=0
    # Train the model
    start_time = time.time()
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        if USE_GPU:
            data, target = image.cuda(), label.cuda()
        else:
            data, target = image, label
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Update target to be the same dimensions as output
        target = target.view(output.shape[0], 1).float()
        # Get accuracy measurements
        acc = training_accuracy(output, target, batch_idx, acc)
        # Calculate the batch's loss
        curr_train_loss = criterion(output, target)
        # Update the training loss
        train_loss.append(curr_train_loss.item())
        # Backward pass
        curr_train_loss.backward()
        # Perform a single optimization step to update parameters
        optimizer.step()
        train_loss.append(curr_train_loss.item())
        # Print debug info every 64 batches
        if (batch_idx) % BATCH_SIZE == 0:
            print('Epoch {}/{}; Iter {}/{}; Loss: {:.4f}; Acc: {:.3f};'
                   .format(curr_epoch, total_num_epochs, batch_idx + 1, len(train_loader), curr_train_loss.item(), acc))
            
    end_time = time.time()
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(validation_loader):
            if USE_GPU:
                data, target = image.cuda(), label.cuda()
            else:
                data, target = image, label
            # Get predicted output
            output = model(data)
            # Update target to be the same dimensions as output
            target = target.view(output.shape[0], 1).float()
            # Get accuracy measurements
            dev_acc = dev_accuracy(output, target)
            # Calculate the batch's loss
            curr_dev_loss = criterion(output, target)
            # Update the dev loss
            dev_loss.append(curr_dev_loss.item())
            dev_acc1.append(dev_acc)
    
    # Calculate average loss
    avg_train_loss = np.mean(np.array(train_loss))
    avg_dev_loss = np.mean(np.array(dev_loss))
    dev_acc = np.mean(np.array(dev_acc1))
    scheduler.step(avg_dev_loss)
    # Update dev loss arrays
    update_scores(dev_loss_arr, dev_acc_arr, avg_dev_loss, dev_acc)

    # Update training loss arrays
    update_scores(train_loss_arr, train_acc_arr, avg_train_loss, acc)

    print('Epoch {}/{}; Avg. Train Loss: {:.4f}; Train Acc: {:.3f}; Epoch Time: {} mins; \nAvg. Dev Loss: {:.4f}; Dev Acc: {:.3f};'
        .format(curr_epoch, total_num_epochs, avg_train_loss, acc, round((end_time - start_time)/ 60., 2), avg_dev_loss, dev_acc))
    
    #wandb.log({'epoch': curr_epoch, 'loss': avg_train_loss, 'accuracy': acc, 'tpr': tpr, 'time_per_epoch_min': round((end_time - start_time)/ 60., 2)})

    if avg_dev_loss < dev_loss_min:
        print('Dev loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
              .format(dev_loss_min, avg_dev_loss))
        dev_loss_min = avg_dev_loss
        is_best = False
        if (dev_acc >= dev_acc_max):
            is_best = True
            dev_acc_max = dev_acc
        state = fetch_state(epoch = curr_epoch, model = model, optimizer = optimizer, 
                            dev_loss_min = dev_loss_min, 
                            dev_acc_max = dev_acc_max)
        save_checkpoint(state = state, is_best = is_best)
        bad_epoch_count = 0
    # If dev loss didn't improve, increase bad_epoch_count and stop if
    # bad_epoch_count >= early_stop_limit
    else:
        bad_epoch_count += 1
        print('{} epochs of increasing dev loss ({:.6f} --> {:.6f}).'
              .format(bad_epoch_count, dev_loss_min, avg_dev_loss))
        if (bad_epoch_count >= patience):
            print('Stopping training')
            stop = True

    if (stop):
        break
    


plt.title('---LOSS VALUE_lr=0.0000005_bs_16---')
plt.plot(train_loss_arr, label='Training loss')
plt.plot(dev_loss_arr , label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.savefig("equicnn_lr_0.0000005_bs_16.png")
#load_checkpoint(model,'./checkpoint/best_v2.pth.tar')
#testing the model
# Evaluate the model

y_t =[]
y_s=[]
# Keep track of dev loss
test_loss = []
test_acc1 = []
# Keep track of accuracy measurements

model.eval()
with torch.no_grad():
    for batch_idx, (image, label) in enumerate(test_loader):
        if USE_GPU:
            data, target = image.cuda(), label.cuda()
        else:
            data, target = image, label
        # Get predicted output
        output = model(data)
        # Update target to be the same dimensions as output
        target = target.view(output.shape[0], 1).float()
    
        #print(output[0])
        # Get accuracy measurements
        test_acc = test_accuracy(output, target,y_t,y_s)
       
        # Calculate the batch's loss
        curr_test_loss = criterion(output, target)
        test_loss.append(curr_test_loss.item())
        test_acc1.append(test_acc.item())
        

# Calculate average loss
avg_test_loss = np.mean(np.array(test_loss))
avgtest_acc = np.mean(np.array(test_acc1))

# performance of model on test data


print('Avg. test Loss: {:.4f}; test Acc: {:.3f}; \n'
    .format( avg_test_loss, avgtest_acc ))


a=roc_auc_score(y_t,y_s)
print("roc_auc_score:",a)
lr_fpr, lr_tpr, _ = roc_curve(y_t,y_s)
plt.plot(lr_fpr, lr_tpr, marker='.', label='simple_cnn')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig("cnn.png")
# calculate precision and recall for each threshold
lr_precision, lr_recall, _ = precision_recall_curve(y_t,y_s)
# calculate scores
lr_f1, lr_auc = f1_score(y_t,y_s), auc(lr_recall, lr_precision)
# summarize scores
print('cnn: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_t,y_s))

