# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:19:32 2021

@author: andre
"""

import os
import json
import torch
import numpy as np
import configargparse
import torch.nn as nn
from utils import plot_loss #, round_
from model.pointnet_pp import PointNet_pp
from torch.utils.data import DataLoader, Dataset



class Dataset(Dataset):
    
    def __init__(self, dataset_path, binarization, device):
        
        self.dataset_path = dataset_path + '/'
        self.models = os.listdir(dataset_path)
        self.binarization = binarization
        self.device = device
        
    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        
        # Load model
        model = np.loadtxt(self.dataset_path + self.models[index])
        
        # Normalize the pointcloud between -1 and 1 and center it at the origin 
        pointcloud = model[:, :3]
        
        # Normalize normals
        normals = model[:, 3:6]
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / np.array([norms]).T
        
        # Concatenate points and normals and convert them to pytorch Tensor
        pointcloud_normals = np.hstack((pointcloud, normals))
        pointcloud_normals = torch.from_numpy(pointcloud_normals)
        pointcloud_normals = pointcloud_normals.float()
        pointcloud_normals = pointcloud_normals.to(self.device)
        
        # Normalize fixmaps between 0 and 1 and convert them to pytorch Tensor
        if self.binarization:
            fixmap = model[:, -1]
        else:
            fixmap = model[:, -2]
            fixmap = (fixmap - np.amin(fixmap)) / (np.amax(fixmap) - np.amin(fixmap))
            
        fixmap = torch.from_numpy(fixmap)
        fixmap = fixmap.float()
        fixmap = fixmap.to(self.device)
        
        return pointcloud_normals, fixmap



def save_model_features(model, log_dir, train_dataset_path, validate_dataset_path):
    
    parent_dir = log_dir
    os.mkdir(parent_dir)
    
    checkpoints_path = 'checkpoints/'
    checkpoints_path = os.path.join(parent_dir, checkpoints_path)
    os.mkdir(checkpoints_path)
    
    figures_path = 'figures/'
    figures_path = os.path.join(parent_dir, figures_path)
    
    # Save model features
    model_features = open(log_dir + '/model_features.txt', 'w+')
    model_features.write(str(model))
    model_features.close()
    
    pointcloud_division = {
        "training": [],
        "validation": []
    }
    
    shapes_train = os.listdir(train_dataset_path)
    pointcloud_division["training"].append(shapes_train)
    
    shapes_val = os.listdir(validate_dataset_path)
    pointcloud_division["validation"].append(shapes_val)
    
    with open(log_dir + '/models_split.json', "w") as fp:
        json.dump(pointcloud_division, fp, indent = 4)
    
    print('Model features and training configuration saved.')

    return checkpoints_path, figures_path



def train(model, train_loader, validate_loader, loss_fn, optimiser,
          num_epochs, batch_size, learning_rate, device, log_dir,
          checkpoints_path, figures_path, scheduler):
    
        
    min_valid_loss = np.inf
    train_losses = []
    val_losses = []
    bestEpoch = 0
    
    print('\n--------------------------------------------------------------')
    
    # Loop along epochs to do the training
    for i in range(num_epochs):
        
        print(f'EPOCH {i+1}')
        
        # First do the training loop
        train_loss = 0.0
        model.train()
        iteration = 1
        
        print('\nTRAINING')
        
        for pointcloud, fixmap in train_loader:
            
            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1
            
            optimiser.zero_grad()
            
            pointcloud, fixmap = pointcloud.to(device), fixmap.to(device)
            
            xyz = pointcloud[:, :, :3]
            normals = pointcloud[:, :, 3:]
            
            prediction = model(xyz, normals)
            
            # Backpropagate error and update weights
            loss = loss_fn(torch.squeeze(prediction), torch.squeeze(fixmap))
            
            loss.backward()
            optimiser.step()
            
            train_loss += loss.item()
        
        
        # Second do the validation loop
        valid_loss = 0.0
        model.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')
        
        for pointcloud, fixmap in validate_loader:
            
            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(validate_loader)), end='')
            iteration += 1
            
            pointcloud, fixmap = pointcloud.to(device), fixmap.to(device)
            
            xyz = pointcloud[:, :, :3]
            normals = pointcloud[:, :, 3:]
            
            prediction = model(xyz, normals)
            
            loss = loss_fn(torch.squeeze(prediction), torch.squeeze(fixmap))
            
            valid_loss += loss.item()
        
        # Save the loss values
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        
        print('')
        print(f'\nTraining Loss: {train_loss} \nValidation Loss: {valid_loss}')
        
            
        # Save the model each 10 epochs
        if i % 10 == 0:
            torch.save(model.state_dict(), checkpoints_path + "/checkpoint_" + str(i) + ".pth")
            
        # Besides, save the best model
        if valid_loss < min_valid_loss:
            
            # Remove previous best model and save current best model
            if i == 0:
                torch.save(model.state_dict(), checkpoints_path + "checkpoint_" + str(i) + "_best.pth")
            else:
                os.remove(checkpoints_path + "checkpoint_" + str(bestEpoch) + "_best.pth")
                torch.save(model.state_dict(), checkpoints_path + "checkpoint_" + str(i) + "_best.pth")
                
                # Delete previous loss files to update the txt file with the best epoch
                os.remove(log_dir + '/train_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/val_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt')
            
            print(f'\nValidation Loss decreased ({min_valid_loss:.6f} ---> {valid_loss:.6f}) \nModel saved')
            
            # Update parameters with the new best model
            min_valid_loss = valid_loss
            bestEpoch = i
            
        print("--------------------------------------------------------------")
        
        
        # Save losses
        np.savetxt(log_dir + '/train_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(train_losses))
        np.savetxt(log_dir + '/val_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(val_losses))
        
        
        scheduler.step()
    
    # Plot losses
    plot_loss(log_dir)



if __name__ == "__main__":
    
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_train', type=str, default='dataset_train', help='Dataset train path')
    p.add_argument('--dataset_val', type=str, default='dataset_val', help='Dataset validation path')
    p.add_argument('--log_dir', type=str, default='saliency_prediction', help='Name of the folder to save the model')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size. Default: 1')
    p.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate. Default = 1e-3')
    p.add_argument('--num_epochs', type=int, default=351, help='Number of epochs. Default = 351')
    p.add_argument('--binarization', type=bool, default=False, help='Choose if binarize the problem or not')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device: "gpu" or "cpu"')
    opt = p.parse_args()
    
    assert not (os.path.isdir(opt.log_dir)), 'The folder log_dir set already exists, remove it or change it'
    
    
    # Select device
    if opt.device == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available() and opt.device == 'gpu':
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
    
    train_dataset = Dataset(opt.dataset_train, opt.binarization, device)
    validate_dataset = Dataset(opt.dataset_val, opt.binarization, device)
    
    # Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=opt.batch_size, shuffle=True)
    
    print('Models used to train: ' + str(len(train_dataset)))
    print('Models used to validate: ' + str(len(validate_dataset)) + '\n')
    
    
    # Create the model
    input_size = train_dataset[0][0].size(dim=1)
    output_size = 1
    model = PointNet_pp(input_size, output_size, opt.binarization).to(device)
    
    
    # Save model features, create checkpoints folder and save the division of the models
    checkpoints_path, figures_path = save_model_features(model, opt.log_dir, opt.dataset_train, opt.dataset_val)
    
    
    # Initialize loss funtion and optimiser
    if opt.binarization:
        loss_fn = nn.BCELoss()
    else:
        loss_fn = nn.MSELoss()
    
    optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.7)


    # Train and save the model
    train(model, train_loader, validate_loader, loss_fn, optimiser,
          opt.num_epochs, opt.batch_size, opt.learning_rate, device,
          opt.log_dir, checkpoints_path, figures_path, scheduler)
