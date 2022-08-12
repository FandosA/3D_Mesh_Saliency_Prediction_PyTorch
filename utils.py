# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:50:04 2022

@author: andre
"""

import os
import re
import math
import json
import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Round numbers
def round_(num):
    
    int_part = math.floor(num)
    decimal_part = num - int_part
    
    if decimal_part < 0.5:
        return int(math.floor(num))
    else:    
        return int(math.ceil(num))


# MQE loss function
def mqe_loss(y_pred, y_true):
    
    squared_error = (y_pred - y_true) ** 4
    sum_squared_error = torch.sum(squared_error)
    loss = sum_squared_error / y_true.size(dim=0)
    
    return loss



# Plot loss of the training
def plot_loss(log_dir):
    
    network_files = os.listdir(log_dir)
    
    train_loss_file = [string for string in network_files if 'train_losses' in string]
    train_losses = np.loadtxt(log_dir + '/' + train_loss_file[0])
    
    validation_loss_file = [string for string in network_files if 'val_losses' in string]
    validation_losses = np.loadtxt(log_dir + '/' + validation_loss_file[0])
    
    bestEpoch = validation_loss_file[0].split('_')
    bestEpoch = bestEpoch[-1]
    bestEpoch = bestEpoch.split('.')
    bestEpoch = bestEpoch[0]
    bestEpoch = int(re.search(r'\d+', bestEpoch).group())
    
    epochs = np.arange(train_losses.shape[0])
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss", c='b')
    plt.plot(bestEpoch, train_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.01, train_losses[bestEpoch]+.01, str(bestEpoch) + ' - ' + str(round(train_losses[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(log_dir + '/train_loss.png')
    
    plt.figure()
    plt.plot(epochs, validation_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, validation_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.001, validation_losses[bestEpoch]+.001, str(bestEpoch) + ' - ' + str(round(validation_losses[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(log_dir + '/val_loss.png')
    
    plt.show()
    
    
def plot_predicts(xyz, fixmap_original, predictions, pointcloud_name):

    grados = np.pi / 2
    rot_matrix_x = np.array([[1, 0, 0], [0, np.cos(grados), -1*np.sin(grados)], [0, np.sin(grados), np.cos(grados)]])

    xyz = rot_matrix_x @ xyz.T
    xyz = xyz.T

    min_size = 10
    max_size = 100
    sizes = (fixmap_original - np.amin(fixmap_original)) * (max_size - min_size) / (np.amax(fixmap_original) - np.amin(fixmap_original)) + min_size
    
    fig3D = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array([np.amax(fixmap_original), np.amin(fixmap_original)])
    fig3D.colorbar(m).set_label('seconds')
    plt.title("Saliency of the original model")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='.', c=fixmap_original, cmap='rainbow', s=sizes)
    ax.grid(False)
    plt.axis('off')
    plt.draw()
    
    fig3D = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array([np.amax(predictions), np.amin(predictions)])
    fig3D.colorbar(m).set_label('seconds')
    plt.title("Saliency of the predicted model")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='.', c=predictions, cmap='rainbow')
    ax.grid(False)
    plt.axis('off')
    plt.draw()
    
    plt.show()
    
    
    
def plot_predicts_together(xyz, fixmap_original, predictions, pointcloud_name):
    
    pointcloud_name_ = pointcloud_name.split('_')
    pointcloud_name_ = pointcloud_name_[0]
    
    # Rotate points
    if pointcloud_name_ == 'bunny':
        
        grados = math.pi / 4.25 - math.pi / 2
        rot_matrix_x = np.array([[1, 0, 0], [0, math.cos(grados), -1*math.sin(grados)], [0, math.sin(grados), math.cos(grados)]])
        points_rotated = rot_matrix_x @ xyz.T
        xyz = points_rotated.T
        
        grados = 2.5
        rot_matrix_z = np.array([[math.cos(grados), -1*math.sin(grados), 0], [math.sin(grados), math.cos(grados), 0], [0, 0, 1]])
        points_rotated = rot_matrix_z @ xyz.T
        xyz = points_rotated.T
        
        grados = -0.3
        rot_matrix_x = np.array([[1, 0, 0], [0, math.cos(grados), -1*math.sin(grados)], [0, math.sin(grados), math.cos(grados)]])
        points_rotated = rot_matrix_x @ xyz.T
        xyz = points_rotated.T
    
    if pointcloud_name_ == 'camel':
        
        grados = math.pi / 2
        rot_matrix_x = np.array([[1, 0, 0], [0, math.cos(grados), -1*math.sin(grados)], [0, math.sin(grados), math.cos(grados)]])
        points_rotated = rot_matrix_x @ xyz.T
        xyz = points_rotated.T
        
        grados = -0.3
        rot_matrix_z = np.array([[math.cos(grados), -1*math.sin(grados), 0], [math.sin(grados), math.cos(grados), 0], [0, 0, 1]])
        points_rotated = rot_matrix_z @ xyz.T
        xyz = points_rotated.T
    
    
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.4))
    
    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array([np.amax(fixmap_original), np.amin(fixmap_original)])
    fig.colorbar(m)
    plt.title("Saliency of the original model")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='.', c=fixmap_original, cmap='rainbow')
    
    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array([np.amax(predictions), np.amin(predictions)])
    fig.colorbar(m)
    plt.title("Saliency of the predicted model")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='.', c=predictions, cmap='rainbow')
    
    pointcloud_name_ = pointcloud_name.split('.')
    pointcloud_name_ = pointcloud_name_[0]
    plt.savefig('model_PerSubject_20k/figures/' + pointcloud_name_)
    
    plt.show()