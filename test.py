import os
import torch
import numpy as np
import configargparse
from model.pointnet_pp import PointNet_pp



def load_test_pointcloud(dataset_path, pointcloud_name, device):
    
    model = np.loadtxt(dataset_path + '/' + pointcloud_name)
    
    # Normalize the pointcloud between -1 and 1 and center it at the origin 
    xyz = model[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / m
    xyz = torch.from_numpy(xyz)
    xyz = torch.reshape(xyz, (1, xyz.size(dim=0), xyz.size(dim=1)))
    xyz = xyz.float()
    xyz = xyz.to(device)
    
    # Normalize normals
    normals = model[:, 3:6]
    """
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.array([norms]).T
    """
    normals = torch.from_numpy(normals)
    normals = torch.reshape(normals, (1, normals.size(dim=0), normals.size(dim=1)))
    normals = normals.float()
    normals = normals.to(device)
    
    return xyz, normals



def save_predicts(log_dir, pointcloud_name, predictions, binarization):
    
    if not os.path.isdir(log_dir + '/predictions'):
        os.mkdir(log_dir + '/predictions')
        
    predictions = torch.squeeze(predictions)
    predictions = predictions.cpu().detach().numpy()
    
    if binarization:
        predictions = np.where(predictions < 0.5, 0, 1)
        
    np.savetxt(log_dir + '/predictions/' + pointcloud_name, predictions)
    print('Predictions over the model "' + pointcloud_name + '" saved.')



if __name__ == '__main__':
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, default='dataset_test',help='Dataset path')
    p.add_argument('--log_dir', type=str, default='saliency_prediction',help='Name of the folder to load the model')
    p.add_argument('--checkpoint', type=str, default='checkpoint_88_best.pth',help='Checkpoint path')
    p.add_argument('--binarization', type=bool, default=False, help='Choose if loading a binarized model')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device: "gpu" or "cpu"')
    opt = p.parse_args()
    
    
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
    
    
    # Load the model and create the model
    input_size = 6
    output_size = 1
    model = PointNet_pp(input_size, output_size, opt.binarization).to(device)
    
    state_dict = torch.load(opt.log_dir + "/checkpoints/" + opt.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Make predictions over the test pointclouds and save them
    for pointcloud_test in os.listdir(opt.dataset_path):
        
        xyz, normals = load_test_pointcloud(opt.dataset_path, pointcloud_test, device)
        
        # Predict saliency
        predictions = model(xyz, normals)
        
        # Save predictions
        save_predicts(opt.log_dir, pointcloud_test, predictions, opt.binarization)