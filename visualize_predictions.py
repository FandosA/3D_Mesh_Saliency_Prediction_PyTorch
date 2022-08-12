import os
import numpy as np
import configargparse
from utils import plot_predicts, plot_predicts_together, plot_loss


def compute_accuracy_binarization(fixmap_original, fixmap_predicted):
    
    acc_total = (fixmap_original==fixmap_predicted).astype(int)
    occurrences_total = np.count_nonzero(acc_total == 1)
    print('Total accuracy:', str((occurrences_total/fixmap_original.shape[0])*100) + '%')
    
    points_succeed = np.where(fixmap_original == 1)
    points_failed = np.where(fixmap_original == 0)

    acc_points_succeed = (fixmap_original[points_succeed]==fixmap_predicted[points_succeed]).astype(int)
    occurrences_points_succeed = np.count_nonzero(acc_points_succeed == 1)
    print('Accuracy out of interesting points:', str((occurrences_points_succeed/len(points_succeed[0]))*100) + '%')
    
    acc_points_failed = (fixmap_original[points_failed]==fixmap_predicted[points_failed]).astype(int)
    occurrences_points_failed = np.count_nonzero(acc_points_failed == 1)
    print('Accuracy out of non-interesting points:', str((occurrences_points_failed/len(points_failed[0]))*100) + '%\n')
    
    

def load_test_pointcloud(dataset_path, log_dir, pointcloud_name, binarization):
    
    model = np.loadtxt(dataset_path + '/' + pointcloud_name)
    
    # Load and normalize the pointcloud between -1 and 1 and center it at the origin 
    xyz = model[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / m
    
    # Load the predicted fixmap by the network
    fixmap_predicted = np.loadtxt(log_dir + '/predictions/' + pointcloud_name)
    
    # Load and normalize original fixmap between 0 and 1
    if binarization:
        fixmap_original = model[:, -1]
        compute_accuracy_binarization(fixmap_original, fixmap_predicted)
    else:
        fixmap = model[:, -2]
        fixmap_original = (fixmap - np.amin(fixmap)) / (np.amax(fixmap) - np.amin(fixmap))
    
    
    return xyz, fixmap_original, fixmap_predicted




if __name__ == '__main__':
    
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, default='dataset_test',help='Dataset path')
    p.add_argument('--log_dir', type=str, default='saliency_prediction',help='Path to the model to test')
    p.add_argument('--binarization', type=bool, default=False, help='Choose if predictions of the binarized model')
    opt = p.parse_args()
    
    pointclouds = os.listdir(opt.log_dir + '/predictions/')
    
    for pointcloud_name in pointclouds:
    
        xyz, fixmap_original, fixmap_predicted = load_test_pointcloud(opt.dataset_path, opt.log_dir,
                                                                      pointcloud_name, opt.binarization)
    
        plot_predicts(xyz, fixmap_original, fixmap_predicted, pointcloud_name)