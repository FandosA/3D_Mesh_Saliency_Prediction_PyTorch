import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from mpl_toolkits.mplot3d import Axes3D
        

if __name__ == '__main__':
    
    dataset = 'dataset_test/'
    models = os.listdir(dataset)
    
    for model in models:
    
        model3D = np.loadtxt(dataset + model)
        
        pointcloud = model3D[:, :3]
        fixmap = model3D[:, -2]
        fixmap_bin = model3D[:, -1]
        
        fig3D = plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        m = cm.ScalarMappable(cmap=cm.rainbow)
        m.set_array([np.amax(fixmap), np.amin(fixmap)])
        fig3D.colorbar(m).set_label('seconds')
        plt.title("Saliency agregated normalized")
        ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], marker='.', c=fixmap, cmap='rainbow')
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
        m.set_array([np.amax(fixmap), np.amin(fixmap)])
        fig3D.colorbar(m).set_label('seconds')
        plt.title("Saliency agregated binarized")
        ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], marker='.', c=fixmap_bin, cmap='rainbow')
        plt.draw()
        
        plt.show()