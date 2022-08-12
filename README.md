# Deep Learning For 3D Mesh Saliency Prediction

This project presents a saliency prediction model for point clouds based on deep learning. To the best of my knowledge, this is the first deep learning approach that predicts saliency on point clouds using real saliency data as ground truth, instead of using point-based methods or methods which study the rarity or curvature of a mesh splitting it in clusters of points. An experiment with 32 people was carried out in order to collect the gaze data that served as ground truth. Then a deep neural network based on previous work, [PointNet++](https://dl.acm.org/doi/10.5555/3295222.3295263), was used and adapted to our problem in order to predict 3D saliency on objects. Finally, results were obtained by training this network with the data collected in the experiment. The study proves that the model is able to recognise the most interesting parts of a mesh. This project was my final Master's degree thesis (full document comming soon).


## Implementation

The first step of the project was to find and obtain 3D meshes of different classes. The total number of meshes used in this experiment was 60, and
they were taken from different public databases. The 3D models can be found in the ``Models_3D/`` folder. As the meshes have been made by different people using different design software, they were processed in order to make them be the same size and orientation. All meshes were scaled and centered at the origin using the same method as in [PointNet++](https://dl.acm.org/doi/10.5555/3295222.3295263). To orient the meshes, the software Blender was used. With this process, all meshes are forced to be sized to fit a 2-by-2 meter cube centered at the origin and oriented in the same direction. The image below shows a couple of examples.

<img src="https://user-images.githubusercontent.com/71872419/184395473-d7eda8a2-f0e5-425f-8c17-2e7e5f111814.png" width="600" height="500">

The experiment was carried out in virtual reality. This decision was made because of three main reasons: (1) it offers more control over the viewing conditions, (2) it provides stereo viewing and (3) it avoids possible distractions from the real world. The equipment used to perform the experiment consists of the HTC Vive Pro virtual reality headset and the [Pupil Labs](https://pupil-labs.com/) eye tracker. To create the virtual environment in which the experiment was carried out, the Unity game engine was used. The Unity scene is shown below.

<img src="https://user-images.githubusercontent.com/71872419/184403632-1689dcb4-6c46-44a0-a242-35319f33d83a.PNG" width="600" height="500">

When all of this was implemented and everything was working properly, people were called in to carry out the experiment. The gaze data was collected and processed to prepare the dataset. The dataset can be found in the folders ``dataset_train/``, ``dataset_val/`` and ``dataset_test/``. It consists of text files that contain the coordinates of the vertices of the meshes, their normals, the time that each vertex was fixated in the experiment, and the digit 1 or 0 that represents that vertex as interesting or not interesting (in case the problem was binarized). Then the text files, corresponding to each mesh, have the following form: (X, Y, Z, NormX, NormY, NormZ, time, binTerm). The image below shows the fixation maps of three participants and the aggregated fixation map of all the participants. The explanation of why the dataset was split in this way and other project details can be found in the document.

<img src="https://user-images.githubusercontent.com/71872419/184412340-42042cbb-049c-4052-97ae-5bc6472d5629.png"  width="400" height="540">

The next step was to modify the [PointNet++](https://dl.acm.org/doi/10.5555/3295222.3295263) network to adapt it to this problem. For that, the size of the output layer was set to 1 (since each vertex has only one value assigned, its fixation map), the activation of the output layer was changed to ReLU (since only values that are 0 or larger than 0 are desired) and the loss function was changed to the Mean Squared Error (since this is a regression problem with continuous values as labels). Once all of this was ready, the network was trained tested. The results obtained for the 4 test meshes is shown below.

![imagen](https://user-images.githubusercontent.com/71872419/184414146-58a6a84a-3570-4e88-83f5-6d2b75f7b8e2.png)
![imagen](https://user-images.githubusercontent.com/71872419/184414213-6e7efdcf-83f1-4dec-970d-dbee2fe01c63.png)


## Run the implementation

To train the model, open a terminal and go to the folder in which you have the project. You can change the default parameters by modifying the ``train.py`` file and then running it, or setting them in the terminal when running it. I recommend the first option, modify the file and then run in the terminal
```
python train.py
```

The training will start and the checkpoints throughout the training will be saved in the folder specified by the ``--log_dir`` parameter. To test the model, set the ``--log_dir`` parameter to the same name as in the training file, and also set the ``--checkpoint`` parameter to the checkpoint with which you want to make predictions. The run
```
python test.py
```

