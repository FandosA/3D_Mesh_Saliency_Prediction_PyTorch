# Deep Learning For 3D Mesh Saliency Prediction

This project presents a saliency prediction model for point clouds based on deep learning. To the best of my knowledge, this is the first deep learning approach that predicts saliency on point clouds using real saliency data as ground truth, instead of using point-based methods or methods which study the rarity or curvature of a mesh splitting it in clusters of points. An experiment with 32 people was carried out in order to collect the gaze data that served as ground truth. Then a deep neural network based on previous work, [PointNet++](https://dl.acm.org/doi/10.5555/3295222.3295263), was used and adapted to our problem in order to predict 3D saliency on objects. Finally, results were obtained by training this network with the data collected in the experiment. The study proves that the model is able to recognise the most interesting parts of a mesh. This project was my final Master's degree thesis (full document comming soon).


## Implementation

The first step of the project was to find and obtain 3D meshes of different classes. The total number of meshes used in this experiment was 60, and
they were taken from different public databases. As the meshes have been made by different people using different design software, they were processed in order to make them be the same size and orientation. All meshes were scaled and centered at the origin using the same method as in [PointNet++](https://dl.acm.org/doi/10.5555/3295222.3295263). To orient the meshes, the software Blender was used. With this process, all meshes are forced to be sized to fit a 2-by-2 meter cube centered at the origin and oriented in the same direction. The image below shows a couple of examples fo the resulting meshes.

<img src="https://user-images.githubusercontent.com/71872419/184395473-d7eda8a2-f0e5-425f-8c17-2e7e5f111814.png" width="600" height="500">
