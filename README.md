## HMR
<p align="center">
 <img src="./images/paper_picked.png" width="800px">
</p>

This is a **pytorch** implementation of [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/abs/1712.06584) by *Angjoo Kanazawa, Michael J. Black, David W. Jacobs*, and *Jitendra Malik*, accompanying by some famous human pose estimation networks and datasets. 
HMR is an end-to end framework for reconstructing a full 3D mesh of a human body from a single RGB image. In contrast to most current methods that compute 2D or 3D joint locations, HMR produce a richer and more useful mesh representation that is parameterized by shape and 3D joint angles. The main objective is to minimize the reprojection loss of keypoints, which allow model to be trained using in-the-wild images that only have ground truth 2D annotations.

## training step
#### 1. download the following datasets.
- [AI challenger keypoint dataset](https://challenger.ai/datasets/keypoint)
- [lsp 14-keypoint dataset](https://pan.baidu.com/s/1BgKRJfggJcObHXkzHH5I5A)
- [lsp 14-keypoint extension dataset](https://pan.baidu.com/s/1uUcsdCKbzIwKCc9SzVFXAA)
- [COCO-2017-keypoint dataset](http://cocodataset.org/)
- [mpi_inf_3dhp 3d keypoint dataset](https://pan.baidu.com/s/1XQZNV3KPtiBi5ODnr7RB9A) 
- [mosh dataset, which used for adv training](https://pan.baidu.com/s/1OWzeMeLS5tKx1XGAiyZ0XA)
#### 2. download human3.6 datasets.
    1. to be continue
#### 3. unzip the downloaded datasets.
#### 4. unzip the model.zip
#### 5. config the softward environment by modify the src/config.py and do_train.sh
#### 6. run ./do_train.sh directly

### environment configurations.
  - install **pytorch0.4**
  - install torchvision
  - install numpy
  - install scipy
  - install h5py
  - install opencv-python
  


### reference papers
- [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937)
- [SMPL: A Skinned Multi-Person Linear Model](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)
- [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://pdfs.semanticscholar.org/4cea/52b44fc5cb1803a07fa466b6870c25535313.pdf)
- [motion and shape capture from sparse markers](http://files.is.tue.mpg.de/black/papers/MoSh.pdf)
- [Unite the People: Closing the Loop Between 3D and 2D Human Representations](https://arxiv.org/abs/1701.02468)
- [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/abs/1712.06584)

### reference resources
- [up-3d dataset](http://files.is.tuebingen.mpg.de/classner/up/)
- [coco-2017 dataset](http://cocodataset.org/)
- [human 3.6m datas](http://vision.imar.ro/human3.6m/description.php)
- [ai challenger dataset](https://challenger.ai/)
