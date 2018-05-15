## Pre-reqs(this code is only used for research only, any other purpose please contact me.

### training step
  #### 1. download the following datasets.
    1. [AI challenger keypoint dataset](https://challenger.ai/datasets/keypoint)
    2. [lsp 14-keypoint dataset](https://pan.baidu.com/s/1BgKRJfggJcObHXkzHH5I5A)
    3. [lsp 14-keypoint extension dataset](https://pan.baidu.com/s/1uUcsdCKbzIwKCc9SzVFXAA)
    4. [COCO-2017-keypoint dataset](http://cocodataset.org/)
    5. [mpi_inf_3dhp 3d keypoint dataset](https://pan.baidu.com/s/1XQZNV3KPtiBi5ODnr7RB9A)
    6. [mosh dataset, which used for adv training](https://pan.baidu.com/s/1OWzeMeLS5tKx1XGAiyZ0XA)

  #### 2. download the following datasets.
    1. to be continue

  ### 3. unzip the downloaded datasets.
  
  ### 4. unzip the model.zip
  ### 5. config the softward environment by modify the src/config.py and do_train.sh
  ### 6. run ./do_train.sh directly

9. install pytorch0.4 and some other required packages such as numpy、scipy、h5py and so on.

10. please read the config.py carefully and change the correspond config.(both resnet50 and hourglass are supported for encoder)

11. run do_train.sh(maybe you should change it's config, if you don't has so many GPU.)

### reference papers
1. [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937)
2. [SMPL: A Skinned Multi-Person Linear Model](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)
3. [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://pdfs.semanticscholar.org/4cea/52b44fc5cb1803a07fa466b6870c25535313.pdf)
4. [motion and shape capture from sparse markers](http://files.is.tue.mpg.de/black/papers/MoSh.pdf)
5. [Unite the People: Closing the Loop Between 3D and 2D Human Representations](https://arxiv.org/abs/1701.02468)
6. [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/abs/1712.06584)

### reference resources
1. [up-3d dataset](http://files.is.tuebingen.mpg.de/classner/up/)
2. [coco-2017 dataset](http://cocodataset.org/)
3. [human 3.6m datas](http://vision.imar.ro/human3.6m/description.php)
4. [ai challenger dataset](https://challenger.ai/)
