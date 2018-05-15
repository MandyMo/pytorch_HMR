## Pre-reqs(this code is only used for research only, any other purpose please contact me.
### training step
1. download the AI challenger keypoint dataset(see the following link.)
```
[AI challenger] https://challenger.ai/datasets/keypoint
```

2. download the lsp dataset
```
[lsp] https://pan.baidu.com/s/1BgKRJfggJcObHXkzHH5I5A
```
3. download the lsp_ext dataset
```
[lsp exp] https://pan.baidu.com/s/1uUcsdCKbzIwKCc9SzVFXAA
```

4. download the COCO_2017 dataset
```
[COCO_2017] http://cocodataset.org/
```

5. download the mpi_inf_3dhp dataset
```
[mpi_inf_3dhp] https://pan.baidu.com/s/1XQZNV3KPtiBi5ODnr7RB9A
```

6. download the human3.6m dataset
```
I split it into 6 parts.(5 parts about image, 1 part about annotations)

```

7. download the mosh dataset
```
[mosh] https://pan.baidu.com/s/1OWzeMeLS5tKx1XGAiyZ0XA
```

8. extract the model.zip file

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
