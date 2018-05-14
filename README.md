## Pre-reqs(this code is only used for research only, any other purpose please contact me.

1. download the AI challenger keypoint dataset(see the following link.)
```
[AI challenger] https://challenger.ai/datasets/keypoint
```

2. download the lsp dataset
```
[lsp] http://sam.johnson.io/research/lsp.html
```
3. download the lsp_ext dataset
```
[lsp exp] http://sam.johnson.io/research/lsp.html
```

4. download the COCO_2017 dataset
```
[COCO_2017] https://coco.com
```

5. download the mpi_inf_3dhp dataset
```
[mpi_inf_3dhp] https://mpi_inf_3dhp
```

6. download the human3.6m dataset
```
[human_4.6m] https://human3.6m
```

7. download the mosh dataset
```
[mosh] https://mosh.com
```

8. extract the model.zip file

9. install pytorch0.4 and some other required packages such as numpy、scipy、h5py and so on.

10. please read the config.py carefully and change the correspond config.(current resnet50 and hourglass is supported for encoder)

11. run do_train.sh(maybe you should change it's config, if you don't has so many GPU.)


If you use any of the MoSh data, please cite: 
```
article{Loper:SIGASIA:2014,
  title = {{MoSh}: Motion and Shape Capture from Sparse Markers},
  author = {Loper, Matthew M. and Mahmood, Naureen and Black, Michael J.},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {33},
  number = {6},
  pages = {220:1--220:13},
  publisher = {ACM},
  address = {New York, NY, USA},
  month = nov,
  year = {2014},
  url = {http://doi.acm.org/10.1145/2661229.2661273},
  month_numeric = {11}
}
```
