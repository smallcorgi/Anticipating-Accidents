# Anticipating Accidents in Dashcam Videos
By Fu-Hsiang Chan, Yu-Ting Chen, Yu Xiang, Min Sun.

### Introduction

Anticipating Accidents in Dashcam Videos is initially described in a [ACCV 2016 paper](https://drive.google.com/file/d/0ByuDEGFYmWsbNkVxcUxhdDRVRkU/view).
We propose a Dynamic-Spatial-Attention (DSA) Recurrent Neural Network (RNN) for anticipating accidents in dashcam videos.

### Requirements

##### Tensoflow 0.12.1
##### Opencv 2.4.9
##### Matplotlib
##### Numpy

### Model Flowchart
![Alt text](./img/flowchart.png "Optional title")


### Dataset & Features

Dataset : [link](http://aliensunmin.github.io/project/dashcam/) (Download the file and put it in "datatset/videos" folder.)

CNN features : [link](https://drive.google.com/open?id=0B8xi2Pbo0n2gaG84ZTNKMXZtbGc) (Download the file and put it in "dataset/features" folder.)

If you need the ground truth of object bounding box and accident location, you can download it.

The format of annotation:

<image name, track_ID, class , x1, y1, x2, y2, 0/1 (no accident/ has accidnet)>

Annotation : [link]()

### Usage

#### Run Demo
```
python accident.py --model ./demo_model/demo_model
```

#### Training
```
python accident.py --mode train --gpu gpu_id
```

#### Testing
```
python accident.py --mode test --model model_path --gpu gpu_id
```

### Citing

Please cite this paper in your publications if you use this code for your research:

    @inproceedings{chan2016anticipating,
        title={Anticipating accidents in dashcam videos},
        author={Chan, Fu-Hsiang and Chen, Yu-Ting and Xiang, Yu and Sun, Min},
        booktitle={Asian Conference on Computer Vision},
        pages={136--153},
        year={2016},
        organization={Springer}
    }
