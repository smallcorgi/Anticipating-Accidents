# Anticipating Accidents in Dashcam Videos
By Fu-Hsiang Chan, Yu-Ting Chen, Yu Xiang, Min Sun

### Introduction

We release the code for Anticipating Accidents in Dashcam Videos. 

### Requirements

* Tensoflow 0.12.1
* Opencv 2.4.9
* Matplotlib
* Numpy

### Model Flowchart
![Alt text](./img/flowchart.png "Optional title")

### Usage

#### The Folder for Demo
* features : [CNN features](https://drive.google.com/open?id=0B8xi2Pbo0n2gaG84ZTNKMXZtbGc) (Download the file and put it in "features" folder.)

* images : Some images for demo.

* model : Model.

#### Run Demo
```
python accident.py
```

#### Training
```
python accident.py --mode train
```

#### Testing
```
python accident.py --mode test --model model_path
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
