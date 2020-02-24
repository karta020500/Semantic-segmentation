# Semantic-segmentation
This project uses U-net a deep learning model and based on Apollo dataset and CamVid dataset for semantic segmentation to extract road markings

# Installation
This project has the following dependencies:

-*Numpy pip install numpy*

-*OpenCV Python apt-get install python-opencv*

-*TensorFlow pip install --upgrade tensorflow-gpu*
# Usage
The only thing you have to do to get started is set up the folders in the following structure:
```
├── "dataset_name"                   
|   ├── images├── test
|   |         ├── train
|   ├── labels├── test
|   |         ├── train
```
# Results

| <img src="https://github.com/karta020500/Semantic-segmentation/blob/master/Apollo_data/label/test/171206_025743401_Camera_5.png" width = "500" height = "300" />  |  <img src="https://github.com/karta020500/Semantic-segmentation/blob/master/path_to_predictions/171206_025743401_Camera_5.png" width = "500" height = "300" /> | 
|:-------:|:-----:|
|ground truth|prediction|
|<img src="https://github.com/karta020500/Semantic-segmentation/blob/master/CamVid_data/label/test/Seq05VD_f03210.png" width = "500" height = "300"/> |  <img src="https://github.com/karta020500/Semantic-segmentation/blob/master/CamVid_data/image/train/prediction.png" width = "500" height = "300" />|
|ground truth|prediction|
