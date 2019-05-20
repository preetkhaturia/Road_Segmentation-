# Road Segmentation
The objective of this project is to segment out the roads from an image using [Fully Convolutional Network](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). The model is trained using Indian Road Dataset [(IDD)](https://idd.insaan.iiit.ac.in/) which is a collabrative project from  Indian Institute of Information Technology, Hyderabad and Intel. It consists of 10,000 images, finely annotated with 34 classes collected from 182 drive sequences on Indian roads. 

The dataset consists of images obtained from a front facing camera attached to a car. The car was driven around Hyderabad, Bangalore cities and their outskirts. The images are mostly of 1080p resolution, but there is also some images with 720p and other resolutions.

![](https://github.com/preetkhaturia/Road_Segmentation-/blob/master/Images/detection.gif)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. I have also added trained model to predict the segmentation on your images.

## Prerequisites
A enviornment.yml file has been added with the repository in order to reproduce the pacakge selection. 

## Files and Directories 
Main.py :Training file, train your model by tweaking the batch_size, epochs and learning rates. Data augmentation can also improve the accuracy of the data. 

helper.py , project_tests.py : Utility files for Main program.

data_creation.py : Creating labeled images from IDD dataset.

video_implement.py : test your implementation on any video.

saver : contains trained model.


## Usage 
##### Training : Run Main.py 
```
python main.py
```
(tweak with different batch size, epochs, training data, learning rate)
### Prediction : Run video_implement.py
```
clip1 = VideoFileClip(Your video path)
```
### Dataset Creation
![alt text](https://github.com/preetkhaturia/Road_Segmentation-/blob/master/Images/dataset.png "Logo Title Text 1")

Label images should be binary images with 1 for road region and 0 for no road region.
To create the dataset, run data_creation.py file. Add your data in data folder.



