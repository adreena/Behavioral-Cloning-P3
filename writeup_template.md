# **Behavioral Cloning** 

The goals / steps of this project are the following:

* Step 1: collecting trainig/validation data from driving simulator
* Step 2: data augmentation using left/right/flipped images and correcting steering angle 
* Step 3: using generators to iterate through the data on the fly instead of storing them in memory.
* Step 4: preprocessing data 
* Step 4: implementing NVIDIA architecture model to train the model
* Step 4: optimizing model with AdamOptimizer to minimize mean squared error
* Step 5: evaluating the model based on validation set
* Step 8: visualizing error loss of training and validation data to monitor over-fitting/under-fitting situations
* Step 6: saving the model and feeding it to drive.py to drive simulator in autonomous mode
* goal : achieving a high accuracy regression model to drive car simulator autonomously 

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Files Submitted & Code Quality

#### 1. Required files to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* solution.ipynb notebook file of the experiment
* writeup_report.md or writeup_report.pdf summarizing the results

Extra experiment:
* Ran the model on 70,000 data of both tracks in a separate model (same architecture) to test how it performs on the 2nd track
* As this amount of data doesn't fit in EC2 instance, I uploaded them to a S3 bucket to eliminate memory restriction. Although it enabled me to train the model with more data, it slowed down my model significantly becuase of reading them from s3 bucket.

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 

### Model Architecture and Training Strategy

#### 1. Data Collection & Augmentation

Training data for keeping the vehicle driving on the road are combination of these main categories: 
  * driving 3,4 laps in the center of road on track #1 (both driving clockwise and counter-clockwise)
  * recovery laps consisting of data from car just returning to the center of the lanes 
  * for each row of data in driving_log.csv , left and right image are also considered as a training data point with +/-0.2 steering angle correction
  * center image is also flipped with angle correction : -1x(steering-angle) of center image

#### 2. Model architecture

This model consists of 7 convolution neural network layers followed by 4 flat layers (including output layer)
Collected data (32144) is splitted into training data/ validation data by 20%, the data is normalized through a Lambda layer to fit in range [-0.5,0.5] before feeding into the first convoluitonal layer.
Images are then cropped from top(70px) and bottom(25px) because these areas don't contain important information for trainign the model, it also helps reduce image size and network complexity.
Conv Layer 1: applies 5x5 filter with filter-depth=24, it's subsampled (maxpool) and activated with a RELU layer to introduce nonlinearity.
Conv Layers 2,3,4 & 5 follow simillar pattern to layer 1 but with deeper fitler size : 36, 48 , 64 & 64
Flat Layer: outputs should be reshaped to flat as they are feeding a fully connected layer
Dense Layer 1,2,3: fully connected layer consist of 100 & 50 & 10 nodes
Model uses Adam optimizer with 5 epochs and batch_size 32

#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
