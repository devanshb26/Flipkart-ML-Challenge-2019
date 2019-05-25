# Flipkart GRiD – Te[a]ch The Machines | 2019

This years challenge was on Object Localization in images.More specifically we had to find the bounding box prediction around an object in  in an image.There was only single object in an image.

Given an image we had to predict the coordinates of the bounding box around the object in that image.During testing the Intersection over union (IOU) was calculated to check the performance of the model.

The original size of image was very high so it was scaled down and greyscaled before training.

# Training Set
It consisted of the image and corresponding coordinates of the bounding box around the object.
Here is the link to the training images which is saved as pickle file due to its large size.
The images were scaled down to size 64X64 before saving to a pickle file
https://drive.google.com/open?id=18tYbNC5vOA-nF5uTywTgLS9WxDhH5oRP

Training set csv file containg the coordinates of the images after being rescaled.
https://drive.google.com/open?id=1odTu4xMmVppXeNnduTE1oG3FR91V2OGa



# Test Set
The images were saved and uploaded as a pickle file on drive and them later used in code. 
https://drive.google.com/open?id=1AKWYNF5LNecWAg0ZBo0CVcfVglBejU6k
The test images were also resized to 64X64


We have used keras library which uses tenserflow in backend
We created a custom neural net having the following layers

# ModelSummary:-


Inspired by VGG-16 with some hyperparameter tuning

_________________________________________________________________
###### Layer (type)                 Output Shape              Param    
=================================================================
input_1 (InputLayer)         (None, 64, 64, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 32)        320       
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 32)        128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 64)        18496     
_________________________________________________________________
batch_normalization_2 (Batch (None, 64, 64, 64)        256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 64, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 32)        18464     
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 32, 64)        18496     
_________________________________________________________________
batch_normalization_4 (Batch (None, 32, 32, 64)        256       
_________________________________________________________________
dropout_2 (Dropout)          (None, 32, 32, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
batch_normalization_5 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 16, 16, 64)        73792     
_________________________________________________________________
batch_normalization_6 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
batch_normalization_7 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 16, 16, 256)       295168    
_________________________________________________________________
batch_normalization_8 (Batch (None, 16, 16, 256)       1024      
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 16, 16, 128)       295040    
_________________________________________________________________
batch_normalization_9 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 16, 16, 256)       295168    
_________________________________________________________________
batch_normalization_10 (Batc (None, 16, 16, 256)       1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 16, 16, 256)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 65536)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 2048)              134219776 
_________________________________________________________________
batch_normalization_11 (Batc (None, 2048)              8192      
_________________________________________________________________
dense_2 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
batch_normalization_12 (Batc (None, 2048)              8192      
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              2098176   
_________________________________________________________________
batch_normalization_13 (Batc (None, 1024)              4096      
_________________________________________________________________
##### dense_4 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 141,706,148
Trainable params: 141,693,604
Non-trainable params: 12,544

We used the 'mse'(L2) loss function , optimizer = adadelta and the trained the network for 100 epochs.
Finally we saved the results into test.csv file

We got an accuracy of around 87% on the test data
All other details are described as comments in the source(ipynb) file

