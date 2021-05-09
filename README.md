# Detection of COVID-19 from Chest X-rays Using Convolutional Neural Networks

Purpose: To compare the performance of different methodologies featuring machine learning and deep convolutional neural networks (CNN) to correctly detect COVID-19 pneumonia from chest x-ray images

Images need to be loaded in COVID-19 Radiography folder.
 
Run main.py to train model using CXR images. Model and number of epochs can be edited in the file. Output folder will be generated in the same directory with time-stamp.

model.py contains pre-built in models from torchvision.

training.py has code with training loops

cnn_ml_fvectors.py utilizes machine learning + pre-trained CNN models to predict labels from CXR images.
