# Deepfake Detection using AlexNet and Meso-4 CNN models

### This is my final-year university project where I've taken the challenge to implement the machine learning models to detect whether a face in an image is geuinne or not

<br>

## This program

- Trains the AlexNet and Meso-4 models using a dataset
- Creates the weights that are used to detect whether the image is genuine or not
- Displays the results either in text file or using plots

Here is an example of what the program shows if images are selected for prediction:

![alt text](https://raw.githubusercontent.com/tajind/Deepfake-Detection/main/assets/AlexnetFigure-04-23-2022%2016-29-50.png)

<br>

## The progam files do as such

- classifier.py hosts the 2 classifier classes
- utils.py hosts the functions for any sort of operations
- traning.py hosts the traning code for the models
- detection.py hosts the detection code for the model
- main.py hosts the UI framework to link everything together through UI

<br>

```
Required libraries to run the program
- TensorFlow & Keras
- numpy
- pandas
- PIL
- matplotlib
- seaborn
- sklearn
- tqdm
```

<br>

#### Resources used

```
Mesonet CNN Paper - https://arxiv.org/pdf/1809.00888.pdf  

Deepfake detection methods - https://ieeexplore.ieee.org/document/9039057

GoogleNet Paper - https://arxiv.org/pdf/1409.4842.pdf

MAD Deepfake - https://christoph-busch.de/projects-mad.html

GAN Information - https://arxiv.org/pdf/1406.2661.pdf | https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/

Perceptron - https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron
```

<br>


## Thank you for reading!