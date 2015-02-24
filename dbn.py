#!/usr/bin/python 
# coding: utf-8

# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2

dataset = datasets.fetch_mldata("MNIST Original")

(trainX, testX, trainY, testY) = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.90)

# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)

dbn = DBN(
	[trainX.shape[1], 300, 10],
	learn_rates = 0.3,
	learn_rate_decays = 0.9,
	epochs = 10,
	verbose = 1)

dbn.fit(trainX, trainY)

preds = dbn.predict(testX)
print classification_report(testY, preds)

# randomly select a few of the test instances
for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
	# classify the digit
	pred = dbn.predict(np.atleast_2d(testX[i]))
 
	# reshape the feature vector to be a 28x28 pixel image, then change
	# the data type to be an unsigned 8-bit integer
	image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
 
	# show the image and prediction
	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
	cv2.imshow("Digit", image)
	cv2.waitKey(0)