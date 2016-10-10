# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 13:20:33 2015

@author: acq14jp
"""

import numpy as np
import scipy.io
from scipy import signal
from PIL import Image as img
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import random as rnd
import time

def main():

	#Set pretraining to True if weights and bias should be initialised including results from pre-training (improves results)
	pretraining = True
	#Set training to True if you want to train this model, set to false if you want to test it using the last trained model
	training = True

	################## Parameters ##############################
	image_dim = 64  # image dimension
	image_channels = 3  # number of channels (rgb, so 3)
	patch_dim = 8  # patch dimension
	input_units = patch_dim * patch_dim * image_channels  # number of input units
	output_units = input_units   # number of output units
	hidden_units = 400  # number of hidden units
	pool_dim = 19  # dimension of pooling region
	res_dim = (image_dim - patch_dim + 1) / pool_dim
	kernel_size = patch_dim
	difficulty = 1 # 1 = Very Easy, 3 = Easy, 5 = Difficult, 8 = Very Difficult

	###########################################################

	start_time = time.time()

	""" Experiments """

	if (pretraining):
		""" Retrieve pretrained parameters and preprocess them """
		weights, bias, zca_white, mean, opt_parameter_vector = load_features(hidden_units, input_units)
	else:
		""" Initialise weights randomly and bias to 0"""
		interval = np.sqrt(6) / np.sqrt(hidden_units + input_units + 1)
		weights = np.asarray(np.random.uniform(low = -interval, high = interval, size = (hidden_units, input_units)))
		bias = np.zeros((hidden_units, hidden_units))

	""" Load Training and Test Data """
	training_images, training_labels = load_training_data()
	test_images, test_labels = load_test_data()

	""" Initialise Network """
	convnet = ConvolutionalNeuralNetwork(weights, bias, kernel_size, pool_dim)

	if (training == False):
		pass
	else: 
		""" Convolve and Pool through Training and Test Images """
		numFeatures = 25 # Convolution Matrix is very large, hence convolve & pool 50 features at a time, decrease if necessary
		iter = numFeatures / hidden_units
		for i in range(iter):
			weights = weights[iter * numFeatures:(iter + 1) * numFeatures, :]
	    	bias = bias[iter * numFeatures:(iter + 1) * numFeatures]

	    	print '1/3... Convolving & Pooling of training images ....'
	    	convolved_features = convnet.convolution(kernel_size, numFeatures, training_images, weights, bias)
	    	pooled_features = convnet.pooling(pool_dim, convolved_features)
	    	pooled_features_training = pooled_features[iter * numFeatures:(iter + 1) * numFeatures, :, :, :]

	    	print '2/3... Convolving & Pooling of test images ....'
	    	convolved_features = convnet.convolution(kernel_size, numFeatures, test_images, weights, bias)
	    	pooled_features = convnet.pooling(pool_dim, convolved_features)
	    	pooled_features_test = pooled_features[iter * numFeatures: (iter + 1) * numFeatures, :, :, :]

	    	print '3/3... Saving pooled features...'
	    	np.save('pooled_features_training.npy', pooled_features_training)
	    	np.save('pooled_features_test.npy', pooled_features_test)
	    	print "Done saving"
    
	weight_decay = 1e-4
	numClasses = 4
	softmax = SoftmaxClassifier(input_units, numClasses, weight_decay)
	# Train and Test Classifier
	model = softmax.softmax_classify(training_images, training_labels)
	predictions = softmax.test_classifier(model, test_images, test_labels)
	print("--- %s seconds ---" % (time.time() - start_time))

###########################################################################################
""" Data Processing """

def load_features(hidden_units, input_units):
	print '\n'
	print 'Loading features ....'
	opt_parameter_vector  = np.load('Autoencoder_results/opt_parameter_vector.npy')
	zca_white  = np.load('Autoencoder_results/zca_white.npy')
	mean = np.load('Autoencoder_results/mean.npy')
	weights = opt_parameter_vector[0:hidden_units * input_units].reshape(hidden_units, input_units)
	bias = opt_parameter_vector[2 * hidden_units * input_units:2 * hidden_units * input_units + hidden_units]

	#zca white = whitening matrix
	weights = np.dot(weights, zca_white)
	bias = bias - np.dot(weights, mean)

	return weights, bias, zca_white, mean, opt_parameter_vector

def load_training_data():
	print 'Loading training data ....'
	training_set = scipy.io.loadmat('training_set.mat')
	training_images = training_set['trainImages']
	training_labels = training_set['trainLabels']
	training_labels = training_labels

	return training_images, training_labels

def load_test_data():
	print 'Loading test data ....'
	test_set = scipy.io.loadmat('test_set.mat')
	test_images = test_set['testImages']
	test_labels = test_set['testLabels']
	test_labels = test_labels

	return test_images, test_labels

###########################################################################################
""" Convolution Layer """

class ConvolutionalNeuralNetwork(object):

	def __init__(self, weights, bias, kernel_size, pool_dim):

		self.weights = weights
		self.bias = bias
		self.kernel_size = kernel_size
		self.pool_dim = pool_dim

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def convolution(self, kernel_size, hidden_units, input_data, weights, bias):
		print 'Convolving images ....'
		numImages = input_data.shape[3]
		numFeatures = hidden_units
		image_dim = input_data.shape[0]
		image_channels = input_data.shape[2]

		convolved_features = np.zeros((numFeatures, numImages, image_dim-kernel_size+1, image_dim-kernel_size+1), dtype=np.float64)
		# (1) compute weights * given 8*8 patch with the upper left corner at (r,c): 
		# ie. Wx(r,c) for all (r,c)
		#SOURCE : http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling#Step_2a:_Implement_convolution

		for i in range(numImages): 

			for j in range(numFeatures):
				convolved_image = np.zeros((image_dim - kernel_size + 1, image_dim - kernel_size + 1))
				
				for k in range(image_channels):
					kernel = np.zeros((kernel_size, kernel_size))
					kernel = weights[j, kernel_size * kernel_size * k:kernel_size * kernel_size * (k+1)].reshape(kernel_size, kernel_size)
					#flip feature matrix before performing conv2 convolution
					kernel = np.flipud(np.fliplr(kernel))
					#obtain the image
					image = input_data[:, :, k, i]
					#mpimg.imsave("kernel"+str(j-k)+".png", kernel)
					#mpimg.imsave("image"+str(i)+".png", image2)
					#convolve image
					convolved_image += scipy.signal.convolve2d(image, kernel, mode='valid')
				#add bias and sigmoid function
				convolved_image += bias[j, 0]
				#apply sigmoid function
				convolved_image = self.sigmoid(convolved_image)
				#mpimg.imsave("convimage"+str(i)+".png", convolved_image)
				convolved_features[j, i, :, :] = convolved_image
				#mpimg.imsave("convfeature"+str(i)+".png", convolved_image)

		return convolved_features

	###########################################################################################
	""" Pooling Layer """
		
	def pooling(self, pool_dim, convolved_features):
		print 'Pooling convolved features ....'

		#we pool over our convolved features within a specific region (=pool dim)
		#then we divide our convolved features into seperate m*n regions and take the mean feature activation over these regions 
		#to obtain the pooled features which can be used for classification
		
		#SOURCE : http://deeplearning.stanford.edu/wiki/index.php/Pooling

		numImages = convolved_features.shape[1]
		numFeatures = convolved_features.shape[0]
		convolved_dim = convolved_features.shape[2]
		pool_size = np.abs(convolved_dim / pool_dim)
		
		pooled_features = np.zeros((numFeatures, numImages, pool_size, pool_size), dtype=np.float64)

		#we're pooling over 4 non-overlapping regions
		for i in range(numFeatures):

			for j in range(numImages):
				convolved_image = convolved_features[i, j, :, :] 
				for k in range(pool_size):
					top = k * pool_size
					bottom = top + pool_dim

					for l in range(pool_size):
						left = l * pool_dim
						right = left + pool_dim

						pool_image = convolved_features[i, j, top : bottom, left : right]
						#or max pooling??
						pooled_features[i, j, k, l] = np.mean(pool_image)

		return pooled_features

###########################################################################################
""" Use pooled features for classification """

class SoftmaxClassifier(object):

	def __init__(self, input_size, numClasses, weight_decay):

		self.input_size = input_size
		self.numClasses = numClasses
		self.weight_decay = weight_decay

		self.parameter_vector = 0.005 * np.random.randn(numClasses, input_size)

	def softmax_classify(self, training_images, training_labels):
		print 'Classifying pooled features ....'
		pooled_features_training = np.load('CNN_results/pooled_features_training.npy')

		numImages = training_images.shape[3]
		input_size = pooled_features_training.size / numImages
		pooled_features_training = np.transpose(pooled_features_training, (0, 2, 3, 1))
		pooled_features_training = pooled_features_training.reshape((input_size, numImages))
		training_labels = training_labels.flatten() - 1
			
		# Train softmax classifier
		model = self.train(input_size, self.numClasses, self.weight_decay, pooled_features_training, training_labels)
		(opt_parameter_vector, input_size, num_classes) = model

		print('... Saving model...')
		np.save('CNN_results/opt_parameter_vector_model', opt_parameter_vector)
		print "Done saving"

		return model

	###########################################################################################
	""" Training the Classifier """

	def cost_and_gradients(self, parameter_vector, num_classes, input_size, weight_decay, input_data, labels):
		print 'Calculating Cost ....'
		#SOURCE: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression#Step_2:_Implement_softmaxCost
		numData = input_data.shape[1]

		#Create a Ground Truth Matrix according to training labels
		ground_truth = scipy.sparse.csr_matrix((np.ones(len(labels)), labels, np.arange(len(labels)+1)))
		ground_truth = np.array(np.transpose(ground_truth.todense()))

		#reshape parameter vector to create dot product with data
		parameter_vector = parameter_vector.reshape(num_classes, input_size)
		M = np.dot(parameter_vector,input_data) # M is a matrix including the weighted data 
		M = M - np.max(M) #avoid numerical overflows by subtracting a large constant before exponential
		class_probabilities = np.exp(M) / np.sum(np.exp(M), axis=0)
		cost = (-1 / numData) * np.sum(ground_truth * np.log(class_probabilities)) + (weight_decay / 2) * np.sum(parameter_vector**2)
		gradients = ((-1 / numData) * np.dot((ground_truth - class_probabilities), np.transpose(input_data)) + weight_decay * parameter_vector).flatten()

		return cost, gradients

	def train(self, input_size, num_classes, weight_decay, input_data, labels):
		print 'Training Classifier ....'
		#SOURCE : http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression#Step_4:_Learning_parameters
		#initialise random parameter vector
		parameter_vector = 0.005 * np.random.randn(num_classes, input_size)
		squared_cross_entropy = lambda x: self.cost_and_gradients(x, num_classes, input_size, weight_decay, input_data, labels)
		optimum = scipy.optimize.minimize(squared_cross_entropy, parameter_vector, method='L-BFGS-B', jac=True, options={'maxiter': 400, 'disp': True})
		opt_parameter_vector = optimum.x

		return opt_parameter_vector, input_size, num_classes

			
	###########################################################################################
	""" Test classifier against test images """

	def test_classifier(self, model, test_images, test_labels):
		print 'Testing trained softmax classifier ....'
		pooled_features_test = np.load('CNN_results/pooled_features_test.npy')
		numImages = test_images.shape[3]
		input_size = pooled_features_test.size / numImages

		difficulty = 1
		captchachallenge, right_label, RightImages, right_label_name, opposite_label_name = generate_captcha_challenge(test_images, test_labels, difficulty)
		print 'Which one of these images is a '+str(right_label_name)+'?'
		test_labels = test_labels.flatten()
		start_time = time.time()
		predictions, class_probabilities = predict(model, captchachallenge)
		print("--- %s seconds ---" % (time.time() - start_time))

		print 'class probabilities of', right_label
		print class_probabilities[right_label]
		maximum = np.argmax(class_probabilities[right_label])
		print 'maximum value is ', maximum

		predictions_class = np.argmax(class_probabilities[right_label], axis=1)
		print 'predictions according to the class probabilities', predictions_class

		print 'Our system computed the following predictions ',predictions
		print 'The solution is image number ', RightImages

		if maximum == 0:
			print "Yes! Model passed this challenge!"
		else: 
			print "Model couldn't pass the challenge..."
		
		print 'Testing model on whole dataset...'
		pooled_features_test = np.transpose(pooled_features_test, (0, 2, 3, 1))
		pooled_features_test = pooled_features_test.reshape((input_size, numImages))
		test_labels = test_labels.flatten() - 1
		predictions, class_probabilities = predict(model, pooled_features_test)
		accuracy = 100*np.sum(predictions == test_labels) / test_labels.shape[0]
		correct = np.sum(predictions == test_labels)
		print correct, " out of ", test_labels.shape[0], "images were classified correctly"
		print "Correct predictions : {0:.2f}%".format(accuracy, dtype=np.float64)

def predict(model, input_data):
		print '\n'
		print 'Making predictions ....'	

		opt_parameter_vector, input_size, num_classes = model
		opt_parameter_vector = opt_parameter_vector.reshape(num_classes, input_size)
		# Class Probabilities is a matrix including the predicted label probabilities
		print 'Calculating Class Probabilities ....'	
		
		M = np.dot(opt_parameter_vector,input_data) # M is a matrix including the weighted data 
		M = M - np.max(M) #avoid numerical overflows by subtracting a large constant before exponential
		class_probabilities = np.exp(M) / np.sum(np.exp(M), axis=0)
		#class_probabilities = np.exp(np.dot(opt_parameter_vector, input_data)) / np.sum(np.exp(np.dot(opt_parameter_vector, input_data)), axis=0)
		#predictions(i) = argmax_c * P( y_c | x_i ) 
		predictions = np.argmax(class_probabilities, axis=0)

		return predictions, class_probabilities

def generate_captcha_challenge(test_images, test_labels, difficulty):
		
		print '####################################################################'
		print 'Generating CAPTCHA challenge...'

		pooled_features_test = np.load('CNN_results/pooled_features_test.npy')
		numImages = test_images.shape[3]
		input_size = pooled_features_test.size / numImages
		test_labels = test_labels.flatten()-1

		######### CREATE CAPTCHA CHALLENGES	
		airplanes = []
		cars = []
		cats = []
		dogs = []
		
		for i in range(numImages-1): 
			if test_labels[i] == 0:
				airplanes.append(i)
			elif test_labels[i] == 1:
				cars.append(i)
			elif test_labels[i] == 2:
				cats.append(i)
			elif test_labels[i] == 3:
				dogs.append(i)

		RightImages = []
		for captcha_image in range(1):
			temp = rnd.sample(xrange(0, numImages-1), 1)
			#temp = rnd.sample(cats, 1)
			RightImages.append(temp)
			right_label = test_labels[RightImages]
		
		if right_label == 0:
			opposite_label = 3
			right_label_name = 'Airplane'
			opposite_label_name = 'Dog'
		elif right_label == 1:
			opposite_label = 2
			right_label_name = 'Car'
			opposite_label_name = 'Cat'
		elif right_label == 2:
			opposite_label = 0
			right_label_name = 'Cat'
			opposite_label_name = 'Airplane'
		elif right_label == 3:
			opposite_label = 1
			right_label_name = 'Dog'
			opposite_label_name = 'Car'
		
		WrongImages = []
		for captcha_image in range(difficulty):
			if right_label == 0: 
				temp = rnd.sample(dogs, 1)
				WrongImages.append(temp)
			elif right_label == 1:
				temp = rnd.sample(cats, 1)
				WrongImages.append(temp)
			elif right_label == 2:
				temp = rnd.sample(airplanes, 1)
				WrongImages.append(temp)
			elif right_label == 3:
				temp = rnd.sample(cars, 1)
				WrongImages.append(temp)
		#print WrongImages
		
		print '\n'
		print 'Generating Images ...'
		challenge_num = 1	
		for i in np.nditer(RightImages):
			imager = test_images[:, :, :, i]
			print 'Class label of the right image #'+str(i)+': ', test_labels[i]
			plt.imsave("GUI/CAPTCHA_challenge/image"+str(challenge_num)+".png", imager)
			challenge_num+=1

			for j in WrongImages:
				j = j[0]
				imagew = test_images[:, :, :, j]
				print 'Class label of wrong image #'+str(j)+': ', test_labels[j]
				plt.imsave("GUI/CAPTCHA_challenge/image"+str(challenge_num)+".png", imagew)
				challenge_num+=1	

		numImages = len(RightImages)+len(WrongImages)
		captchachallenge = np.zeros((25, numImages, 3, 3), dtype=np.float64)

		test_right_captchachallenge = np.array([test_images[:, :, :, i] for i in RightImages])
		test_wrong_captchachallenge = np.array([test_images[:, :, :, j] for j in WrongImages])

		test_right_captchachallenge = np.transpose(test_right_captchachallenge, (4, 1, 2, 3, 0))
		test_right_captchachallenge = test_right_captchachallenge.reshape(test_right_captchachallenge.shape[1:])

		test_wrong_captchachallenge = np.transpose(test_wrong_captchachallenge, (4, 1, 2, 3, 0))
		test_wrong_captchachallenge = test_wrong_captchachallenge.reshape(test_wrong_captchachallenge.shape[1:])

		input_units = 192
		hidden_units = 400
		kernel_size = 8
		pool_dim = 19
		weights, bias, zca_white, mean, opt_parameter_vector = load_features(hidden_units, input_units)

		""" Initialise Network """
		convnet = ConvolutionalNeuralNetwork(weights, bias, kernel_size, pool_dim)

		""" Convolve and Pool through Training and Test Images """
		numFeatures = 25 # Convolution Matrix is very large, hence convolve & pool 50 features at a time, decrease if necessary
		iter = numFeatures / hidden_units
		for i in range(iter):
			weights = weights[iter * numFeatures:(iter + 1) * numFeatures, :]
	    	bias = bias[iter * numFeatures:(iter + 1) * numFeatures]

	    	print '\n'
	    	print '1/3... Convolving & Pooling of right images ....'
	    	convolved_features = convnet.convolution(kernel_size, numFeatures, test_right_captchachallenge, weights, bias)
	    	pooled_features = convnet.pooling(pool_dim, convolved_features)
	    	pooled_features_test_right_captchachallenge_features = pooled_features[iter * numFeatures:(iter + 1) * numFeatures, :, :, :]

	    	print '\n'
	    	print '2/3... Convolving & Pooling of wrong images ....'
	    	convolved_features = convnet.convolution(kernel_size, numFeatures, test_wrong_captchachallenge, weights, bias)
	    	pooled_features = convnet.pooling(pool_dim, convolved_features)
	    	pooled_features_test_wrong_captchachallenge_features = pooled_features[iter * numFeatures: (iter + 1) * numFeatures, :, :, :]

	    	print '\n'
	    	print '3/3... Saving pooled features...'
	    	np.save('GUI/pooled_features_test_right_captchachallenge_features.npy', pooled_features_test_right_captchachallenge_features)
	    	np.save('GUI/pooled_features_test_wrong_captchachallenge_features.npy', pooled_features_test_wrong_captchachallenge_features)
	    	print "Done saving"
	    
		######### CONCATENATE ALL CAPTCHA CHALLENGES 
		#right_captchachallenge = np.array([pooled_features_test[:, i, :, :] for i in RightImages])
		#wrong_captchachallenge = np.array([pooled_features_test[:, j, :, :] for j in WrongImages])

		pooled_features_test_right_captchachallenge_features = np.transpose(pooled_features_test_right_captchachallenge_features, (1, 0, 2, 3))
		pooled_features_test_wrong_captchachallenge_features = np.transpose(pooled_features_test_wrong_captchachallenge_features, (1, 0, 2, 3))

		captchachallenge = np.concatenate((pooled_features_test_right_captchachallenge_features, pooled_features_test_wrong_captchachallenge_features))
		captchachallenge = np.reshape(captchachallenge, (25, numImages, 3, 3))
		
		#########RESHAPE 
		captchachallenge = np.transpose(captchachallenge, (0, 2, 3, 1))
		captchachallenge = captchachallenge.reshape((input_size, numImages))

		return captchachallenge, right_label, RightImages, right_label_name, opposite_label_name

if __name__ == '__main__':
	main()