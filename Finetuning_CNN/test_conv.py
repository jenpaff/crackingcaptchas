import unittest
import numpy as np
from convnet_captchas import *

image_dim = 64  # image dimension
image_channels = 3  # number of channels (rgb, so 3)
patch_dim = 8  # patch dimension
num_patches = 50000  # number of patches
convNum = 8 # number of images for convolution
input_units = patch_dim * patch_dim * image_channels  # number of input units
output_units = input_units   # number of output units
hidden_units = 400  # number of hidden units
epsilon = 0.1  # epsilon for ZCA whitening
pool_dim = 19  # dimension of pooling region

class TestData(unittest.TestCase):
    
    def test_load_features(self):
        weights, bias, opt_parameter, zca_white, mean = load_features(hidden_units, input_units)
        self.assertEqual(opt_parameter.shape, (input_units, input_units))
        #self.assertEqual(zca_white.shape, (input_units, input_units))
        #self.assertEqual(mean.shape, (input_units, input_units))
        
    def test_load_training_data(self):
        training_images, training_labels = load_training_data()
        self.assertEqual(training_images.shape, (image_dim, image_dim, image_channels, training_images.shape[3]))
        self.assertEqual(training_labels.shape, (training_images.shape[3], 1))

class TestConvolutionLayer(unittest.TestCase):
    
    def test_convolution_and_pooling(self):
        kernel_size = 8
        weights, bias, opt_parameter, zca_white, mean = load_features(hidden_units, input_units)
        training_images, training_labels = load_training_data()
        test_conv_images = training_images[:, :, :, 0:8] # only use 8 for testing 
        numFeatures = 25
        convnet = ConvolutionalNeuralNetwork(weights, bias, kernel_size, pool_dim)
        convolved_features = convnet.convolution(kernel_size, numFeatures, test_conv_images, weights, bias)
        self.assertEqual(convolved_features.shape, (numFeatures, test_conv_images.shape[3], image_dim-patch_dim+1, image_dim-patch_dim+1))
        pooled_features = convnet.pooling(pool_dim, convolved_features)
        self.assertEqual(pooled_features.shape, (numFeatures, test_conv_images.shape[3], 3,3))#image_dim-patch_dim+1, image_dim-patch_dim+1))

class TestSoftmaxLayer(unittest.TestCase):
        
    def test_cost_function(self):
        #SOURCE: http://napitupulu-jon.appspot.com/posts/Gradient-Checking.html

        #(1) Pick an example z
        weight_decay = 1e-4
        numClasses = 4

        training_images, training_labels = load_training_data()
        pooled_features_training = np.load('CNN_results/pooled_features_training.npy')

        numImages = training_images.shape[3]
        input_size = pooled_features_training.size / numImages
        pooled_features_training = np.transpose(pooled_features_training, (0, 2, 3, 1))
        pooled_features_training = pooled_features_training.reshape((input_size, numImages))
        training_labels = training_labels.flatten() - 1

        #(2) Compute the loss Q(z,w) for the current w
        softmax = SoftmaxClassifier(input_units, numClasses, weight_decay)
        parameter_vector = 0.005 * np.random.randn(numClasses, input_size)
        cost, gradients = softmax.cost_and_gradients(parameter_vector, numClasses, input_size, weight_decay, pooled_features_training, training_labels)
        self.assertFalse(np.isnan(cost))
        squared_cross_entropy = lambda x: softmax.cost_and_gradients(x, numClasses, input_size, weight_decay, pooled_features_training, training_labels)

        #(3) Compute the gradients numerically
        learn_rate = 0.0001
        num_gradient = np.zeros(parameter_vector.shape)
        for i in range(parameter_vector.shape[0]):
            #apply a slight perturbation, e.g. increment and subtract learning rate
            theta_epsilon_plus = np.array(parameter_vector, dtype=np.float64)
            theta_epsilon_plus[i] = parameter_vector[i] + learn_rate
            theta_epsilon_minus = np.array(parameter_vector, dtype=np.float64)
            theta_epsilon_minus[i] = parameter_vector[i] - learn_rate

            num_gradient[i] = (squared_cross_entropy(theta_epsilon_plus)[0] - squared_cross_entropy(theta_epsilon_minus)[0]) / (2 * learn_rate)
            if i % 100 == 0:
                print "Computing gradient for input:", i

        print gradients, num_gradient

        # Compare numerically computed gradients with the ones obtained from backpropagation
        diff = np.linalg.norm(num_gradient - gradients) / np.linalg.norm(num_gradient + gradients)
        diff_msg = diff
        self.assertTrue(diff < (1e-9), msg=diff_msg)
        print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n\n", diff

if __name__ == '__main__':
	unittest.main()