# -*- coding: utf-8 -*-
"""
Created on Sat Aug 01 16:48:14 2015

@author: acq14jp
"""

import unittest
import numpy as np
import numpy.testing as npt
from extractFeatures import *

""" Define parameters """
image_channels  = 3 # 3 channels for RGB images
patchDim  = 8 #patch dimension
numPatches = 100000 # number of patches
input_units = patchDim * patchDim * image_channels #192
hidden_units = 400#5 #side length of representative image patches
sparsity_rate = 0.035 # desired average activation of hidden units
weight_decay = 0.003 # weight decay parameter
sparsity_penalty = 5 # weight of sparsity penalty term
reg_zca = 0.1

class TestData(unittest.TestCase):
    
    def test_processing_patches(self):
        patches_white, mean, zca_white = data_processing(numPatches, reg_zca)
        self.assertEqual(patches_white.shape, (input_units, numPatches))

class TestAutoencoder(unittest.TestCase):
    
    def test_parameters(self):
        autoencoder = SparseLinearAutoencoder(input_units, hidden_units, sparsity_rate, weight_decay, sparsity_penalty)
        interval = np.sqrt(6) / np.sqrt(hidden_units + input_units + 1)
        weight_input = autoencoder.parameter_vector[0:hidden_units * input_units].reshape(hidden_units, input_units)
        weight_output = autoencoder.parameter_vector[hidden_units * input_units:2 * hidden_units * input_units].reshape(input_units, hidden_units)
        bias_input = autoencoder.parameter_vector[2 * hidden_units * input_units:2 * hidden_units * input_units + hidden_units].reshape(hidden_units, 1)
        bias_output = autoencoder.parameter_vector[2 * hidden_units * input_units + hidden_units:].reshape(input_units, 1)
        
        #test whether weights have the correct shape and aree in the correct range
        self.assertEqual(weight_input.shape, (hidden_units, input_units))
        self.assertEqual(weight_output.shape, (input_units, hidden_units))
        
        self.assertTrue(-interval < (weight_input.all())) #> interval)
        self.assertTrue(-interval < (weight_output.all())) #> interval)
        
        #test whether bias have the correct shape and are initialised to zero
        self.assertEqual(bias_input.shape, (hidden_units, 1))
        self.assertEqual(bias_output.shape, (input_units, 1))
        
        self.assertEqual(bias_input.all(), 0)
        self.assertEqual(bias_output.all(), 0)
        
        #test parameter_vector
        self.assertEqual(autoencoder.parameter_vector.shape, ((weight_input.shape[0]*weight_input.shape[1]+weight_output.shape[0]*weight_output.shape[1]+bias_input.shape[0]+bias_output.shape[0]), ))
        
    def test_feedforward(self):
        autoencoder = SparseLinearAutoencoder(input_units, hidden_units, sparsity_rate, weight_decay, sparsity_penalty)
        patches_white, mean, zca_white = data_processing(numPatches, reg_zca)
        input_activations, hidden_layer, hidden_activations, output_layer, output_activations = autoencoder.feedforward(autoencoder.parameter_vector, patches_white)
        #test layers and activations
        self.assertEqual(input_activations.shape, patches_white.shape)
        self.assertEqual(hidden_layer.shape, (hidden_units, numPatches))
        self.assertEqual(hidden_activations.shape, (hidden_units, numPatches))
        # values in the hidden layer should be greater than 0 and smaller than 1 > sigmoid function
        self.assertFalse(0 > hidden_activations.all())
        self.assertFalse(1 < hidden_activations.all())
        self.assertEqual(output_layer.shape, (input_units, numPatches))
        self.assertEqual(output_activations.shape, output_layer.shape) # we apply a linear function, therefore output can be smaller and greater than 1 
        
        #test rho = sparsity parameter, should be a value close to 0
        rho = np.tile(sparsity_rate, hidden_units)
        self.assertTrue(0.5 < rho.all())
        self.assertEqual(rho.shape, (hidden_units, ))
        
        #test rho_hat = average activation of hidden units, should be 
        rho_hat = np.sum(hidden_activations, axis=1) / numPatches
        self.assertFalse(0.5 > rho_hat.all())
        self.assertEqual(rho_hat.shape, (hidden_units, ))
        
        #test error terms (deltas)
        error_term_output, error_term_hidden = autoencoder.error_terms(autoencoder.parameter_vector, numPatches, patches_white)
        self.assertEqual(error_term_output.shape, (input_units, numPatches))
        self.assertEqual(error_term_hidden.shape, (hidden_units, numPatches))
    
    def test_cost_function(self):
        #SOURCE : http://napitupulu-jon.appspot.com/posts/Gradient-Checking.html
        #(1) Pick an example z
        hidden_units = 5
        input_units = 8
        patches_white = np.random.rand(8, 10)

        #(2) Compute the loss Q(z,w) for the current w
        autoencoder = SparseLinearAutoencoder(input_units, hidden_units, sparsity_rate, weight_decay, sparsity_penalty)
        #(2) Compute the loss Q(z,w) for the current w
        cost, backprop_gradient = autoencoder.cost_function(autoencoder.parameter_vector, input_units, hidden_units, sparsity_penalty, sparsity_rate, weight_decay, patches_white)
        self.assertFalse(np.isnan(cost))
        J = lambda x: autoencoder.cost_function(x, input_units, hidden_units, sparsity_penalty, sparsity_rate, weight_decay, patches_white)

        #(3) Compute the gradients numerically
        learn_rate = 0.0001
        num_gradient = np.zeros(autoencoder.parameter_vector.shape)
        for i in range(autoencoder.parameter_vector.shape[0]):
            #apply a slight perturbation, e.g. increment and subtract learning rate
            theta_epsilon_plus = np.array(autoencoder.parameter_vector, dtype=np.float64)
            theta_epsilon_plus[i] = autoencoder.parameter_vector[i] + learn_rate
            theta_epsilon_minus = np.array(autoencoder.parameter_vector, dtype=np.float64)
            theta_epsilon_minus[i] = autoencoder.parameter_vector[i] - learn_rate

            num_gradient[i] = (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * learn_rate)
            if i % 100 == 0:
                print "Computing gradient for input:", i

        print backprop_gradient, num_gradient

        # Compare numerically computed gradients with the ones obtained from backpropagation
        diff = np.linalg.norm(num_gradient - backprop_gradient) / np.linalg.norm(num_gradient + backprop_gradient)
        diff_msg = diff
        self.assertTrue(diff < (1e-9), msg=diff_msg)
        print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n\n", diff

if __name__ == '__main__':
	unittest.main()