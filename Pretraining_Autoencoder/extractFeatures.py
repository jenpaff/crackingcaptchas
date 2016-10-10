# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:11:55 2015

@author: jenpaff
"""
import numpy as np
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt

def main():
    ################## Parameters ##############################

    image_channels = 3  # number of channels (rgb, so 3)
    patch_dim = 8  # patch dimension
    input_units = patch_dim * patch_dim * image_channels  # number of input units
    output_units = input_units   # number of output units
    hidden_units = 400  # number of hidden units
    sparsity_rate = 0.035 # desired average activation of hidden units
    weight_decay = 0.003 # weight decay parameter
    sparsity_penalty = 5 # weight of sparsity penalty term
    reg_zca = 0.1
    numPatches = 100000

    """ Load & Preprocess Patches """
    patches_white, mean, zca_white = data_processing(numPatches, reg_zca)
    numPatches = patches_white.shape[1] 

    ###########################################################

    """ Initialise Autoencoder """
    autoencoder = SparseLinearAutoencoder(input_units, hidden_units, sparsity_rate, weight_decay, sparsity_penalty)

    """ Cost Function, Gradients & Optimisation """
    squared_mean_error = lambda x: autoencoder.cost_function(x, input_units, hidden_units, sparsity_penalty, sparsity_rate, weight_decay, patches_white)
    optimal_parameters = scipy.optimize.minimize(squared_mean_error, autoencoder.parameter_vector, method='L-BFGS-B', jac=True, options={'maxiter': 3, 'disp': True})
    opt_parameter_vector = optimal_parameters.x

    """ Save features as .npy Data """
    print('Saving features...')
    #Save learnt features and preprocessing matrices
    np.save('ConvolutionalNeuralNetwork/Autoencoder_results/opt_parameter_vector.npy', opt_parameter_vector)
    np.save('ConvolutionalNeuralNetwork/Autoencoder_results/zca_white.npy', zca_white)
    np.save('ConvolutionalNeuralNetwork/Autoencoder_results/mean.npy', mean)
    print('Saved.')

    """ Visualise Results """
    #Visualise the weights of the optimum parameter
    weights = opt_parameter_vector[0:hidden_units * input_units].reshape(hidden_units, input_units)
    #b = opt_parameter_vector[2 * hidden_units * input_units:2 * hidden_units * input_units + hidden_units]
    
###########################################################################################
""" Data Processing """

def data_processing(numPatches, reg_zca):
    print 'Preprocessing Data ....'
    # Load Data

    patches = scipy.io.loadmat('samples.mat')
    patches = np.array(patches['patches'])

    # Center Data
    mean = np.mean(patches, axis=1, keepdims=True)
    patches_cent = patches - mean

    # Whiten Data: learning features by Krivskhy et al & lab week 8
    # PAPER: http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    conv_matrix = np.dot(patches_cent, np.transpose(patches_cent)) / numPatches
    #use np.linalg.svd instead of np.linalg.eigh, because more numerical stable!
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(conv_matrix)
    decorrelation_matrix = np.diag(1. / np.sqrt(eigenvalues + reg_zca))
    zca_white = np.dot(np.dot(eigenvectors, decorrelation_matrix), eigenvectors_T)
    patches_white = np.dot(zca_white, patches_cent)

    return patches_white, mean, zca_white

###########################################################################################

class SparseLinearAutoencoder(object):

    def __init__(self, input_units, hidden_units, sparsity_rate, weight_decay, sparsity_penalty):

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.sparsity_rate = sparsity_rate
        self.weight_decay = weight_decay
        self.sparsity_penalty = sparsity_penalty

        """ Initialise weights  """
        interval = np.sqrt(6) / np.sqrt(hidden_units + input_units + 1)
        weight_input = np.asarray(np.random.uniform(low = -interval, high = interval, size = (hidden_units, input_units)))
        weight_output = np.asarray(np.random.uniform(low = -interval, high = interval, size = (input_units, hidden_units)))

        """ Initialise bias to zero """
        bias_input = np.zeros((hidden_units, 1))
        bias_output = np.zeros((input_units, 1))

        self.parameter_vector = np.concatenate((weight_input.flatten(), 
                                        weight_output.flatten(),
                                        bias_input.flatten(),
                                        bias_output.flatten()))

    """ Logistic Function """

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    ###########################################################################################

        """ Backpropagation Algorithm
        (1) Perform forward pass to compute activations for layers L2, L3, ...
        (2) For each output unit i in output layer n(l) set δ(nl)i = ∂/∂z_i(nl) (1/2 * |y - hypo_W,b (x)|^2) = - (y(i - a_i(nl) . f'(z_i(nl))))
        (3) For l = n(l)-1, n(l)-2, n(l)-3, ..., 2:
            For each node i in layer l set δ(l) = ( SUM ( W_ji(l) δ_j (l+1 ))) . f'(z(l))
        (4) Compute partial derivatives:
            ∂/∂W_ij(l) J(W,b;x,y) = a_j(l) δ_i (l+1)
            ∂/∂b_i(l) J(W,b;x,y) = δ_i (l+1)    
        """

    def feedforward(self, parameter_vector, patches):
        print 'Performing Feedforward Pass....'

        """ Unroll values from parameter vector """
        weight_input = parameter_vector[0:self.hidden_units * self.input_units].reshape(self.hidden_units, self.input_units)
        weight_output = parameter_vector[self.hidden_units * self.input_units:2 * self.hidden_units * self.input_units].reshape(self.input_units, self.hidden_units)
        bias_input = parameter_vector[2 * self.hidden_units * self.input_units:2 * self.hidden_units * self.input_units + self.hidden_units].reshape(self.hidden_units, 1)
        bias_output = parameter_vector[2 * self.hidden_units * self.input_units + self.hidden_units:].reshape(self.input_units, 1)

        """ Compute Feedforward Pass """
        input_layer = patches # input layer is equal to patches
        numPatches = patches.shape[1]
        numInput = 1
        numHidden = 1
        numOutput = 1
        hidden_layer = np.zeros((self.hidden_units, numPatches))
        output_layer = np.zeros((self.hidden_units, numPatches))
        input_activations = np.zeros((self.input_units, numPatches))
        hidden_activations = np.zeros((self.hidden_units, numPatches))
        output_activations = np.zeros((self.input_units, numPatches))

        #input activations
        for i in range(numInput):
            input_activations = input_layer # activations (output values) of input layer is equal to input layer

        #hidden activations
        for j in range(numHidden):
            for i in range(numInput):
                hidden_layer = np.dot(weight_input, input_activations) + bias_input #hidden layer is the weighted sum of the inputs
            hidden_activations = self.sigmoid(hidden_layer) #activations (output values) of hidden layer 

        #output activations
        for k in range(numOutput):
            for j in range(numOutput):
                output_layer = np.dot(weight_output, hidden_activations) + bias_output #output layer is the weighted sum of the activations from the hidden layer
            output_activations =  output_layer # note that the linear decoder outputs a linear function in the output layer

        return input_activations, hidden_layer, hidden_activations, output_layer, output_activations

    def error_terms(self, parameter_vector, numPatches, patches):
        print 'Computing error terms....'
        
        """
        Compute deltas (error terms) of all output and hidden neurons
        error term of output neuron: difference between the network's activation and the true target value
        error term of hidden neuron: weigthed average of the error terms that use activations of the input layer as input
        """
        weight_output = parameter_vector[self.hidden_units * self.input_units:2 * self.hidden_units * self.input_units].reshape(self.input_units, self.hidden_units)  
        input_activations, hidden_layer, hidden_activations, output_layer, output_activations = self.feedforward(parameter_vector, patches)
        #Add sparsity penalty term (KL divergence) to ensure hidden units' activations are close to 0
        rho_hat = np.sum(hidden_activations, axis=1) / numPatches #rho_hat = averaged sum of activation of hidden units
        rho = np.tile(self.sparsity_rate, self.hidden_units) #rho = sparsity parameter (usually close to 0)
        KL_div_delta = np.transpose(np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), ((numPatches, 1))))
        
        #Compute deltas
        error_term_output = -(patches - output_activations)
        error_term_hidden = (np.dot(np.transpose(weight_output), error_term_output) + self.sparsity_penalty * KL_div_delta) * self.sigmoid_prime(hidden_layer)

        return error_term_output, error_term_hidden

    def cost_function(self, parameter_vector, input_units, hidden_units, sparsity_penalty, sparsity_rate, weight_decay, patches):
        print 'Computing cost function....'

        """ Unroll values from parameter vector """
        numPatches = patches.shape[1]
        weight_input = parameter_vector[0:hidden_units * input_units].reshape(hidden_units, input_units)
        weight_output = parameter_vector[hidden_units * input_units:2 * hidden_units * input_units].reshape(input_units, hidden_units)
        input_activations, hidden_layer, hidden_activations, output_layer, output_activations = self.feedforward(self.parameter_vector, patches)
        error_term_output, error_term_hidden = self.error_terms(parameter_vector, numPatches, patches)
        
        #Add sparsity penalty term (KL divergence) to ensure hidden units' activations are close to 0
        rho_hat = np.sum(hidden_activations, axis=1) / numPatches
        rho = np.tile(sparsity_rate, hidden_units)
        KL_divergence = rho * np.log(rho / rho_hat) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
        
        #Compute cost function
        print 'Computing Cost Function....'
        cost = np.sum((output_activations - patches) ** 2) / (2 * numPatches) + \
               (weight_decay / 2) * (np.sum(weight_input ** 2) + np.sum(weight_output ** 2)) + \
               sparsity_penalty * np.sum(KL_divergence)

        print 'Computing Partial derivatives....'
        #Compute partial derivatives of the weights
        weight_input_grad = np.dot( error_term_hidden, ( np.transpose(patches))) / numPatches + weight_decay * weight_input
        weight_output_grad = np.dot( error_term_output, ( np.transpose(hidden_activations))) / numPatches + weight_decay * weight_output
        #Compute partial derivatives of the bias    
        bias_input_grad = np.sum( error_term_hidden, axis=1) / numPatches
        bias_output_grad = np.sum( error_term_output, axis=1) / numPatches

        gradients = np.concatenate(
            (weight_input_grad.reshape(hidden_units * input_units),
            weight_output_grad.reshape(hidden_units * input_units),
            bias_input_grad.reshape(hidden_units, ),
            bias_output_grad.reshape(input_units, )
            ))

        return cost, gradients

if __name__ == '__main__':
	main()