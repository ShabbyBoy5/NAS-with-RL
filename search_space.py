import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D,
                                     GlobalMaxPooling2D, GlobalAveragePooling2D,
                                     DepthwiseConv2D, SeparableConv2D, Conv2DTranspose,
                                     Dropout, BatchNormalization, Flatten, Dense,
                                     ReLU, ELU, LeakyReLU, PReLU, ThresholdedReLU,
                                     Activation,) # Added Sequential import
from tensorflow.keras.initializers import (HeNormal, HeUniform, RandomNormal,
                                            RandomUniform)
from tensorflow.keras.regularizers import l1, l2, l1_l2

from config import INPUT_SHAPE, NUM_CLASSES, LAYER_TYPES # Import LAYER_TYPES from config.py


SEARCH_SPACE = {
    'Conv2D': {
        'filters': [16, 32, 64, 96, 128, 160, 192, 224, 256],
        'kernel_size': [3, 5, 7, 9, 11],
        'strides': [1, 2], # Removed 3 for simplicity, kept 1,2
        'padding': ['same', 'valid'],
        'kernel_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'bias_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'kernel_regularizer': [l1, l2, l1_l2, None], # Added None for no regularization
        'activation': ['relu', 'elu', 'selu', 'swish', 'leaky_relu'] # Added leaky_relu
    },
    'Conv2DTranspose': {
        'filters': [16, 32, 64, 96, 128, 160, 192, 224, 256],
        'kernel_size': [3, 5, 7, 9, 11],
        'strides': [1, 2], # Removed 3 for simplicity, kept 1,2
        'padding': ['same', 'valid'],
        'kernel_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'bias_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'kernel_regularizer': [l1, l2, l1_l2, None], # Added None for no regularization
        'activation': ['relu', 'elu', 'selu', 'swish', 'leaky_relu'] # Added leaky_relu
    },
    'SeparableConv2D': {
        'filters': [16, 32, 64, 96, 128, 160, 192, 224, 256],
        'kernel_size': [3, 5, 7, 9, 11],
        'strides': [1, 2], # Removed 3 for simplicity, kept 1,2
        'padding': ['same', 'valid'],
        'kernel_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'bias_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'kernel_regularizer': [l1, l2, l1_l2, None], # Added None for no regularization
        'activation': ['relu', 'elu', 'selu', 'swish', 'leaky_relu'] # Added leaky_relu
    },
    'DepthwiseConv2D': {
        'kernel_size': [3, 5, 7, 9, 11],
        'strides': [1, 2], # Removed 3 for simplicity, kept 1,2
        'padding': ['same', 'valid'],
        'kernel_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'bias_initializer': [HeNormal, HeUniform, RandomNormal, RandomUniform],
        'kernel_regularizer': [l1, l2, l1_l2, None], # Added None for no regularization
        'activation': ['relu', 'elu', 'selu', 'swish', 'leaky_relu'] # Added leaky_relu
    },
    'MaxPooling2D': {
        'pool_size': [2, 3, 4, 5],
        'strides': [1, 2], # Removed 3,4,5 for simplicity, kept 1,2
        'padding': ['same', 'valid']
    },
    'AveragePooling2D': {
        'pool_size': [2, 3, 4, 5],
        'strides': [1, 2], # Removed 3,4,5 for simplicity, kept 1,2
        'padding': ['same', 'valid']
    },
    'GlobalMaxPooling2D': {},
    'GlobalAveragePooling2D': {},
    'Dropout': {
        'rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    'BatchNormalization': {},
    'Flatten': {},
    'Dense': {
        'units': [8, 16, 32, 64, 128, 256, 512],
        'activation': ['sigmoid', 'tanh', 'relu', 'elu', 'selu', 'swish', 'leaky_relu'] # Added leaky_relu
    }
}

POOLING_LAYERS = ['MaxPooling2D', 'AveragePooling2D', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D']
CONVOLUTIONAL_LAYERS = ['Conv2D', 'Conv2DTranspose', 'SeparableConv2D', 'DepthwiseConv2D']
REGULARIZATION_LAYERS = ['Dropout', 'BatchNormalization']


def sample_layer_params(layer_type):
    """Samples parameters for a given layer type from the search space."""
    params = {}
    layer_search_space = SEARCH_SPACE.get(layer_type)
    if layer_search_space:
        for param_name, param_values in layer_search_space.items():
            if param_values: # Only sample if there are values to choose from
                params[param_name] = param_values[tf.random.uniform(shape=[], minval=0, maxval=len(param_values), dtype=tf.int32)]
    return params


def is_valid_architecture(architecture):
    """Checks if the generated architecture is valid based on constraints (Table III)."""
    constraints_violated = False
    for i, layer_config in enumerate(architecture):
        layer_type = layer_config['type']

        # Constraint 1: First layer must be Convolutional
        if i == 0 and layer_type not in CONVOLUTIONAL_LAYERS:
            constraints_violated = True
            break

        # Constraint 2: No Conv/Pool after Flatten
        if layer_type in CONVOLUTIONAL_LAYERS + POOLING_LAYERS:
            for previous_layer_config in architecture[:i]:
                if previous_layer_config['type'] == 'Flatten':
                    constraints_violated = True
                    break
            if constraints_violated:
                break

        # Constraint 3: Pooling layer after Flatten Layer - Not applicable in sequential CNN

        # Constraint 4: Dropout after Pooling Layer
        if layer_type == 'Dropout':
            if i > 0 and architecture[i-1]['type'] not in POOLING_LAYERS:
                constraints_violated = True
                break

        # Constraint 5: No Dense Layer can be there before a Flatten Layer
        if layer_type == 'Dense':
            flatten_found = False
            for previous_layer_config in architecture[:i]:
                if previous_layer_config['type'] == 'Flatten':
                    flatten_found = True
                    break
            if not flatten_found:
                constraints_violated = True
                break

    return not constraints_violated


def get_layer_instance(layer_config):
    """Creates a Keras layer instance based on the layer configuration."""
    layer_type = layer_config['type']
    params = layer_config['params']

    if layer_type == 'Conv2D':
        activation_name = params.get('activation')
        activation_layer = None
        if activation_name == 'leaky_relu':
            activation_layer = LeakyReLU() # Use LeakyReLU layer instance
            params.pop('activation', None) # Remove 'activation' from params for Conv2D layer itself
        elif activation_name: # For other activation names, they can be passed as string to Conv2D
            params['activation'] = activation_name # Keep activation in params for Conv2D

        conv2d_layer = Conv2D(**params)
        if activation_layer:
            return Sequential([conv2d_layer, activation_layer]) # Wrap Conv2D and Activation in Sequential
        return conv2d_layer

    elif layer_type == 'Conv2DTranspose':
        activation_name = params.get('activation')
        activation_layer = None
        if activation_name == 'leaky_relu':
            activation_layer = LeakyReLU() # Use LeakyReLU layer instance
            params.pop('activation', None) # Remove 'activation' from params for Conv2DTranspose layer itself
        elif activation_name: # For other activation names, they can be passed as string to Conv2DTranspose
            params['activation'] = activation_name # Keep activation in params for Conv2DTranspose

        conv2dtranspose_layer = Conv2DTranspose(**params)
        if activation_layer:
            return Sequential([conv2dtranspose_layer, activation_layer]) # Wrap Conv2DTranspose and Activation in Sequential
        return conv2dtranspose_layer

    elif layer_type == 'SeparableConv2D':
        activation_name = params.get('activation')
        activation_layer = None
        if activation_name == 'leaky_relu':
            activation_layer = LeakyReLU() # Use LeakyReLU layer instance
            params.pop('activation', None) # Remove 'activation' from params for SeparableConv2D layer itself
        elif activation_name: # For other activation names, they can be passed as string to SeparableConv2D
            params['activation'] = activation_name # Keep activation in params for SeparableConv2D

        separableconv2d_layer = SeparableConv2D(**params)
        if activation_layer:
            return Sequential([separableconv2d_layer, activation_layer]) # Wrap SeparableConv2D and Activation in Sequential
        return separableconv2d_layer

    elif layer_type == 'DepthwiseConv2D':
        activation_name = params.get('activation')
        activation_layer = None
        if activation_name == 'leaky_relu':
            activation_layer = LeakyReLU() # Use LeakyReLU layer instance
            params.pop('activation', None) # Remove 'activation' from params for DepthwiseConv2D layer itself
        elif activation_name: # For other activation names, they can be passed as string to DepthwiseConv2D
            params['activation'] = activation_name # Keep activation in params for DepthwiseConv2D

        depthwiseconv2d_layer = DepthwiseConv2D(**params)
        if activation_layer:
            return Sequential([depthwiseconv2d_layer, activation_layer]) # Wrap DepthwiseConv2D and Activation in Sequential
        return depthwiseconv2d_layer


    elif layer_type == 'MaxPooling2D':
        return MaxPooling2D(**params)
    elif layer_type == 'AveragePooling2D':
        return AveragePooling2D(**params)
    elif layer_type == 'GlobalMaxPooling2D':
        return GlobalMaxPooling2D()
    elif layer_type == 'GlobalAveragePooling2D':
        return GlobalAveragePooling2D()
    elif layer_type == 'Dropout':
        return Dropout(**params)
    elif layer_type == 'BatchNormalization':
        return BatchNormalization()
    elif layer_type == 'Flatten':
        return Flatten()
    elif layer_type == 'Dense':
        activation_name = params.get('activation')
        activation_layer = None
        if activation_name == 'leaky_relu':
            activation_layer = LeakyReLU() # Use LeakyReLU layer instance
            params.pop('activation', None) # Remove 'activation' from params for Dense layer itself
        elif activation_name: # For other activation names, they can be passed as string to Dense
            params['activation'] = activation_name # Keep activation in params for Dense

        dense_layer = Dense(**params, units=params.get('units', NUM_CLASSES) if layer_config.get('is_output', False) else params.get('units', 128))
        if activation_layer:
            return Sequential([dense_layer, activation_layer]) # Wrap Dense and Activation in Sequential
        return dense_layer

    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
