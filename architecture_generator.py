# architecture_generator.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from search_space import LAYER_TYPES, sample_layer_params, get_layer_instance, is_valid_architecture
from config import INPUT_SHAPE, NUM_CLASSES, MAX_LAYERS
import numpy as np

class ArchitectureGenerator:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.layer_types = LAYER_TYPES
        self.input_shape = INPUT_SHAPE
        self.num_classes = NUM_CLASSES
        self.max_layers = MAX_LAYERS


    def generate_architecture(self, actions):
        """Generates a CNN architecture based on actions from DQN agent."""
        architecture = []
        current_input_shape = self.input_shape

        for action_index in actions:
            if len(architecture) >= self.max_layers: # Limit architecture depth
                break

            layer_type = self.layer_types[action_index % len(self.layer_types)] # Map action index to layer type

            if layer_type == 'Flatten' and any(layer['type'] == 'Flatten' for layer in architecture): # Only one Flatten layer
                continue # Skip if Flatten already exists

            layer_params = sample_layer_params(layer_type) # Sample parameters based on layer type

            layer_config = {'type': layer_type, 'params': layer_params}
            architecture.append(layer_config)

        # Ensure architecture ends with Flatten and Dense output layer if not already present
        if not any(layer['type'] == 'Flatten' for layer in architecture):
            architecture.append({'type': 'Flatten', 'params': {}})
        architecture.append({'type': 'Dense', 'params': {}, 'is_output': True}) # Output Dense layer

        if not is_valid_architecture(architecture): # Validate architecture against constraints
            return None # Return None if invalid

        return architecture


    def build_model_from_architecture(self, architecture):
        """Builds a Keras Sequential model from the architecture list."""
        model = Sequential()
        first_layer = True
        current_shape = self.input_shape

        for layer_config in architecture:
            layer_instance = get_layer_instance(layer_config)
            if first_layer:
                model.add(layer_instance(input_shape=current_shape)) # Set input shape for first layer
                first_layer = False
            else:
                model.add(layer_instance)

            if isinstance(layer_instance, (Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D)):
                # Update shape only for layers that change spatial dimensions (simplified for demonstration)
                # In a real scenario, you'd need to calculate output shape more accurately based on layer parameters.
                if layer_config['params'].get('strides', 1) > 1:
                     if current_shape[0] is not None: # Handle cases where input shape is partially unknown
                         current_shape = (current_shape[0] // layer_config['params'].get('strides', 1) or None,
                                          current_shape[1] // layer_config['params'].get('strides', 1) or None,
                                          layer_config['params'].get('filters', current_shape[2])) # Assume filters are the last dim if conv layer

        model.add(Dense(self.num_classes, activation='softmax')) # Ensure output layer is correctly added

        return model


    def get_parameter_count(self, architecture):
        """Calculates the total number of trainable parameters in an architecture."""
        model = self.build_model_from_architecture(architecture)
        if model:
            return model.count_params()
        return 0