# improved_architecture_generator.py
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from search_space import LAYER_TYPES, sample_layer_params, get_layer_instance, is_valid_architecture
from config import INPUT_SHAPE, NUM_CLASSES, MAX_LAYERS
import numpy as np

class ImprovedArchitectureGenerator:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.layer_types = LAYER_TYPES
        self.input_shape = INPUT_SHAPE
        self.num_classes = NUM_CLASSES
        self.max_layers = MAX_LAYERS
        
        # Keep track of successful architectures and their validation accuracy
        self.successful_architectures = []

    def get_layer_sequence(self, architecture):
        """Extracts the layer type sequence from an architecture."""
        return [layer['type'] for layer in architecture]

    def can_add_layer_type(self, current_architecture, layer_type):
        """Checks if adding the layer type would maintain a valid architecture."""
        # Check constraint 1: First layer must be convolutional
        if len(current_architecture) == 0 and layer_type not in ['Conv2D', 'Conv2DTranspose', 'SeparableConv2D', 'DepthwiseConv2D']:
            return False, "first_layer_not_conv"
        
        # Check constraint 2: No Conv/Pool after Flatten
        if layer_type in ['Conv2D', 'Conv2DTranspose', 'SeparableConv2D', 'DepthwiseConv2D', 
                          'MaxPooling2D', 'AveragePooling2D', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D']:
            if 'Flatten' in current_architecture:
                return False, "conv_after_flatten"
        
        # Check constraint 4: Dropout layer should follow a pooling layer
        if layer_type == 'Dropout' and len(current_architecture) > 0:
            if current_architecture[-1] not in ['MaxPooling2D', 'AveragePooling2D', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D']:
                return False, "dropout_not_after_pooling"
        
        # Check constraint 5: No Dense layer before a Flatten layer
        if layer_type == 'Dense' and 'Flatten' not in current_architecture:
            return False, "dense_before_flatten"
            
        return True, None

    def generate_architecture(self, actions, agent=None):
        """Generates a CNN architecture based on actions from DQN agent.
           If agent is provided, constraint violations will be recorded."""
        architecture = []
        current_architecture_sequence = []
        valid_actions = []

        # Map actions to layer types and generate architecture
        for action_idx, action_index in enumerate(actions):
            if len(architecture) >= self.max_layers:
                break

            # Map action index to layer type
            layer_type = self.layer_types[action_index % len(self.layer_types)]
            
            # Check if adding this layer would be valid
            is_valid, constraint_type = self.can_add_layer_type(current_architecture_sequence, layer_type)
            
            if not is_valid:
                if agent:
                    agent.record_constraint_violation(constraint_type)
                continue
                
            # This action is valid, so keep it
            valid_actions.append(action_index)
                
            # Generate layer parameters
            layer_params = sample_layer_params(layer_type)
            
            # Add layer to architecture
            layer_config = {'type': layer_type, 'params': layer_params}
            architecture.append(layer_config)
            
            # Update our current architecture sequence
            current_architecture_sequence.append(layer_type)

        # Add essential layers if missing
        self._complete_architecture(architecture, current_architecture_sequence)

        # Final validation check
        if not is_valid_architecture(architecture):
            return None
            
        return architecture

    def _complete_architecture(self, architecture, current_architecture_sequence):
        """Completes the architecture by adding necessary layers."""
        # Ensure there is a Flatten layer before Dense layers if not already present
        if 'Dense' in current_architecture_sequence and 'Flatten' not in current_architecture_sequence:
            # Find position to insert Flatten (before first Dense)
            dense_index = current_architecture_sequence.index('Dense')
            architecture.insert(dense_index, {'type': 'Flatten', 'params': {}})
            current_architecture_sequence.insert(dense_index, 'Flatten')
            
        # Ensure there is a Flatten layer if there are convolutional layers
        conv_layers = ['Conv2D', 'Conv2DTranspose', 'SeparableConv2D', 'DepthwiseConv2D']
        has_conv = any(layer in conv_layers for layer in current_architecture_sequence)
        
        if has_conv and 'Flatten' not in current_architecture_sequence:
            architecture.append({'type': 'Flatten', 'params': {}})
            current_architecture_sequence.append('Flatten')
            
        # Add output layer if not already present
        if not any(layer.get('is_output', False) for layer in architecture):
            # Make sure we have a Flatten layer before adding the output
            if 'Flatten' not in current_architecture_sequence and has_conv:
                architecture.append({'type': 'Flatten', 'params': {}})
                current_architecture_sequence.append('Flatten')
                
            architecture.append({
                'type': 'Dense', 
                'params': {'units': self.num_classes, 'activation': 'softmax'}, 
                'is_output': True
            })
            current_architecture_sequence.append('Dense')

    def build_model_from_architecture(self, architecture):
        """Builds a Keras Sequential model from the architecture list."""
        if not architecture:
            print("Error: Empty architecture provided")
            return None
            
        model = Sequential()
        
        # Add input layer
        model.add(InputLayer(input_shape=self.input_shape))
        
        # Add all layers from the architecture
        for i, layer_config in enumerate(architecture):
            try:
                layer_instance = get_layer_instance(layer_config)
                
                # Special handling for output layer
                if layer_config.get('is_output', False):
                    # Use NUM_CLASSES for output layer units
                    if layer_config['type'] == 'Dense':
                        layer_instance = Dense(self.num_classes, activation='softmax')
                
                model.add(layer_instance)
                
            except Exception as e:
                print(f"Error adding layer {i} ({layer_config['type']}): {e}")
                return None
                
        return model

    def get_parameter_count(self, architecture):
        """Calculates the total number of trainable parameters in an architecture."""
        model = self.build_model_from_architecture(architecture)
        if model:
            return model.count_params()
        return 0
        
    def store_successful_architecture(self, architecture, accuracy):
        """Stores successful architectures for future reference."""
        self.successful_architectures.append((architecture, accuracy))
        
        # Sort by accuracy (descending)
        self.successful_architectures.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top 20 architectures
        if len(self.successful_architectures) > 20:
            self.successful_architectures = self.successful_architectures[:20]