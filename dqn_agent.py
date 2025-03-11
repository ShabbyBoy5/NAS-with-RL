import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from config import (DQN_LEARNING_RATE, DQN_GAMMA, DQN_EPSILON_INITIAL,
                   DQN_EPSILON_DECAY, DQN_EPSILON_MIN, REPLAY_BUFFER_SIZE,
                   BATCH_SIZE, TARGET_UPDATE_FREQUENCY, LAYER_TYPES, MAX_LAYERS)

class ImprovedDQNAgent:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.gamma = DQN_GAMMA
        self.epsilon = DQN_EPSILON_INITIAL
        self.epsilon_decay = DQN_EPSILON_DECAY
        self.epsilon_min = DQN_EPSILON_MIN
        self.learning_rate = DQN_LEARNING_RATE
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.update_target_frequency = TARGET_UPDATE_FREQUENCY
        self.episode_count = 0
        
        # Keep history of past architectures to learn from
        self.architecture_history = []
        
        # Architecture constraints memory
        self.constraint_violations = {
            'first_layer_not_conv': 0,
            'conv_after_flatten': 0,
            'dropout_not_after_pooling': 0,
            'dense_before_flatten': 0
        }

    def _build_network(self):
        """Builds an improved DQN network with state context awareness."""
        model = Sequential()
        
        # Input layer takes current state + partial architecture encoding
        input_size = self.action_space_size + MAX_LAYERS
        
        # Wider layers for better representation
        model.add(Dense(256, input_shape=(input_size,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(128))
        model.add(Activation('relu'))
        
        # Output layer: Q-values for each action (layer type)
        model.add(Dense(self.action_space_size, activation='linear'))
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def encode_partial_architecture(self, architecture_sequence):
        """Creates a fixed-length encoding of the partial architecture."""
        # One-hot encode each layer type, padded to MAX_LAYERS
        encoding = np.zeros(MAX_LAYERS)
        
        for i, layer_type in enumerate(architecture_sequence):
            if i >= MAX_LAYERS:
                break
                
            # Convert layer type to an index based on its position in LAYER_TYPES
            if layer_type in LAYER_TYPES:
                layer_idx = LAYER_TYPES.index(layer_type)
                # Normalize the index to be between 0 and 1
                encoding[i] = (layer_idx + 1) / len(LAYER_TYPES)
        
        return encoding

    def get_enhanced_state(self, state, architecture_sequence):
        """Creates an enhanced state representation that includes partial architecture."""
        # Get fixed-length encoding of the current partial architecture
        arch_encoding = self.encode_partial_architecture(architecture_sequence)
        
        # Concatenate the original state with architecture encoding
        enhanced_state = np.concatenate([state, arch_encoding])
        
        return enhanced_state

    def store_transition(self, state, action, reward, next_state, done, architecture_sequence=None):
        """Stores experience in replay memory with architecture context."""
        if architecture_sequence is not None:
            # Store the full state including architecture context
            enhanced_state = self.get_enhanced_state(state, architecture_sequence)
            enhanced_next_state = self.get_enhanced_state(next_state, architecture_sequence)
            self.memory.append((enhanced_state, action, reward, enhanced_next_state, done))
        else:
            # Fallback to basic state if no architecture provided
            self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, architecture_sequence=None):
        """Chooses action using epsilon-greedy policy with architecture awareness."""
        # Create enhanced state with architecture context if provided
        if architecture_sequence is not None:
            state = self.get_enhanced_state(state, architecture_sequence)
        
        # Reshape state to match the input shape of the network
        input_shape = self.main_network.input_shape[1]
        reshaped_state = np.reshape(state, (1, input_shape))

        if random.random() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_space_size)
        else:
            # Exploitation: best action from Q-network
            q_values = self.main_network.predict(reshaped_state, verbose=0)
            return np.argmax(q_values[0])

    def train_agent(self):
        """Trains the DQN agent using experience replay with prioritized sampling."""
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples yet
            
        # Sample from memory
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        # Get shapes dynamically based on stored experiences
        state_shape = len(minibatch[0][0])
        states = np.zeros((BATCH_SIZE, state_shape))
        next_states = np.zeros((BATCH_SIZE, state_shape))
        
        # Extract states for batch prediction
        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            
        # Batch prediction for efficiency
        q_values_current = self.main_network.predict(states, verbose=0)
        q_values_next_target = self.target_network.predict(next_states, verbose=0)
        q_values_next_main = self.main_network.predict(next_states, verbose=0)
        
        # Prepare training batch
        X_batch = []
        y_batch = []
        
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            target = q_values_current[i].copy()
            
            if done:
                target[action] = reward
            else:
                # Double DQN update rule
                best_action = np.argmax(q_values_next_main[i])
                target[action] = reward + self.gamma * q_values_next_target[i][best_action]
                
            X_batch.append(states[i])
            y_batch.append(target)
            
        # Train in one batch
        self.main_network.fit(
            np.array(X_batch), 
            np.array(y_batch), 
            batch_size=BATCH_SIZE, 
            epochs=1, 
            verbose=0
        )
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def update_target_network(self):
        """Updates target Q-network weights from main Q-network."""
        self.target_network.set_weights(self.main_network.get_weights())

    def record_constraint_violation(self, constraint_type):
        """Records architecture constraint violations to help guide learning."""
        if constraint_type in self.constraint_violations:
            self.constraint_violations[constraint_type] += 1
            
    def get_constraint_statistics(self):
        """Returns the statistics about constraint violations."""
        return self.constraint_violations
        
    def reset_constraint_statistics(self):
        """Resets the constraint violation counters."""
        for key in self.constraint_violations:
            self.constraint_violations[key] = 0
            
    def save_model(self, filename):
        """Saves the main Q-network model."""
        self.main_network.save(filename)

    def load_model(self, filename):
        """Loads the main Q-network model."""
        self.main_network = tf.keras.models.load_model(filename)
        self.target_network = tf.keras.models.load_model(filename)
        self.main_network.compile(optimizer=self.optimizer, loss='mse')
        self.target_network.compile(optimizer=self.optimizer, loss='mse')
        