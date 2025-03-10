import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from config import (DQN_LEARNING_RATE, DQN_GAMMA, DQN_EPSILON_INITIAL,
                    DQN_EPSILON_DECAY, DQN_EPSILON_MIN, REPLAY_BUFFER_SIZE,
                    BATCH_SIZE, TARGET_UPDATE_FREQUENCY, LAYER_TYPES)

ACTION_SPACE_SIZE_CONSTANT = len(LAYER_TYPES)

class DQNAgent:
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

    def _build_network(self):
        """Builds the DQN (Controller) network using a Dense network instead of LSTM to avoid issues."""
        model = Sequential()
        # Replace LSTM with Dense layers to avoid the tensor conversion issue
        model.add(Dense(128, activation='relu', input_shape=(self.action_space_size,)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(optimizer='Adam', loss='mse') # Compile here
        return model

    def store_transition(self, state, action, reward, next_state, done):
        """Stores experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Chooses action using epsilon-greedy policy."""
        # Reshape state to match the input shape of the network
        reshaped_state = np.reshape(state, (1, self.action_space_size))

        if random.random() <= self.epsilon:
            return random.randrange(self.action_space_size)  # Explore: random action
        else:
            q_values = self.main_network.predict(reshaped_state)  # Exploit: best action from Q-network
            return np.argmax(q_values[0])

    def train_agent(self):
        """Trains the DQN agent using experience replay."""
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples in memory yet

        minibatch = random.sample(self.memory, BATCH_SIZE)

        # Reshape states for the network input
        states = np.array([np.reshape(transition[0], (self.action_space_size,)) for transition in minibatch])
        next_states = np.array([np.reshape(transition[3], (self.action_space_size,)) for transition in minibatch])

        # Predict Q-values for current and next states
        q_values_current = self.main_network.predict_on_batch(states)
        q_values_next_target = self.target_network.predict_on_batch(next_states)
        q_values_next_main = self.main_network.predict_on_batch(next_states)  # For Double DQN

        input_states = []
        targets = []

        for index, (state, action, reward, _, done) in enumerate(minibatch):
            target_q_value = q_values_current[index]  # Initialize target with current Q-values

            if done:
                target_q_value[action] = reward  # Terminal state, target is just reward
            else:
                # Double DQN: Use main network to choose action, target network to evaluate
                best_action_next = np.argmax(q_values_next_main[index])
                target_q_value[action] = reward + self.gamma * q_values_next_target[index][best_action_next]  # Discounted future reward

            input_states.append(np.reshape(state, (self.action_space_size,)))  # Reshape state
            targets.append(target_q_value)

        # Train the network (removed explicit fit - compilation is in _build_network now)
        with tf.GradientTape() as tape:
            predicted_q_values = self.main_network(np.array(input_states)) # Predict Q-values for batch states
            loss = tf.keras.losses.MeanSquaredError()(np.array(targets), predicted_q_values) # Calculate MSE loss

        gradients = tape.gradient(loss, self.main_network.trainable_variables) # Get gradients
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables)) # Apply gradients


        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay  # Decay epsilon for exploration

    def update_target_network(self):
        """Updates target Q-network weights from main Q-network."""
        self.target_network.set_weights(self.main_network.get_weights())
        self.target_network.compile(optimizer=self.optimizer, loss='mse') # Ensure target network is also compiled

    def decay_epsilon(self):
        """Decays epsilon value."""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def save_model(self, filename):
        """Saves the main Q-network model."""
        self.main_network.save(filename)

    def load_model(self, filename):
        """Loads the main Q-network model."""
        self.main_network = tf.keras.models.load_model(filename)
        self.target_network = tf.keras.models.load_model(filename)  # Load target network as well, keep them synced
        self.main_network.compile(optimizer=self.optimizer, loss='mse')  # Compile loaded models
        self.target_network.compile(optimizer=self.optimizer, loss='mse') # Compile loaded models
