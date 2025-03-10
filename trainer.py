# trainer.py
import tensorflow as tf
import numpy as np
import os
import json
from dqn_agent import DQNAgent
from architecture_generator import ArchitectureGenerator
from utils import load_dataset, train_and_evaluate_architecture
from one_shot_training import apply_one_shot_training # Import even if simplified
from config import (MAX_LAYERS, LAYER_TYPES, CONTROLLER_EPOCHS, M_EPOCHS_PER_CNN,
                    MODEL_SAVE_PATH, RESULTS_SAVE_PATH, REWARD_ACCURACY_WEIGHT,
                    REWARD_PARAMETER_PENALTY_WEIGHT, N_CNN_MODELS_PER_EPISODE,
                    OPTIMIZERS_CNN, LEARNING_RATES_CNN)

# Ensure save directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

# Initialize DQN Agent and Architecture Generator
action_space_size = len(LAYER_TYPES) # Action space is the number of layer types
dqn_agent = DQNAgent(action_space_size)
architecture_generator = ArchitectureGenerator(action_space_size)

# Load Dataset (for input shape and num_classes in config)
load_dataset() # Call once to initialize dataset info in config if needed

# Experience Replay Buffer (for CNN architectures and their performance)
trained_architectures_buffer = [] # List to store (architecture, accuracy, params) tuples

# --- Main Training Loop ---
for episode in range(CONTROLLER_EPOCHS):
    print(f"--- Episode {episode + 1}/{CONTROLLER_EPOCHS} ---")
    episode_architectures_performance = [] # Store performance of architectures in this episode

    for cnn_model_index in range(N_CNN_MODELS_PER_EPISODE):
        print(f"  Generating CNN Architecture {cnn_model_index + 1}/{N_CNN_MODELS_PER_EPISODE}")

        state = np.zeros((1, 1, action_space_size)) # Initial state for LSTM (e.g., all zeros)
        architecture_actions = [] # Actions chosen by DQN for this architecture
        architecture_sequence = [] # Layer type sequence for this architecture

        for _ in range(MAX_LAYERS): # Generate up to MAX_LAYERS layers
            action = dqn_agent.choose_action(state) # Agent chooses action (layer type index)
            architecture_actions.append(action)

            layer_type = LAYER_TYPES[action % len(LAYER_TYPES)] # Get layer type from action index
            architecture_sequence.append(layer_type)

            next_state = np.zeros((1, 1, action_space_size)) # For simplicity, next state is also zeros. In more complex NAS, state could represent architecture so far.

            # --- Train and Evaluate Generated CNN Architecture ---
            generated_architecture = architecture_generator.generate_architecture(architecture_actions)

            if generated_architecture: # Check if architecture is valid
                apply_one_shot_training(None, trained_architectures_buffer) # Placeholder for one-shot training

                best_accuracy = 0.0
                best_params = 0
                for lr_cnn in LEARNING_RATES_CNN:
                    for optimizer_cnn in OPTIMIZERS_CNN:
                        accuracy, params = train_and_evaluate_architecture(generated_architecture, epochs=M_EPOCHS_PER_CNN, learning_rate=lr_cnn, optimizer_name=optimizer_cnn)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = params


                parameter_penalty = best_params / 1e6 # Normalize parameter count to millions for penalty

                reward = (REWARD_ACCURACY_WEIGHT * best_accuracy) - (REWARD_PARAMETER_PENALTY_WEIGHT * parameter_penalty)  # Modified reward function

                print(f"    Architecture: {[layer['type'] for layer in generated_architecture]}, Validation Accuracy: {best_accuracy:.4f}, Params: {best_params}, Reward: {reward:.4f}")

                episode_architectures_performance.append({
                    'architecture': [layer['type'] for layer in generated_architecture],
                    'accuracy': best_accuracy,
                    'params': best_params,
                    'reward': reward
                })

                dqn_agent.store_transition(state, action, reward, next_state, False) # Store experience in replay buffer

                state = next_state # Move to next state (for simplicity, next state is always zero in this example)
            else:
                reward = -1.0 # Penalty for invalid architecture
                print("    Invalid Architecture Generated, Reward: -1.0")
                dqn_agent.store_transition(state, action, reward, next_state, False) # Store experience for invalid architecture

                state = next_state


        dqn_agent.train_agent() # Train DQN agent after each episode (or after a batch of CNN models)
        if episode % dqn_agent.update_target_frequency == 0:
            dqn_agent.update_target_network() # Update target network periodically

        dqn_agent.decay_epsilon() # Decay epsilon after each episode

    # --- Episode Summary and Saving ---
    print("\n--- Episode Summary ---")
    avg_reward_episode = np.mean([perf['reward'] for perf in episode_architectures_performance])
    avg_accuracy_episode = np.mean([perf['accuracy'] for perf in episode_architectures_performance])
    print(f"  Average Reward: {avg_reward_episode:.4f}, Average Accuracy: {avg_accuracy_episode:.4f}")

    # Sort architectures by reward and save top performing ones
    episode_architectures_performance.sort(key=lambda x: x['reward'], reverse=True) # Sort by reward
    top_architectures = episode_architectures_performance[:5] # Save top 5 architectures

    episode_results = {
        'episode': episode + 1,
        'average_reward': avg_reward_episode,
        'average_accuracy': avg_accuracy_episode,
        'top_architectures': top_architectures
    }

    # Save episode results to JSON file
    results_filename = os.path.join(RESULTS_SAVE_PATH, f"episode_results_{episode+1}.json")
    with open(results_filename, 'w') as f:
        json.dump(episode_results, f, indent=4)
    print(f"  Episode results saved to: {results_filename}")


    if (episode + 1) % 10 == 0: # Save DQN model every 10 episodes
        dqn_agent.save_model(os.path.join(MODEL_SAVE_PATH, f"dqn_agent_episode_{episode+1}"))
        print(f"  DQN Agent model saved at episode {episode+1}")


print("\n--- Training Complete ---")
print(f"Top architectures and DQN agent models saved in '{RESULTS_SAVE_PATH}' and '{MODEL_SAVE_PATH}' respectively.")