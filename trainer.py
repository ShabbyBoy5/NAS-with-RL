# improved_trainer.py
import tensorflow as tf
import numpy as np
import os
import json
from dqn_agent import ImprovedDQNAgent
from architecture_generator import ImprovedArchitectureGenerator
from utils import load_dataset, train_and_evaluate_architecture
from one_shot_training import apply_one_shot_training
from config import (MAX_LAYERS, LAYER_TYPES, CONTROLLER_EPOCHS, M_EPOCHS_PER_CNN,
                   MODEL_SAVE_PATH, RESULTS_SAVE_PATH, REWARD_ACCURACY_WEIGHT,
                   REWARD_PARAMETER_PENALTY_WEIGHT, N_CNN_MODELS_PER_EPISODE,
                   OPTIMIZERS_CNN, LEARNING_RATES_CNN,BATCH_SIZE)

# Ensure save directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

# Initialize Improved DQN Agent and Architecture Generator
action_space_size = len(LAYER_TYPES)
dqn_agent = ImprovedDQNAgent(action_space_size)
architecture_generator = ImprovedArchitectureGenerator(action_space_size)

# Load Dataset (for input shape and num_classes in config)
load_dataset()

# Training history tracking
training_history = {
    'episode_rewards': [],
    'episode_accuracies': [],
    'valid_architecture_rate': [],
    'constraint_violations': []
}

def get_state(architecture_sequence):
    """
    Create state representation for the DQN agent.
    Counts the number of each layer type in the current architecture.
    """
    state = np.zeros(action_space_size)
    
    if not architecture_sequence:
        return state
        
    for layer_type in architecture_sequence:
        if layer_type in LAYER_TYPES:
            idx = LAYER_TYPES.index(layer_type)
            state[idx] += 1
    
    # Normalize to prevent state values from growing too large
    if np.sum(state) > 0:
        state = state / np.sum(state)
        
    return state

# --- Main Training Loop ---
for episode in range(CONTROLLER_EPOCHS):
    print(f"--- Episode {episode + 1}/{CONTROLLER_EPOCHS} ---")
    
    episode_architectures_performance = []
    valid_architecture_count = 0
    total_architecture_count = 0
    
    # Reset constraint violation counters for this episode
    dqn_agent.reset_constraint_statistics()
    
    for cnn_model_index in range(N_CNN_MODELS_PER_EPISODE):
        print(f" Generating CNN Architecture {cnn_model_index + 1}/{N_CNN_MODELS_PER_EPISODE}")
        
        # Generate architecture step by step
        architecture_sequence = []
        actions_taken = []
        states = []
        rewards = []
        next_states = []
        dones = []
        
        # Create an architecture layer by layer
        for layer_idx in range(MAX_LAYERS):
            current_state = get_state(architecture_sequence)
            states.append(current_state)
            
            # Agent chooses action (layer type)
            action = dqn_agent.choose_action(current_state, architecture_sequence)
            actions_taken.append(action)
            
            # Map action to layer type
            layer_type = LAYER_TYPES[action % len(LAYER_TYPES)]
            
            # Check if this layer would maintain a valid architecture
            is_valid, constraint_type = architecture_generator.can_add_layer_type(architecture_sequence, layer_type)
            
            if is_valid:
                architecture_sequence.append(layer_type)
                
                # Check if we can stop here (have a complete architecture)
                has_flatten = 'Flatten' in architecture_sequence
                has_dense_after_flatten = False
                
                if has_flatten:
                    flatten_idx = architecture_sequence.index('Flatten')
                    has_dense_after_flatten = 'Dense' in architecture_sequence[flatten_idx:]
                
                if has_flatten and has_dense_after_flatten and len(architecture_sequence) >= 3:
                    # We have a potentially complete architecture
                    break
            else:
                if constraint_type:
                    dqn_agent.record_constraint_violation(constraint_type)
                
                # Skip this action - give a negative reward but continue building
                rewards.append(-0.5)  # Negative reward for invalid layer choice
                next_states.append(current_state)  # State doesn't change
                dones.append(False)   # Not terminal state
                
                # Store this negative experience
                dqn_agent.store_transition(
                    current_state, 
                    action, 
                    -0.5, 
                    current_state, 
                    False,
                    architecture_sequence
                )
                
                # Skip to next layer choice
                continue
        
        # Generate the full architecture with collected actions
        total_architecture_count += 1
        generated_architecture = architecture_generator.generate_architecture(actions_taken, dqn_agent)
        
        if generated_architecture:
            valid_architecture_count += 1
            print(f" Generated valid architecture with {len(generated_architecture)} layers")
            print(f" Layer sequence: {[layer['type'] for layer in generated_architecture]}")
            
            # Try different learning rates and optimizers for this architecture
            best_accuracy = 0.0
            best_params = 0
            
            for optimizer_cnn in OPTIMIZERS_CNN:
                for lr_cnn in LEARNING_RATES_CNN:
                    print(f" Training with optimizer={optimizer_cnn}, learning_rate={lr_cnn}")
                    
                    accuracy, params = train_and_evaluate_architecture(
                        generated_architecture, 
                        epochs=M_EPOCHS_PER_CNN,
                        learning_rate=lr_cnn,
                        optimizer_name=optimizer_cnn
                    )
                    
                    print(f" Training result: accuracy={accuracy:.4f}, params={params}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = params
            
            # Calculate reward based on accuracy and parameter count
            parameter_penalty = best_params / 1e6  # Normalize parameter count
            reward = (REWARD_ACCURACY_WEIGHT * best_accuracy) - (REWARD_PARAMETER_PENALTY_WEIGHT * parameter_penalty)
            
            # Bonus for completing a working architecture
            reward += 0.5
            
            print(f" Architecture performance: Accuracy={best_accuracy:.4f}, Params={best_params}, Reward={reward:.4f}")
            
            # Store performance for this architecture
            episode_architectures_performance.append({
                'architecture': [layer['type'] for layer in generated_architecture],
                'accuracy': best_accuracy,
                'params': best_params,
                'reward': reward
            })
            
            # Store successful architecture for future reference
            architecture_generator.store_successful_architecture(generated_architecture, best_accuracy)
            
            # Generate sequential training examples from this architecture
            # This creates multiple training examples from a single architecture
            partial_arch = []
            for idx, layer_type in enumerate(architecture_sequence):
                current_state = get_state(partial_arch)
                action = LAYER_TYPES.index(layer_type)
                
                # The next state after adding this layer
                partial_arch.append(layer_type)
                next_state = get_state(partial_arch)
                
                # Is this the final layer?
                is_terminal = (idx == len(architecture_sequence) - 1)
                
                # Calculate proportional reward for this step
                # Earlier layers get less reward, final layers get more
                step_reward = reward * ((idx + 1) / len(architecture_sequence)) if is_terminal else 0.1
                
                # Store this transition
                dqn_agent.store_transition(
                    current_state, 
                    action, 
                    step_reward, 
                    next_state, 
                    is_terminal,
                    partial_arch
                )
                
        else:
            print(" Generated invalid architecture")
            # Store negative experience for invalid architecture
            dqn_agent.store_transition(
                get_state([]), 
                actions_taken[0] if actions_taken else 0, 
                -1.0, 
                get_state([]), 
                True,
                []
            )
        
        # Train the DQN agent every few architectures
        if len(dqn_agent.memory) >= BATCH_SIZE:
            dqn_agent.train_agent()
    
    # Calculate valid architecture ratio
    valid_ratio = valid_architecture_count / max(1, total_architecture_count)
    print(f" Valid architecture ratio: {valid_ratio:.2f} ({valid_architecture_count}/{total_architecture_count})")
    
    # Update target network at the end of each episode
    dqn_agent.update_target_network()
    
    # Episode summary
    print("\n--- Episode Summary ---")
    
    # Store constraint violation statistics
    constraint_stats = dqn_agent.get_constraint_statistics()
    print(f" Constraint violations: {constraint_stats}")
    
    if episode_architectures_performance:
        avg_reward_episode = np.mean([perf['reward'] for perf in episode_architectures_performance])
        avg_accuracy_episode = np.mean([perf['accuracy'] for perf in episode_architectures_performance])
        print(f" Average Reward: {avg_reward_episode:.4f}, Average Accuracy: {avg_accuracy_episode:.4f}")
        
        # Update training history
        training_history['episode_rewards'].append(float(avg_reward_episode))
        training_history['episode_accuracies'].append(float(avg_accuracy_episode))
        training_history['valid_architecture_rate'].append(float(valid_ratio))
        training_history['constraint_violations'].append(constraint_stats)
        
        # Sort architectures by reward and save top performing ones
        episode_architectures_performance.sort(key=lambda x: x['reward'], reverse=True)
        top_architectures = episode_architectures_performance[:5]
        
        # Save episode results
        episode_results = {
            'episode': episode + 1,
            'average_reward': float(avg_reward_episode),
            'average_accuracy': float(avg_accuracy_episode),
            'valid_architecture_rate': float(valid_ratio),
            'constraint_violations': constraint_stats,
            'top_architectures': top_architectures
        }
    else:
        print(" No valid architectures generated in this episode")
        episode_results = {
            'episode': episode + 1,
            'average_reward': 0.0,
            'average_accuracy': 0.0,
            'valid_architecture_rate': 0.0,
            'constraint_violations': constraint_stats,
            'top_architectures': []
        }
    
    # Save episode results to JSON file
    results_filename = os.path.join(RESULTS_SAVE_PATH, f"episode_results_{episode+1}.json")
    with open(results_filename, 'w') as f:
        json.dump(episode_results, f, indent=4)
    
    print(f" Episode results saved to: {results_filename}")
    
    # Save DQN model periodically
    if (episode + 1) % 10 == 0 or episode == CONTROLLER_EPOCHS - 1:
        dqn_agent.save_model(os.path.join(MODEL_SAVE_PATH, f"dqn_agent_episode_{episode+1}"))
        print(f" DQN Agent model saved at episode {episode+1}")
        
        # Save overall training history
        history_filename = os.path.join(RESULTS_SAVE_PATH, "training_history.json")
        with open(history_filename, 'w') as f:
            json.dump(training_history, f, indent=4)

print("\n--- Training Complete ---")
print(f"Top architectures and DQN agent models saved in '{RESULTS_SAVE_PATH}' and '{MODEL_SAVE_PATH}' respectively.")