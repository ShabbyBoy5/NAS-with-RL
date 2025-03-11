# config.py

# --- DQN Agent Hyperparameters ---

DQN_LEARNING_RATE = 0.001

DQN_GAMMA = 0.99 # Discount factor

DQN_EPSILON_INITIAL = 1.0

DQN_EPSILON_DECAY = 0.0001 # Adjusted epsilon decay

DQN_EPSILON_MIN = 0.01

REPLAY_BUFFER_SIZE = 10000

BATCH_SIZE = 32

TARGET_UPDATE_FREQUENCY = 5 # Update target network every N episodes

CONTROLLER_EPOCHS = 100 # Number of Controller (DQN) training epochs

M_EPOCHS_PER_CNN = 10 # Number of epochs to train each generated CNN in NAS loop

N_CNN_MODELS_PER_EPISODE = 20 # Number of CNN architectures to generate per DQN episode

# --- Reward Function Weights ---

REWARD_ACCURACY_WEIGHT = 1.5 # Increased emphasis on accuracy

REWARD_PARAMETER_PENALTY_WEIGHT = 0.005 # Decreased emphasis on parameter count

REWARD_ENERGY_WEIGHT = 0.0 # Weight for energy (set to 0 for simplicity)

# --- Search Space Parameters ---

MAX_LAYERS = 8 # Maximum number of layers in CNN architecture

INPUT_SHAPE = (32, 32, 3) # Input shape for CIFAR datasets (adjust for MNIST)

NUM_CLASSES = 10 # Number of classes for CIFAR10 (adjust for MNIST/CIFAR100)

# --- Layer Types ---

LAYER_TYPES = [
    'Conv2D', 'Conv2DTranspose', 'SeparableConv2D', 'DepthwiseConv2D',
    'MaxPooling2D', 'AveragePooling2D', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D',
    'Dropout', 'BatchNormalization', 'Flatten', 'Dense'
]

# --- Training Parameters ---

DATASET_NAME = 'cifar10' # 'mnist', 'cifar10', 'cifar100'

VALIDATION_SPLIT = 0.1 # Validation data split

INITIAL_EPSILON = 1.0

EPSILON_DECAY = 0.001

MIN_EPSILON = 0.01

BATCH_SIZE_CNN = 64

LEARNING_RATE_CNN = 0.001

OPTIMIZERS_CNN = ['adam', 'rmsprop', 'sgd'] # Optimizers to try for each CNN

LEARNING_RATES_CNN = [0.01, 0.001, 0.0001] # Learning rates to try for each CNN

# --- Save and Load Paths ---

MODEL_SAVE_PATH = 'saved_models/'

RESULTS_SAVE_PATH = 'results/'
