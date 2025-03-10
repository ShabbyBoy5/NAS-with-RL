# utils.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from config import DATASET_NAME, VALIDATION_SPLIT, BATCH_SIZE_CNN, LEARNING_RATE_CNN, OPTIMIZERS_CNN, LEARNING_RATES_CNN, LAYER_TYPES # ADDED LAYER_TYPES here
import numpy as np

def load_dataset():
    """Loads and preprocesses the specified dataset."""
    if DATASET_NAME == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
        x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    elif DATASET_NAME == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) # One-hot encode
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)   # One-hot encode

    elif DATASET_NAME == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=100) # One-hot encode
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)   # One-hot encode
    else:
        raise ValueError(f"Dataset '{DATASET_NAME}' not supported.")

    num_validation_samples = int(len(x_train) * VALIDATION_SPLIT)
    x_val = x_train[:num_validation_samples]
    y_val = y_train[:num_validation_samples]
    x_train = x_train[num_validation_samples:]
    y_train = y_train[num_validation_samples:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def train_and_evaluate_architecture(architecture, epochs=10, learning_rate=LEARNING_RATE_CNN, optimizer_name='adam'):
    """Trains and evaluates a CNN architecture, trying different optimizers and learning rates."""

    model = None
    try: # Catch potential errors during model building (e.g., invalid shapes)
        model = architecture_generator.build_model_from_architecture(architecture)
    except Exception as e:
        print(f"Error building model: {e}")
        return 0.0, 0 # Return 0 accuracy and 0 parameters if model build fails


    if model is None: # Check if model build was successful
        return 0.0, 0

    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy' if DATASET_NAME in ['cifar10', 'cifar100'] else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    (x_train, y_train), (x_val, y_val), _ = load_dataset() # Load dataset for training

    try:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE_CNN, validation_data=(x_val, y_val), verbose=1) # Train model
        validation_accuracy = history.history['val_accuracy'][-1] if history.history['val_accuracy'] else 0.0 # Get last epoch val accuracy
        parameter_count = model.count_params()
        return validation_accuracy, parameter_count

    except Exception as e:
        print(f"Error training model: {e}")
        return 0.0, 0 # Return 0 accuracy and 0 parameters if training fails


def get_optimizer(optimizer_name, learning_rate):
    """Returns a Keras optimizer instance based on name."""
    if optimizer_name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")


import architecture_generator # Import here to avoid circular import issue
architecture_generator = architecture_generator.ArchitectureGenerator(len(LAYER_TYPES)) # Initialize after function definitions