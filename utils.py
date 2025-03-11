# utils.py
import tensorflow as tf
from keras.datasets import mnist, cifar10, cifar100
from config import DATASET_NAME, VALIDATION_SPLIT, BATCH_SIZE_CNN, LEARNING_RATE_CNN
import numpy as np

def load_dataset():
    """Loads and preprocesses the specified dataset."""
    if DATASET_NAME == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Reshape to include channel dimension and normalize
        x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
        x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    elif DATASET_NAME == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    elif DATASET_NAME == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    else:
        raise ValueError(f"Dataset '{DATASET_NAME}' not supported.")

    # Split training data into train and validation sets
    num_validation_samples = int(len(x_train) * VALIDATION_SPLIT)
    x_val = x_train[:num_validation_samples]
    y_val = y_train[:num_validation_samples]
    x_train = x_train[num_validation_samples:]
    y_train = y_train[num_validation_samples:]

    # Print dataset shapes for debugging
    print(f"Dataset loaded: {DATASET_NAME}")
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_optimizer(optimizer_name, learning_rate):
    """Returns a Keras optimizer instance based on name."""
    if optimizer_name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        print(f"Warning: Optimizer '{optimizer_name}' not supported. Using Adam.")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train_and_evaluate_architecture(architecture, epochs=10, learning_rate=LEARNING_RATE_CNN, optimizer_name='adam'):
    """Trains and evaluates a CNN architecture."""
    # Import here to avoid circular import
    from architecture_generator import ImprovedArchitectureGenerator
    from config import LAYER_TYPES
    
    if not architecture:
        print("Error: Empty architecture provided")
        return 0.0, 0
    
    try:
        # Create generator and build model
        arch_generator = ImprovedArchitectureGenerator(len(LAYER_TYPES))
        model = arch_generator.build_model_from_architecture(architecture)
        
        if model is None:
            print("Error: Failed to build model")
            return 0.0, 0
        
        # Compile model

    
        optimizer = get_optimizer(optimizer_name, learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load dataset
        (x_train, y_train), (x_val, y_val), _ = load_dataset()
        
        # Add early stopping to avoid wasting time on poor architectures
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE_CNN,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Get final validation accuracy and parameter count
        if history.history['val_accuracy']:
            validation_accuracy = max(history.history['val_accuracy'])  # Use best accuracy
        else:
            validation_accuracy = 0.0
            
        parameter_count = model.count_params()
        
        return validation_accuracy, parameter_count
    
    except Exception as e:
        import traceback
        print(f"Error in train_and_evaluate_architecture: {e}")
        traceback.print_exc()
        return 0.0, 0