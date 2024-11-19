import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner import HyperParameters, RandomSearch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Classes for Data Collection, Cleaning, and Labeling
class DataCollection:
    """Handles data loading from CIFAR-10 dataset."""
    def load_data(self):
        # Load CIFAR-10 dataset from Keras datasets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

class DataCleaning:
    """Placeholder for data cleaning, e.g., handling missing values."""
    @staticmethod
    def remove_missing(data):
        # Returns data as-is, no actual cleaning for CIFAR-10
        return data

class FeatureEngineering:
    """Handles feature scaling and normalization."""
    @staticmethod
    def normalize(x_train, x_test):
        # Normalize pixel values to range 0-1
        return x_train / 255.0, x_test / 255.0

class DataLabeling:
    """Converts labels to categorical format for neural networks."""
    def to_categorical(self, y, num_classes):
        # Convert integer labels to one-hot encoding
        y = np.asarray(y, dtype='int32')
        categorical = np.zeros((y.size, num_classes))
        categorical[np.arange(y.size), y.flatten()] = 1
        return categorical

# Classes for Building and Tuning CNN and Dense Models
class CNNModelTuner:
    """Builds a CNN model with tunable hyperparameters."""
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self, hp):
        # Construct a CNN model with hyperparameters for tuning
        model = Sequential()
        # First convolutional layer with tunable filters
        model.add(Conv2D(
            filters=hp.Int('conv_1_filters', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3), padding='same', activation='relu', input_shape=self.input_shape
        ))
        model.add(BatchNormalization())
        # Second convolutional layer with tunable filters
        model.add(Conv2D(
            filters=hp.Int('conv_2_filters', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3), activation='relu'
        ))
        model.add(MaxPooling2D())
        # Dropout layer with tunable dropout rate
        model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

        # Fully connected dense layers
        model.add(Flatten())
        model.add(Dense(
            units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
            activation='relu'
        ))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model with tunable learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

# Hyperparameter tuning for Dense Model
class DenseModelTuner:
    """Builds a Dense (MLP) model with tunable hyperparameters."""
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self, hp):
        # Construct a Dense (fully-connected) model with hyperparameters
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            # First dense layer with tunable units
            Dense(
                units=hp.Int('dense_1_units', min_value=64, max_value=256, step=64),
                activation='relu'
            ),
            # Dropout layer with tunable dropout rate
            Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile model with tunable learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    

# Class for Monitoring and Logging Model Metrics
class ModelMonitoring:
    """Tracks and logs model metrics during and after training."""
    
    def __init__(self):
        self.history = None

    def start_monitoring(self, model, train_data, validation_data, epochs):
        # Training the model and saving history for monitoring
        self.history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[self.get_callbacks()]
        )
        return self.history

    def get_callbacks(self):
        # Define callbacks for monitoring
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
        ]
    
    def plot_training_metrics(self):
        # Plotting training and validation accuracy/loss
        if self.history is None:
            print("No training history available to plot.")
            return

        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def log_confusion_matrix(self, model, x_test, y_test):
        # Generate predictions and log confusion matrix
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

# Model Evaluation
class ModelEvaluation:
    """Evaluates the model on test data and prints the performance metrics."""
    def evaluate(self, model, x_test, y_test):
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Loss:", loss, "Accuracy:", accuracy)
        return loss, accuracy

# Main function to execute data processing, model tuning, and evaluation
def main():
    # Step 1: Load and preprocess data
    data_collector = DataCollection()
    (x_train, y_train), (x_test, y_test) = data_collector.load_data()

    # Step 2: Data cleaning (if needed)
    cleaner = DataCleaning()
    x_train, y_train, x_test, y_test = cleaner.remove_missing((x_train, y_train, x_test, y_test))

    # Step 3: Label encoding
    labeler = DataLabeling()
    y_train = labeler.to_categorical(y_train, 10)
    y_test = labeler.to_categorical(y_test, 10)

    # Step 4: Feature scaling
    feature_engineer = FeatureEngineering()
    x_train, x_test = feature_engineer.normalize(x_train, x_test)

    # Step 5: Data augmentation for better model generalization
    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    # Step 6: Hyperparameter tuning for CNN model
    cnn_tuner = CNNModelTuner(input_shape=(32, 32, 3), num_classes=10)
    tuner = RandomSearch(
        cnn_tuner.build_model,
        objective='val_accuracy',
        max_trials=5,  # Number of hyperparameter configurations to try
        executions_per_trial=3,  # Re-train each configuration this many times
        directory='cnn_tuner_dir',  # Directory for saving logs
        project_name='cnn_model_tuning'  # Name of the tuning project
    )
    # Conduct hyperparameter tuning on the training data
    tuner.search(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test))

    # Step 7: Evaluate the best CNN model
    best_cnn_model = tuner.get_best_models(num_models=1)[0]
    evaluator = ModelEvaluation()
    cnn_metrics = evaluator.evaluate(best_cnn_model, x_test, y_test)

    # Step 8: Monitor the best CNN model
    monitor = ModelMonitoring()
    monitor.start_monitoring(best_cnn_model, datagen.flow(x_train, y_train, batch_size=64), (x_test, y_test), epochs=5)
    monitor.plot_training_metrics()
    monitor.log_confusion_matrix(best_cnn_model, x_test, y_test)

    # Step 9: Hyperparameter tuning for Dense (MLP) model
    dense_tuner = DenseModelTuner(input_shape=(32, 32, 3), num_classes=10)
    dense_tuner = RandomSearch(
        dense_tuner.build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='dense_tuner_dir',
        project_name='dense_model_tuning'
    )
    dense_tuner.search(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test))

    # Step 10: Evaluate the best Dense (MLP) model
    best_dense_model = dense_tuner.get_best_models(num_models=1)[0]
    dense_metrics = evaluator.evaluate(best_dense_model, x_test, y_test)

if __name__ == "__main__":
    main()
