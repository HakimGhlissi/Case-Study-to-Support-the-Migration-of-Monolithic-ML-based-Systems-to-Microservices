from flask import Flask, jsonify, request
from io import BytesIO
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from kerastuner import RandomSearch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import classes from the provided code
from ML_PipelineRevised import (DataCollection, DataCleaning, DataLabeling, 
                       FeatureEngineering, CNNModelTuner, ModelMonitoring, ModelEvaluation)

app = Flask(__name__)

# Define class names corresponding to CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/load_and_preprocess', methods=['POST'])
def load_and_preprocess():
    global x_train, y_train, x_test, y_test
    
    # Load and preprocess the CIFAR-10 dataset
    data_collector = DataCollection()
    (x_train, y_train), (x_test, y_test) = data_collector.load_data()

    # Data cleaning, normalization, and labeling
    cleaner = DataCleaning()
    x_train, y_train, x_test, y_test = cleaner.remove_missing((x_train, y_train, x_test, y_test))
    labeler = DataLabeling()
    y_train = labeler.to_categorical(y_train, num_classes=10)
    y_test = labeler.to_categorical(y_test, num_classes=10)
    feature_engineer = FeatureEngineering()
    x_train, x_test = feature_engineer.normalize(x_train, x_test)
    
    return jsonify({"message": "Data loaded and preprocessed successfully."})

@app.route('/train', methods=['POST'])
def train_model():
    if 'x_train' not in globals():
        return jsonify({"error": "Data not loaded and preprocessed."}), 400
    
    global cnn_model, history
    epochs = 1  # Set epochs

    # Set up data augmentation
    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    # Model tuning and training
    cnn_tuner = CNNModelTuner(input_shape=(32, 32, 3), num_classes=10)
    tuner = RandomSearch(
        cnn_tuner.build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='cnn_tuner_dir',
        project_name='cifar10'
    )
    tuner.search(datagen.flow(x_train, y_train, batch_size=32),
                 epochs=epochs,
                 validation_data=(x_test, y_test))

    # Get best model
    cnn_model = tuner.get_best_models(num_models=1)[0]

    # Monitoring model
    monitor = ModelMonitoring()
    history = monitor.start_monitoring(cnn_model, datagen.flow(x_train, y_train), (x_test, y_test), epochs)

    # Save the model
    model_path = "best_cnn_model.h5"
    cnn_model.save(model_path)

    return jsonify({"message": "Model trained and saved successfully.", "model_path": model_path})


@app.route('/train_dense', methods=['POST'])
def train_dense_model():
    if 'x_train' not in globals():
        return jsonify({"error": "Data not loaded and preprocessed."}), 400

    global dense_model, dense_history
    epochs = 1  # Set epochs for demonstration

    # Set up data augmentation (optional)
    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    # Model tuning and training
    dense_tuner = DenseModelTuner(input_shape=(32, 32, 3), num_classes=10)
    tuner = RandomSearch(
        dense_tuner.build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='dense_tuner_dir',
        project_name='dense_cifar10'
    )
    tuner.search(datagen.flow(x_train, y_train, batch_size=32),
                 epochs=epochs,
                 validation_data=(x_test, y_test))

    # Get best model
    dense_model = tuner.get_best_models(num_models=1)[0]

    # Monitoring model
    monitor = ModelMonitoring()
    dense_history = monitor.start_monitoring(dense_model, datagen.flow(x_train, y_train), (x_test, y_test), epochs)

    # Save the model
    model_path = "best_dense_model.h5"
    dense_model.save(model_path)

    return jsonify({"message": "Dense model trained and saved successfully.", "model_path": model_path})


@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    if 'cnn_model' not in globals():
        return jsonify({"error": "Model not trained."}), 400

    model_path = request.json['model_path']
    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found."}), 404

    # Load the model
    model = load_model(model_path)
    
    # Evaluate the model
    evaluator = ModelEvaluation()
    loss, accuracy = evaluator.evaluate(model, x_test, y_test)

    return jsonify({"loss": loss, "accuracy": accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    model_path = request.json['model_path']
    image_url = request.json['image_url']

    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found."}), 404

    # Load the model
    model = load_model(model_path)

    # Load and preprocess the image from URL
    response = requests.get(image_url)
    img = image.load_img(BytesIO(response.content), target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[class_idx]

    return jsonify({"prediction": predicted_class})

@app.route('/plot_metrics', methods=['GET'])
def plot_metrics():
    if 'history' not in globals():
        return jsonify({"error": "No training history available."}), 400

    # Plotting accuracy and loss metrics
    monitor = ModelMonitoring()
    monitor.history = history
    monitor.plot_training_metrics()

    return jsonify({"message": "Training metrics plotted successfully."})

@app.route('/confusion_matrix', methods=['GET'])
def plot_confusion_matrix():
    if 'cnn_model' not in globals():
        return jsonify({"error": "Model not trained."}), 400

    # Log and display confusion matrix
    monitor = ModelMonitoring()
    monitor.log_confusion_matrix(cnn_model, x_test, y_test)

    return jsonify({"message": "Confusion matrix displayed successfully."})

if __name__ == '__main__':
    app.run(debug=True)
