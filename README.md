# CIFAR10MSPREDICT: A Pattern-Guided Microservices Identification Methodology

This repository contains the code and resources for the research project:  
**“A Pattern-Guided Microservices Identification Methodology to Support the Migration of Monolithic ML-based Systems to Microservices.”**

The project demonstrates a systematic methodology for migrating monolithic machine learning (ML)-based systems into a microservice-based architecture using design patterns. It is built on a monolithic ML image classification system using the CIFAR-10 dataset, and the methodology is validated by transforming the system into a microservices-based architecture.

---

## **Repository Structure**

### **Monolithic System**
The original monolithic system includes:
- A web interface for image classification using trained ML models.
- Key phases of the ML pipeline:
  - **Preprocessing**: Data cleaning, transformation, labeling, and feature engineering.
  - **Model Training**: Model initialization, hyperparameter tuning, and optimization.
  - **Model Evaluation**: Performance assessment using metrics and validation.
  - **(Optional)** Model Deployment and Monitoring phases for production use.

### **Microservices-Based System**
The migrated system decomposes the monolith into independent microservices based on ML pipeline phases:
1. **Preprocessing Microservice**  
   Handles data collection, cleaning, transformation, and feature engineering. Includes strategies like basic preprocessing and data augmentation.
2. **Model Training Microservice**  
   Trains models such as CNN and DenseNet. Implements flexibility through the Strategy pattern for choosing between different training models and the State pattern for managing training life-cycle phases.
3. **Model Evaluation Microservice**  
   Assesses model performance with metrics and validation tasks. Ensures models meet quality standards before deployment.
4. **Model Monitoring Microservice**  
   Tracks the performance of deployed models using the Observer pattern for real-time updates.

Each microservice adheres to the **Single Responsibility Principle**, ensuring scalability and maintainability.

---

## **Architecture**

### **Monolithic Architecture**
The monolithic architecture incorporates all phases of the ML pipeline in a single system, depicted in the "Monolith System Overview" diagram.

### **Microservices Architecture**
The migrated architecture separates the ML pipeline into distinct microservices. This system is structured using design patterns to enhance flexibility and modularity, as shown in the "Final Microservices-Based ML System Architecture" diagram.

**Design Patterns Utilized**:
- **Strategy Pattern**: Applied in preprocessing and training microservices for dynamic selection of strategies/models.
- **State Pattern**: Used in training microservices to manage model training life-cycle stages.
- **Observer Pattern**: Implemented in monitoring microservices for real-time performance tracking.

---

## **Case Study**

### **Objective**
To validate the proposed methodology, we used a monolithic ML-based image classification system built with Python and Flask using the CIFAR-10 dataset. It allows users to upload images and receive predictions based on trained models.

### **Dataset**
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html): A widely-used benchmark dataset for image classification containing 60,000 32x32 color images in 10 classes.

### **Models Used**
1. **Convolutional Neural Network (CNN)**
2. **DenseNet**

---

## **Setup and Usage**

### **Prerequisites**
- Python 3.8 or later
- Flask
- RabbitMQ (for the Event-Driven Architecture version)
- Other dependencies specified in `requirements.txt`

### **Steps to Run**

#### **Monolithic System**
1. Navigate to the `monolithic/` directory.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
3. Run the Flask server:
```bash
Copy code
python app.py

Access the web interface at http://localhost:5000.
Microservices-Based System
Navigate to the microservices/ directory.
Start RabbitMQ (for EDA-based architecture).
Deploy individual microservices:
Preprocessing
Training
Evaluation
Monitoring
Run each microservice using:
bash
Copy code
python service_name.py
Access the API Gateway or web interface.
Methodology
The proposed methodology is divided into three steps:

Layered ML Architecture
Decomposes the monolith using the Layered ML Architecture Pattern to identify ML-related components.
ML Pipeline Pattern
Identifies microservice candidates within the ML layer by mapping distinct pipeline stages.
GoF Design Patterns
Applies design patterns (Strategy, State, Observer) to restructure and optimize the identified microservices.
Contribution
This project showcases a novel approach for transitioning ML-based monolithic systems into microservices using structured patterns. The proposed methodology enhances:

Scalability: Independent scaling of microservices.
Maintainability: Modular design with reduced coupling.
Flexibility: Adaptable to various ML applications.
References
Yokoyama, S. et al.: Layered ML Architecture Pattern
Amershi, S. et al.: ML Pipeline Pattern
Gamma, E. et al.: GoF Design Patterns
Take et al.: Adaptation of Design Patterns for ML Context