# NeuroStage
"NeuroStage is a framework that allows users to create and manage deep learning projects in a structured and modular way, adapted for TensorFlow. It includes integration with tools like Tensorboard, enabling users to efficiently track and improve their models."

# Purpose
NeuroStage was born from the idea of automatically generating projects, with a specific focus on building deep learning models using TensorFlow. It is a tool designed for new users who need a standard structure without having to worry about organizing the project from scratch.

# Índice

1. [Design](#Design)
2. [Features](#Features)
3. [Installation](#installation) 
4. [Usage Flow](#usage-flow)
5. [Common Errors](https://github.com/catalina-delgado/NeuroStage/blob/main/TROUBLESHOOTING.md)
   
# Design
It is designed as a layer-based pattern (for building modules, architectures, and training) which is excellent for organizing a TensorFlow framework for deep learning testing. This modular approach facilitates integration with TensorBoard and promotes scalability. 

## Modules

**Layers** Define base layers here (e.g., convolutional, attention, etc.) that can be used in models. These layers form the building blocks for your deep learning models.

**Models** Combine the layers to create specific architectures for evaluation. This module allows you to design and implement various deep learning models by reusing and combining different layers.

**Training** Conduct experiments with specific configurations, logging metrics, parameters, and artifacts. This module focuses on the training process, helping you to configure and run training sessions, and track the performance and results.

# Features

| Feature                  | DeepTrain                                              |
|--------------------------|--------------------------------------------------------|
| Model Management         | Allows customization for versioning and saving models. |
| Test Automation          | Executes each training session in series as defined by the training module. |                                                       |
| Tool Compatibility       | TensorFlow, TensorBoard, OpenCV, Numpy                                            |
| Open Source              | MIT License                                            |
| Flexibility              | Preconfigured but flexible, define rules and processes as per your case |
| Collaboration            | Avalable                                               |


## Project Structure
```
my_project/
│
├── config.py             # Project configuration file
├── utils.py              # General utilities file
├── functions.py          # Training functions file
├── imports.py            # Library imports file
├── experiments/          # Folder for experiments
└── src/                  # Main source code folder
    ├── layers/           # Folder for implementing custom layers
    │   └── layer_a.py    # Example content
    │   └── layer_b.py 
    ├── models/           # Folder for defining models
    │   └── model_a.py    # Example content
    │   └── model_b.py 
    └── training/         # Folder for compiling and starting training
        └── train_a.py    # Example content
        └── train_b.py
```
# Installation
To install **NeuroStage**, simply run the following command:
``` 
pip install neurostage
```
For more detailed information, visit the project page on PyPI:
[NeuroStage](https://pypi.org/project/neurostage/)
# Usage-Flow
## Start a new project
To start a new project, use the following command. You can replace `my_project` with your desired project name:

```
stage startproject my_project
```
## create a new layer
File: `src/layers/layer_custom.py`
```python
from imports import tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        return self.dense(inputs)

```
## create a new model
File: `src/models/model_custom.py`
```python
from imports import tf
from src.layers.layer_custom import CustomLayer

class ModelCustom():
    def __init__(self): 
        super(Model, self).__init__() 
        self.layer = CustomLayer(64)
        
        
    def build_model(self, input):
        x = self.layer(input)
        x = self.layer(x)
        x = self.layer(x)
        
        model = tf.keras.Model(inputs=input, outputs=x)
        
        return model
```
## Create a training runner
To ensure that the framework automatically recognizes the class to execute with the `run` command, the training file **must start with the word "train"** in its filename.

### Example:
File: `src/training/train_custom.py`  
```python
from functions import NeuroStage
from imports import tf, np
from src.models.model import ModelCustom

class TrainModel(NeuroStage):
    def __init__(self, batch_size=32, epochs=4, model_name='', models=None):
        super().__init__()
        
        self.BATCH_SIZE = batch_size
        self.EPHOCS = epochs
        self.MODEL_NAME = model_name
        
        input = tf.keras.Input(shape=(256, 256, 1))  
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.95)
        self.architecture = ModelCustom()
        print(models)
        self.model = self.architecture.build_model(input)
        self.model.compile(optimizer=self.optimizer,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
                  
    def train(self):
        X_train = np.random.rand(100, 256, 256, 1)
        y_train = np.random.randint(0, 2, 100) 
        X_val = np.random.rand(20, 256, 256, 1) 
        y_val = np.random.randint(0, 2, 20)
        
        self.init_fit(self.model, X_train, y_train, X_val, y_val, self.EPHOCS, self.BATCH_SIZE, self.MODEL_NAME)
```
By following this naming convention, the framework will automatically detect and execute the training class when running the following command:
```
stage run
```

## Training function: init_fit
The `init_fit` function is responsible for training a deep learning model using TensorFlow, providing essential features for monitoring, saving, and restoring the model.

### Key functionalities:

1. **TensorBoard logging:**
   - Logs training metrics in the `experiments/{model_name}/logs-<timestamp>` directory.
   - Allows visualization of training performance using TensorBoard.

2. **Model checkpointing:**
   - Saves the best model based on validation accuracy at `experiments/{model_name}/{model_name}.h5`.
   - Ensures only the best version is stored.

3. **Resetting model states:**
   - Useful for models with recurrent layers that maintain states.

4. **Model training:**
   - Trains the model using the provided data and defined parameters.
   - Uses callbacks for logging and checkpointing.

5. **Loading best weights:**
   - After training, loads the best saved weights to ensure optimal performance.

6. **Final model saving:**
   - Saves the fully trained model for later use.

7. **Completion messages:**
   - Provides feedback with the model save path and training completion message.

### Example usage inside the training script:
```python
self.init_fit(self.model, X_train, y_train, X_val, y_val, self.EPHOCS, self.BATCH_SIZE, self.MODEL_NAME)
```

This function helps streamline the training workflow, ensuring efficient tracking and reproducibility.

## Execution
```
stage run --batch_size 32 --epochs 10 --model_name my_model
```
