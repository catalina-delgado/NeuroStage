# NeuroStage
"NeuroStage is a framework that allows users to create and manage deep learning projects in a structured and modular way, adapted for TensorFlow. It includes integration with tools like Tensorboard, enabling users to efficiently track and improve their models."

# Purpose
NeuroStage was created to simplify the automatic generation of projects, with a strong focus on developing deep learning models using TensorFlow and running experiments sequentially. It is designed for new users who seek a standardized project structure without the hassle of organizing everything from scratch.

# Índice

- [NeuroStage](#neurostage)
- [Purpose](#purpose)
- [Índice](#índice)
- [Design](#design)
  - [Modules](#modules)
- [Features](#features)
  - [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage-Flow](#usage-flow)
  - [Start a new project](#start-a-new-project)
  - [create a new layer](#create-a-new-layer)
  - [create a new model](#create-a-new-model)
  - [Create a training runner](#create-a-training-runner)
  - [Training function: init\_fit](#training-function-init_fit)
    - [Key functionalities:](#key-functionalities)
    - [Example usage inside the training script:](#example-usage-inside-the-training-script)
  - [Execution](#execution)
  - [Experiments](#experiments)
- [Contribute 📚](#contribute-)
- [Try and Star the Project 🌟](#try-and-star-the-project-)
   
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
├── src/                  # Main source code folder
│    ├── layers/           # Folder for implementing custom layers
│    │   └── layer_a.py    # Example content
│    │   └── layer_b.py 
│    ├── models/           # Folder for defining models
│    │   └── model_a.py    # Example content
│    │   └── model_b.py 
├── training/             # Folder for compiling and starting training
│    └── train_a.py        # Example content
│    └── train_b.py
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
        self.conv = tf.keras.layers.Conv2D(units, 3, activation='relu')
        self.dense = tf.keras.layers.Dense(units)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

```
## create a new model
File: `src/models/model_custom.py`
```python
from imports import tf
from src.layers.layer_custom import CustomLayer

class ModelCustom():
    def __init__(self): 
        super(ModelCustom, self).__init__() 
        self.layer = CustomLayer(64)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def build_model(self, input):
        x = self.layer(input)
        x = self.dense(x)
        
        model = tf.keras.Model(inputs=input, outputs=x)
        
        return model
```
## Create a training runner
To ensure that the framework automatically recognizes the class to execute with the `run` command, the training file **must start with the word "train"** in its filename.

File: `training/train_custom.py`  
```python
from functions import NeuroStage
from imports import tf, np
from src.models.model_custom import ModelCustom

class TrainModel(NeuroStage):
    def __init__(self, batch_size=32, epochs=4):
        super().__init__()
        
        self.BATCH_SIZE = batch_size
        self.EPHOCS = epochs
        self.MODEL_NAME = 'example_model'
        
        input = tf.keras.Input(shape=(256, 256, 1))  
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.95)
        self.architecture = ModelCustom()
        self.model = self.architecture.build_model(input)
        self.model.compile(optimizer=self.optimizer,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
                  
    def train(self):
        X_train = np.random.rand(100, 256, 256, 1)
        y_train = np.random.randint(0, 2, 100) 
        X_val = np.random.rand(20, 256, 256, 1) 
        y_val = np.random.randint(0, 2, 20)
        X_test = np.random.rand(20, 256, 256, 1) 
        y_test = np.random.randint(0, 2, 20)
        
        history = self.init_fit(self.model, X_train, y_train, X_val, y_val, X_test, y_test, self.EPHOCS, self.BATCH_SIZE, self.MODEL_NAME)
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
   - Saves the best model based on validation accuracy at `experiments/{model_name}/{model_name}.h5`
   - Ensures only the best version is stored when save_best_only=`True`. If save_best_only=`False`, saves the model at the end of every epoch individually.

3. **Model training:**
   - Trains the model using the provided data and defined parameters.
   - Applies custom callbacks for checkpointing and TensorBoard logging.
   - Supports integration of custom layers if needed.

4. **Custom layers support:**
   - Allows registration of custom layers before training, ensuring compatibility when saving and loading models.

5. **Model saving:**
   - Saves the fully trained model for later use.

6. **Testing saved models:**
   - If models are saved after each epoch (save_best_only=False), automatically evaluates them on test data and logs the test loss and accuracy.

7. **Completion messages:**
   - Provides feedback with the model save path and training completion message.

### Example usage inside the training script:
```python
self.init_fit(self.model, X_train, y_train, X_val, y_val, self.EPHOCS, self.BATCH_SIZE, model_name=MODEL_NAME)
```

This function helps streamline the training workflow, ensuring efficient tracking and reproducibility.

## Execution
Enter your project folder, for this example the `my_project` folder, and run your training files.
```
cd my_project
stage run --batch_size 32 --epochs 10 
```

## Experiments
To visualize experiment results, navigate to the experiments folder and execute the following command to launch TensorBoard:
```
tensorboard --logdir=experiments
```
Once TensorBoard is running, click on the provided server link in the terminal (usually starting with `http://localhost:6006`) to open the dashboard in your web browser.

# Contribute 📚

We welcome contributions to NeuroStage! If you have ideas for new features, improvements, or bug fixes, feel free to contribute by submitting a pull request.

# Try and Star the Project 🌟

If you find NeuroStage helpful, please consider trying it out and giving it a star on GitHub. Your support helps us grow and improve the project!

Thank you for being part of our community! 💙