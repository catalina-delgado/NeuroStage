"""
NeuroStage Framework

Version: 1.0
Propósito: Framework para entrenamiento, evaluación y gestión de experimentos en Deep Learning.
PYPI page: https://pypi.org/project/neurostage/

"""
        
from imports import *


class NeuroStage():
    def __init__(self):
        print("=" * 50)
        print(f"NeuroStage Framework v1.0")
        print("=" * 50)

    def get_summary(self, model, file_name):
        file_path = os.path.join('graphs', file_name)
        os.makedirs(file_path, exist_ok=True)

        with open(os.path.join(file_path, f'{file_name}.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()

        plot_model(model, to_file=os.path.join(file_path, f'{file_name}.png'),
                   show_shapes=True, show_layer_names=True)
    
    def init_fit(self, model, x_train, y_train, x_val, y_val, x_test=None, y_test=None, EPOCHS=100, BATCH_SIZE=32, model_name='my_model', custom_layers=False, save_best_only=False):
        print(f'Training model: {model_name}')
        
        log_dir = f"experiments/{model_name}/logs-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        model_dir = f"experiments/{model_name}/"
        os.makedirs(model_dir, exist_ok=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [tensorboard_callback]
        
        filepath_best = f'{model_dir}/saved_model'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath_best, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max',
            verbose=1,
            save_format='tf'
        )
        
        if save_best_only:
            callbacks.append(checkpoint)
            metrics = {
                "loss": [],
                "accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
        else:
            class SaveEachEpoch(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    save_path = os.path.join(model_dir, f"model_epoch_{epoch + 1}")
                    self.model.save(save_path)
            
            metrics = {
                "loss": [],
                "accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
                "test_loss": [],
                "test_accuracy": []
            }

            callbacks.append(SaveEachEpoch())
        
        if custom_layers:
            for name, layer in custom_layers.items():
                print(f"Adding custom layer: {name}")
                tf.keras.utils.get_custom_objects()[name] = layer

        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        
        
        history = model.fit(
            x_train, 
            y_train, 
            epochs=EPOCHS, 
            validation_data=(x_val, y_val), 
            callbacks=callbacks,
        )
        
        for epoch_logs in history.history['loss']:
            metrics["loss"].append(epoch_logs)
        
        for epoch_logs in history.history['accuracy']:
            metrics["accuracy"].append(epoch_logs)

        for epoch_logs in history.history['val_loss']:
            metrics["val_loss"].append(epoch_logs)

        for epoch_logs in history.history['val_accuracy']:
            metrics["val_accuracy"].append(epoch_logs)
            
        tf.keras.backend.clear_session()
        gc.collect()
        
        if save_best_only:
            print(f"Best model saved at {filepath_best}")
        else:
            print("\nTesting all models saved :")
            model_paths = [f for f in os.listdir(model_dir) if f.startswith("model_epoch_")]
            model_paths.sort(key=lambda x: int(x.split("_")[-1])) 
            
            for i, model_file in enumerate(model_paths):
                model_path = os.path.join(model_dir, model_file)
                try:
                    loaded_model = tf.keras.models.load_model(model_path)
                    if x_test is not None and y_test is not None:
                        test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
                        metrics["test_loss"].append(test_loss)
                        metrics["test_accuracy"].append(test_accuracy)
                        print(f"Model {i + 1} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
                    else:
                        print(f"Model {i + 1} is missing X_test and y_test for evaluation.")
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {e}")

        print("Training completed and logged in TensorBoard")

        return metrics