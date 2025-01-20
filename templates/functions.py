from imports import *

class NeuroStage():
    def __init__(self):
        print('NeuroStage')
        
    def get_summary(self, model):
        return model.summary()
    
    def init_fit(self, model, x_train, y_train, x_val, y_val, EPHOCS, BATCH_SIZE, model_name=''):
        
        log_dir = f"experiments/{model_name}/logs-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        filepath = f'experiments/{model_name}/{model_name}.h5'
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max',
            verbose=1
        )
        
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        
        history = model.fit(x_train, y_train, epochs=EPHOCS, 
                            validation_data=(x_val, y_val), 
                            callbacks=[tensorboard_callback, checkpoint]
                            )
        
        model.load_weights(filepath)
       
        model.save(filepath)
        print(filepath)
            
        print("Training completed and logged in TensorBoard")