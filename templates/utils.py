import os
import sys
import importlib

def run_train_scripts(batch_size, epochs, model_name, project_dir):
    
    train_dir = os.path.join(project_dir, 'training')
    sys.path.insert(0, train_dir)

    for filename in os.listdir(train_dir):
        if filename.startswith("train") and filename.endswith(".py"):
            module_name = filename[:-3]
            module = importlib.import_module(module_name)
            
            train_class = None
            for name, obj in vars(module).items():
                if name.lower().startswith("train") and isinstance(obj, type):
                    train_class = obj
                    break
            
            if train_class:
                print(f"Running {module_name}.{train_class.__name__}()")
                instance = train_class(batch_size=batch_size, epochs=epochs, model_name=model_name, models=models)

                for method_name in dir(instance):
                    if method_name.startswith("train"):
                        method = getattr(instance, method_name)
                        if callable(method):
                            print(f"Running {model_name}.{module_name}.{train_class.__name__}.{method_name}()")
                            method()
            else:
                print(f"{module_name} does not have a class that starts with `Train`")
