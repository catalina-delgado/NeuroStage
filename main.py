import argparse
import os
import sys
import importlib
import shutil
import traceback
import templates.utils as utils

class CustomArgumentParser(argparse.ArgumentParser):
    """Customize the parser's error handling to provide more user-friendly messages"""
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n")
        sys.stderr.write("You must specify a project name when using the 'startproject' command.\n")
        self.print_help()
        sys.exit(2)


class InitMain:
    """Primary class for managing NeuroStage commands and operations"""

    def __init__(self):
        pass

    def execute(self):
        
        parser = CustomArgumentParser(
            description="NeuroStage: A framework for testing and training deep learning models."
        )
        subparsers = parser.add_subparsers(dest='command')

        # Subcommand 'startproject'
        startproject_parser = subparsers.add_parser('startproject', help='Start a New Project')
        startproject_parser.add_argument('name', type=str, help="New Project Name")

        # Subcommand 'run'
        run_parser = subparsers.add_parser('run', help="Run the current project")
        run_parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
        run_parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")

        # Argumento independiente: listar modelos
        parser.add_argument('--list', action='store_true', help="List the available models")
        parser.add_argument('--help_train', action='store_true', help="Show usage examples")

        # Parsear los argumentos
        args = parser.parse_args()

        # Procesar los comandos
        if args.command == "startproject":
            self.create_project(args.name)
        elif args.command == "run":
            self.run_project(args)
        elif args.list:
            self.list_models()
        elif args.help_train:
            self.show_help()
        else:
            parser.print_help()
            sys.exit(1)

    def create_project(self, project_name):
        os.makedirs(project_name, exist_ok=True)

        subdirs = [
            "src",
            "src/layers",
            "src/models",
            "experiments",
            "training"
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join(project_name, subdir), exist_ok=True)

        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
      
        # Crear archivos iniciales
        files_to_create = {
            # Archivos que se deben copiar
            os.path.join(templates_dir, "__init__.py"): os.path.join(project_name, "__init__.py"),
            os.path.join(templates_dir, "functions.py"): os.path.join(project_name, "functions.py"),
            os.path.join(templates_dir, "imports.py"): os.path.join(project_name, "imports.py"),
            
            # Archivos que se deben crear con contenido predeterminado
            os.path.join(project_name, "config.py"): f"PROJECT_NAME = '{project_name}'\n",
            os.path.join(project_name, "src/__init__.py"): '',
            os.path.join(project_name, "src/layers/__init__.py"): "# Layer initialization",
            os.path.join(project_name, "src/models/__init__.py"): "# Models initialization",
            os.path.join(project_name, "experiments/__init__.py"): "# Experiments initialization",
            os.path.join(project_name, "training/__init__.py"): "# Training initialization",
        }

        for src, dest in files_to_create.items():
            
            if 'templates' in src:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w") as f:
                    f.write("")
                shutil.copy(src, dest)  # Copiar archivo
            else:
                # Crear el archivo con el contenido predeterminado
                os.makedirs(os.path.dirname(src), exist_ok=True)  # Crear directorios si no existen
                with open(src, "w") as f:
                    f.write(dest if isinstance(dest, str) else "")

        print(f"Project '{project_name}' successfully created with the basic structure")
        
    def run_project(self, args):
        try:
            config_path = os.path.join(os.getcwd(), 'config.py')
            if not os.path.exists(config_path):
                print("No project found in the current directory")
                sys.exit(1)
                
            current_working_dir = os.getcwd()
            project_dir = os.path.join(current_working_dir)

            if project_dir not in sys.path:
                sys.path.insert(0, project_dir)

            config = importlib.import_module("config")
            utils.run_train_scripts(args.batch_size, args.epochs, config.PROJECT_NAME, project_dir)
            
        except ModuleNotFoundError as e:
            print("No valid project found in the current directory")
            print(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            print("An unexpected error occurred:")
            print(traceback.format_exc())
            sys.exit(1)

    def list_models(self):
        models_dir = os.path.join(os.getcwd(), 'src/models')
        if not os.path.exists(models_dir):
            print("The 'src/models' folder does not exist")
            return

        print("Available models:")
        for filename in os.listdir(models_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                print(f" - {filename[:-3]}")

    def show_help(self): 
        """Shows usage examples for training models.""" 
        print( """ 
              Usage example: 
              1. Train a specific model: 
              stage run --batch_size 32 --epochs 100
              
              2. List the available models: 
              stage --list 
              
              3. Create a new project: 
              stage startproject my_project 
              """ )


def main():
    InitMain().execute()

