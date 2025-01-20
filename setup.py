from setuptools import setup, find_packages 
import os 

with open(os.path.join(os.path.dirname(__file__), 'README_pypi.md'), 'r', encoding='utf-8') as fh: 
    long_description = fh.read()

setup(
    name="neurostage",  
    version="0.2",
    packages=find_packages(include=['templates', 'templates.*']),
    py_modules=["main", "__main__"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0,<2.0.0",          # Compatible con NumPy 1.x
        "tensorflow>=2.10.0,<3.0.0",    # Compatible con TensorFlow 2.x
        "opencv-python>=4.5.0,<5.0.0",   # Compatible con OpenCV 4.x
        "tensorboard>=2.10.0,<3.0.0"    # Compatible con TensorFlow 2.x y TensorBoard 2.x
    ],
    entry_points={
        "console_scripts": [
            "stage=main:main", 
        ],
    },
    author='Catalina Delgado', 
    author_email='catalina08delgado@gmail.com', 
    description='A framework for managing deep learning projects', 
    long_description=long_description,
    long_description_content_type='text/markdown', 
    url='https://github.com/catalina-delgado/NeuroStage', 
    classifiers=[ 'Programming Language :: Python :: 3', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent', ], 
    keywords=['training', 'deepLearning', 'framework'],
    python_requires='>=3.6',
)
