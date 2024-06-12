from setuptools import setup, find_packages

setup(
    name='my_deep_learning_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'torch',
        'torchmetrics',
        'sklearn',
        'pytorch-lightning',
        'statsmodels',
    ],
    description='A deep learning library using PyTorch Lightning',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/my_deep_learning_lib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
