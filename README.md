# PyTorch Lightning Basic Library

This repository provides a basic framework for deep learning tasks using PyTorch Lightning. It includes essential modules for data preparation, model definition, training, and evaluation.

## Contents

- **[data_module.py](pytorch_lightning_basic_lib/data_module.py)**: Handles data preparation and loading.
- **[model.py](pytorch_lightning_basic_lib/model.py)**: Contains the definition of the regression model.
- **[train.py](pytorch_lightning_basic_lib/train.py)**: Provides the training loop for the model.
- **[evaluate.py](pytorch_lightning_basic_lib/evaluate.py)**: Includes functions for model evaluation.

## Modules

### Data Module

1. **TabularDataModule**: A PyTorch Lightning DataModule for handling tabular data.

### Model

1. **RegressionModel**: A simple regression model defined using PyTorch.

### Training

1. **train_model**: Function to train the regression model using PyTorch Lightning's Trainer.

### Evaluation

1. **evaluate_model**: Function to evaluate the trained model's performance.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- PyTorch Lightning

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mincasurong/pytorch_lightning_basic_lib.git
    cd pytorch_lightning_basic_lib
    pip install -r requirements.txt
    ```

### Usage

1. Prepare your data using `TabularDataModule`.
2. Define and train your model using `RegressionModel` and `train_model`.
3. Evaluate your trained model using `evaluate_model`.

## Example

Here's a basic example to get you started:

```python
from my_deep_learning_lib.data_module import TabularDataModule
from my_deep_learning_lib.model import RegressionModel
from my_deep_learning_lib.train import train_model
from my_deep_learning_lib.evaluate import evaluate_model
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
data_module = TabularDataModule(df, target_column='your_target_column', drop_columns=['drop_column1', 'drop_column2'])
data_module.prepare_data()
data_module.setup()

# Define and train the model
model = RegressionModel(input_dim=len(data_module.selected_features))
trained_model = train_model(data_module, model)

# Evaluate the model
evaluate_model(trained_model, data_module)
```

### Reference
For more details on the implementation, please refer to the PyTorch Lightning documentation: [PyTorch Lightning Docs.](https://pytorch-lightning.readthedocs.io/en/stable/)

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additions.

### Acknowledgements
Special thanks to the PyTorch and PyTorch Lightning communities for their continuous support and resources.

### License
This project is licensed under the MIT License 
