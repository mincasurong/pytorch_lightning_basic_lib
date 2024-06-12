#%%


from pytorch_deep_learning_lib.data_module import TabularDataModule
from pytorch_deep_learning_lib.model import RegressionModel
from pytorch_deep_learning_lib.train import train_model
from pytorch_deep_learning_lib.evaluate import evaluate_model
import pandas as pd

data_choice = 3

# Load the dataset based on choice
if data_choice == 1:
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHousVal'] = california.target
    target_column = 'MedHousVal'
    drop_columns = []

elif data_choice == 2:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
    df = pd.read_excel(url)
    df.columns = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                    'Overall_Height', 'Orientation', 'Glazing_Area', 
                    'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
    target_column = 'Heating_Load'
    drop_columns = []

elif data_choice == 3:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    df = pd.read_excel(url)
    df.columns = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 
                    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 
                    'Age', 'Concrete_Compressive_Strength']
    target_column = 'Concrete_Compressive_Strength'
    drop_columns = []


# Hyperparameters
max_epochs = 200
batch_size = 16
test_size = 0.2
hidden_dims = [128, 64, 32]
lr = 0.0001
dropout_rate = 0.0
patience = 10

# Load your dataset
data_module = TabularDataModule(df, target_column, drop_columns,
                                batch_size=batch_size, test_size=test_size
                                )
data_module.prepare_data()
data_module.setup()

# Define and train the model
model = RegressionModel(
    input_dim=len(data_module.selected_features),  
    hidden_dims=hidden_dims, 
    lr=lr, 
    dropout_rate=dropout_rate
    )
trained_model = train_model(data_module, model, max_epochs=max_epochs, patience=patience)

# Evaluate the model
evaluate_model(trained_model, data_module)
# %%
