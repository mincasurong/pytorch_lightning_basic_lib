import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pytorch_lightning import LightningDataModule
from statsmodels.stats.outliers_influence import variance_inflation_factor

class TabularDataModule(LightningDataModule):
    def __init__(self, df, target_column, drop_columns=None, batch_size=32, test_size=0.2, random_state=42):
        super().__init__()
        self.df = df
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self):
        if self.drop_columns:
            self.df = self.df.drop(columns=self.drop_columns)
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column].fillna(self.df[column].mode()[0])
                self.df[column] = LabelEncoder().fit_transform(self.df[column])
            else:
                self.df[column].fillna(self.df[column].median())

    def setup(self, stage=None):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train, self.selected_features = self.feature_selection(X_train, y_train)
        X_test = X_test[self.selected_features]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_val = scaler.transform(X_val)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_val = y_val.values
        self.y_test = y_test.values

    def feature_selection(self, X, y):
        df = X.copy()
        df['target'] = y
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        target_correlation = correlation_matrix['target'].sort_values(ascending=False)
        threshold = 0.1
        selected_features = target_correlation[abs(target_correlation) > threshold].index.tolist()
        selected_features.remove('target')
        X_selected = df[selected_features]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_selected.columns
        vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
        high_vif_threshold = 5
        features_to_drop = vif_data[vif_data["VIF"] > high_vif_threshold]["feature"]
        selected_features = [f for f in selected_features if f not in features_to_drop]
        return X[selected_features], selected_features

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32),
                                      torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1))
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),
                                    torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1))
        return DataLoader(val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        test_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32),
                                     torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1))
        return DataLoader(test_dataset, batch_size=self.batch_size)
