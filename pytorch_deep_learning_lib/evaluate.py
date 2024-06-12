import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, data_module):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data_module.X_test, dtype=torch.float32).to(model.device)
        y_hat = model(x).cpu().numpy()
        y = torch.tensor(data_module.y_test, dtype=torch.float32).view(-1, 1).cpu().numpy()
        mse = mean_squared_error(y, y_hat)
        mae = mean_absolute_error(y, y_hat)
        r2 = r2_score(y, y_hat)
        print(f'Mean Squared Error: {mse:.2f}')
        print(f'Mean Absolute Error: {mae:.2f}')
        print(f'R^2 Score: {r2:.2f}')
        plt.figure(figsize=(12, 5))
        plt.scatter(y, y_hat, alpha=0.7)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r')
        plt.tight_layout()
        plt.show()
