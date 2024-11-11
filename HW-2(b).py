import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt

#Define the L2-regularized regression function
def l2_regularized_regression(X_train, y_train, X_test, y_test, lambdas):
    train_mse = []
    test_mse = []
    
    for l in lambdas:
        # Identity matrix (size based on number of features in X)
        n_features = X_train.shape[1]
        I = np.eye(n_features)
        
        # Compute the coefficients using Ridge regression formula
        w = np.linalg.inv(X_train.T.dot(X_train) + l * I).dot(X_train.T).dot(y_train)
        
        # Predict on both training and test sets
        y_train_pred = X_train.dot(w)
        y_test_pred = X_test.dot(w)
        
        # Compute the MSE for training and test sets
        train_mse.append(np.mean((y_train - y_train_pred) ** 2))
        test_mse.append(np.mean((y_test - y_test_pred) ** 2))
    
    return train_mse, test_mse

def main():
    # Get the path to the current script
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Use a relative path to the data folder
    train_files = [
        os.path.join(base_path, "data", "train-100-100.csv"),
        os.path.join(base_path, "data", "train-50(1000)-100.csv"),
        os.path.join(base_path, "data", "train-100(1000)-100.csv")
    ]

    test_files = [
        os.path.join(base_path, "data", "test-100-100.csv"),
        os.path.join(base_path, "data", "test-1000-100.csv"),
        os.path.join(base_path, "data", "test-1000-100.csv")
    ]
 
    lambdas = np.arange(1, 151, 1)

    # Create subplots, 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Loop through 3 datasets
    for i, (train_file, test_file) in enumerate(zip(train_files, test_files)):
        # Load the data
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Assume the last column in the CSV file is the label y, and the remaining columns are the features X
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Compute MSE for each λ
        train_mse, test_mse = l2_regularized_regression(X_train, y_train, X_test, y_test, lambdas)

        # Plot MSE vs Lambda on the subplot
        ax = axes[i]  
        ax.plot(lambdas, train_mse, label='Train MSE')
        ax.plot(lambdas, test_mse, label='Test MSE')
        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('MSE')
        ax.set_title(f'Dataset {i+1}')
        ax.legend()
        ax.grid()

    # Adjust subplot layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
