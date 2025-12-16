import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import utils

os.environ["KERAS_BACKEND"] = "torch"
from keras.utils import to_categorical

train_dataset_path = r"./data/train_dataset"
gestures = sorted(os.listdir(train_dataset_path)) if os.path.exists(train_dataset_path) else []
model_folder_path = r"./model"

decimal_delta = 1e-6


def is_better_model(accuracy, best_evaluation_accuracy, loss_item, best_evaluation_loss):
    accuracy_difference = accuracy - best_evaluation_accuracy
    return accuracy_difference > decimal_delta or (
        0.0 < accuracy_difference <= decimal_delta and loss_item < best_evaluation_loss
    )


def main():
    X = np.load(f"{model_folder_path}/X_all.npy")
    y = np.load(f"{model_folder_path}/y_all.npy")

    # Convert the labels into a one-hot encoded format for classification
    # For example, if there are 3 classes, label 1 becomes [0, 1, 0], label 2 becomes [0, 0, 1], etc.
    y = to_categorical(y).astype(int)

    # Split the dataset into training and testing sets
    # - `X_train` and `y_train`: Input features and labels for training
    # - `X_test` and `y_test`: Input features and labels for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # save train and test files
    np.save(f"{model_folder_path}/X_train.npy", X_train)
    np.save(f"{model_folder_path}/X_test.npy", X_test)
    np.save(f"{model_folder_path}/y_train.npy", y_train)
    np.save(f"{model_folder_path}/y_test.npy", y_test)

    # split train to train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    np.save(f"{model_folder_path}/X_train.npy", X_train)
    np.save(f"{model_folder_path}/X_val.npy", X_val)
    np.save(f"{model_folder_path}/y_train.npy", y_train)
    np.save(f"{model_folder_path}/y_val.npy", y_val)

    # load checkpoint
    X_train = np.load(f"{model_folder_path}/X_train.npy")
    X_val = np.load(f"{model_folder_path}/X_val.npy")
    X_test = np.load(f"{model_folder_path}/X_test.npy")
    y_train = np.load(f"{model_folder_path}/y_train.npy")
    y_val = np.load(f"{model_folder_path}/y_val.npy")
    y_test = np.load(f"{model_folder_path}/y_test.npy")

    # Check if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert data to PyTorch tensors and move to the GPU
    X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float64).to(device)
    y_train = torch.tensor(y_train.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices
    y_test = torch.tensor(y_test.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices
    y_val = torch.tensor(y_val.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices

    input_size = 258
    hidden_size = 64
    num_classes = len(gestures)
    model = utils.CustomLSTM(input_size, hidden_size, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model on the GPU
    num_epochs = 500
    loss_history = []
    best_evaluation_accuracy = 0.0
    best_evaluation_loss = 1e9 + 7

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()  # Reset gradients
        y_pred = model(X_train)  # Forward pass
        loss = criterion(y_pred, y_train)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss = loss.item()  # Accumulate loss

        # save loss history
        loss_history.append([epoch, epoch_loss])

        # evaluate model using 'val' set & save if better
        with torch.no_grad():
            model.eval()
            y_hat = model(X_val)
            loss = criterion(y_hat, y_val)
            loss_item = loss.item()

            accuracy = (y_hat.argmax(dim=1) == y_val).float().mean()
            if is_better_model(accuracy, best_evaluation_accuracy, loss_item, best_evaluation_loss):
                # save best model so far
                file_path = f"{model_folder_path}/trained_model.pt"
                torch.save(model.state_dict(), file_path)
                print(
                    f"Epoch [{epoch}/{num_epochs}], best model so far saved with accuracy of {accuracy} and loss of {loss_item} (measured with 'val' set)"
                )

                # update best values
                best_evaluation_accuracy = accuracy
                best_evaluation_loss = loss_item

    # Test the model using 'test' set
    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
        test_loss = criterion(y_hat, y_test)

        # calculate performance metrics: accuracy
        accuracy = (y_hat.argmax(dim=1) == y_test).float().mean()
        print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}")

        # convert y_test to numpy for performance metrics in scikit-learn
        y_test = y_test.detach().cpu().numpy()

        # save y_hat to disk
        y_hat = y_hat.argmax(dim=1).detach().cpu().numpy()
        np.save(f"{model_folder_path}/y_hat.npy", y_hat)

        # calculate performance metrics: f1 score
        f1_scr = f1_score(y_test, y_hat, average="macro")
        print(f"F1 Score (macro): {f1_scr}")

        # calculate performance metrics: confusion matrix
        conf_matrix = confusion_matrix(y_test, y_hat)
        print("Confusion Matrix:")
        print(conf_matrix)

    # Save loss history
    loss_history = np.array(loss_history)
    np.save(f"{model_folder_path}/loss_history.npy", loss_history)

    pass


if __name__ == "__main__":
    main()
