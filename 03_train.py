import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "torch"
from keras.utils import to_categorical

train_dataset_path = r'./data/train_dataset'
gestures = sorted(os.listdir(train_dataset_path)) if os.path.exists(train_dataset_path) else []
model_folder_path = r"./model"

dropout = 0.2

# Define your custom LSTM model and move to the GPU
class ExtractLastTimeStep(nn.Module):
    def forward(self, x):
        # LSTM returns (output, (h_n, c_n))
        output, _ = x 
        # Extract the last time step: Shape (batch, seq, hidden) -> (batch, hidden)
        return output[:, -1, :]


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout, dtype=torch.float64).double(),

            # Custom layer to handle the output tuple and slice the last step
            ExtractLastTimeStep(),

            # Layer 1
            nn.Linear(hidden_size, 64, dtype=torch.float64),
            nn.BatchNorm1d(64, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(64, 128, dtype=torch.float64),
            nn.BatchNorm1d(128, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(128, 64, dtype=torch.float64),
            nn.BatchNorm1d(64, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 4
            nn.Linear(64, 32, dtype=torch.float64),
            nn.BatchNorm1d(32, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(dropout-0.1),

            # Layer 5
            nn.Linear(32, 32, dtype=torch.float64),
            # Batch norm here is optional, often skipped right before output
            nn.ReLU(),

            # Output Layer (No BN/Dropout/ReLU on the final logits)
            nn.Linear(32, num_classes, dtype=torch.float64),
        )

    def forward(self, x):
        return self.model(x)


def main():
    X = np.load(f'{model_folder_path}/X_all.npy')
    y = np.load(f'{model_folder_path}/y_all.npy')

    # Convert the labels into a one-hot encoded format for classification
    # For example, if there are 3 classes, label 1 becomes [0, 1, 0], label 2 becomes [0, 0, 1], etc.
    y = to_categorical(y).astype(int)

    # Split the dataset into training and testing sets
    # - `X_train` and `y_train`: Input features and labels for training
    # - `X_test` and `y_test`: Input features and labels for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # save train and test files
    np.save(f'{model_folder_path}/X_train.npy', X_train)
    np.save(f'{model_folder_path}/X_test.npy', X_test)
    np.save(f'{model_folder_path}/y_train.npy', y_train)
    np.save(f'{model_folder_path}/y_test.npy', y_test)

    # split train to train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    np.save(f'{model_folder_path}/X_train.npy', X_train)
    np.save(f'{model_folder_path}/X_val.npy', X_val)
    np.save(f'{model_folder_path}/y_train.npy', y_train)
    np.save(f'{model_folder_path}/y_val.npy', y_val)

    # load checkpoint
    X_train = np.load(f'{model_folder_path}/X_train.npy')
    X_val = np.load(f'{model_folder_path}/X_val.npy')
    X_test = np.load(f'{model_folder_path}/X_test.npy')
    y_train = np.load(f'{model_folder_path}/y_train.npy')
    y_val = np.load(f'{model_folder_path}/y_val.npy')
    y_test = np.load(f'{model_folder_path}/y_test.npy')

    # define model filename
    def get_model_filename(epoch: None | int = None):
        if epoch is None:
            return f"{model_folder_path}/trained_model.pt"
        else:
            return f"{model_folder_path}/trained_model_epoch_{epoch}.pt"

    model_filename = get_model_filename()

    # Check if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert data to PyTorch tensors and move to the GPU
    X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
    y_train = torch.tensor(y_train.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices
    y_test = torch.tensor(y_test.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices

    input_size = 258
    hidden_size = 64
    num_classes = len(gestures)
    model = CustomLSTM(input_size, hidden_size, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model on the GPU
    num_epochs = 500
    loss_history = []

    for epoch in range(1, num_epochs+1):
        model.train()
        optimizer.zero_grad() # Reset gradients
        outputs = model(X_train) # Forward pass
        loss = criterion(outputs, y_train) # Compute loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        epoch_loss = loss.item() # Accumulate loss

        # save loss history
        loss_history.append([epoch, epoch_loss])

        # save best model

        if epoch % 10 == 0:
            # Calculate average loss for the epoch
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # save model as checkpoint
            torch.save(model.state_dict(), get_model_filename(epoch=epoch))


    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        accuracy = (y_pred.argmax(dim=1) == y_test).float().mean()
        print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

        # save y_pred to disk
        y_pred = y_pred.detach().cpu().to_numpy()
        np.save(f'{model_folder_path}/y_pred.npy', y_pred)

    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

    # Save loss history
    loss_history = np.array(loss_history)
    np.save(f"{model_folder_path}/loss_history.npy", loss_history)

    pass


if __name__ == "__main__":
    main()

