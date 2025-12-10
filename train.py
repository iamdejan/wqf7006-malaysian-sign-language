import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

gestures = ["ribut", "nasi_lemak", "pandai", "panas", "baik", "bila", "tandas", "apa", "beli_2", "hari", "anak_lelaki", "panas_2", "beli", "hi", "marah", "boleh", "assalamualaikum", "apa_khabar", "tidur", "masalah", "abang", "polis", "perlahan_2", "perlahan", "saudara", "siapa", "bagaimana", "bahasa_isyarat", "baik_2", "bapa_saudara", "berapa", "hujan", "kakak", "keluarga", "mana", "payung", "perempuan", "lelaki", "curi", "berlari", "sampai", "mari", "pergi_2", "emak", "ada", "mohon", "kereta", "suka", "ayah", "main", "buang", "lemak", "minum", "bomba", "pukul", "buat", "bawa", "tanya", "anak_perempuan", "sejuk", "kacau", "ambil", "pensil", "emak_saudara", "teh_tarik", "berjalan", "sudah", "lupa", "jahat", "tolong", "bola", "bas", "masa", "baca", "kesakitan", "pandai_2", "jumpa", "dapat", "arah", "teksi", "dari", "jam", "sekolah", "jangan", "nasi", "makan", "bapa", "pergi", "pinjam", "pen"]

# define the folder path and the file names
# model_folder_path = "/content/drive/MyDrive/Colab Notebooks/[WQF7006] Computer Vision And Image Processing/Group Project/model"
model_folder_path = "./model"

X_train = np.load(f'{model_folder_path}/X_train.npy')
y_train = np.load(f'{model_folder_path}/y_train.npy')

X_test = np.load(f'{model_folder_path}/X_test.npy')
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
# device = torch.device("cpu")

# Convert data to PyTorch tensors and move to the GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices
y_test = torch.tensor(y_test.argmax(axis=1), dtype=torch.long).to(device)  # Convert to class indices

# Define your custom LSTM model and move to the GPU
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x

# Create empty lists to store loss and accuracy values
train_losses = []
test_losses = []
test_accuracies = []

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

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad() # Reset gradients
    outputs = model(X_train) # Forward pass
    loss = criterion(outputs, y_train) # Compute loss
    loss.backward() # Backward pass
    optimizer.step() # Update weights

    epoch_loss += loss.item() # Accumulate loss

    if (epoch + 1) % 10 == 0:
        # save loss history
        loss_history.append([epoch, epoch_loss])

        # Calculate average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # save model as checkpoint
    torch.save(model.state_dict(), get_model_filename(epoch=epoch))
    pass


# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    accuracy = (test_outputs.argmax(dim=1) == y_test).float().mean()
    print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")

# Save loss history
loss_history = np.array(loss_history)
np.save(f"{model_folder_path}/loss_history.npy", loss_history)
