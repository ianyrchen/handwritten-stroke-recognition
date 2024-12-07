import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TraceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TraceClassifier, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: padded tensor of shape (batch_size, seq_len, input_dim)
        _, hidden = self.rnn(x)  # hidden: (1, batch_size, hidden_dim)
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        out = self.fc(hidden)  # (batch_size, output_dim)
        return out


if __name__ == '__main__':

    dataset_data, dataset_labels = torch.load('character_recognition/dataset_data_labels.pth')

    INPUT_DIM = 2  # x, y coordinates (2 dimensions)
    HIDDEN_DIM = 128
    OUTPUT_DIM = 36  # 0-9 + a-z
    BATCH_SIZE = 32

    model = TraceClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_samples = len(dataset_data)
    for epoch in range(50):
        indices = torch.randperm(num_samples).tolist()

        epoch_loss = 0
        epoch_correct = 0  
        epoch_total = 0  

        for i in range(0, num_samples, BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            batch_data = [dataset_data[idx] for idx in batch_indices]
            batch_labels = [dataset_labels[idx] for idx in batch_indices]

            batch_data_tensors = [torch.tensor(data_point, dtype=torch.float32) for data_point in batch_data]
            #padding
            x_padded = nn.utils.rnn.pad_sequence(batch_data_tensors, batch_first=True)

            y_tensor = torch.tensor(batch_labels, dtype=torch.long)  

            # forward
            outputs = model(x_padded)  
            loss = criterion(outputs, y_tensor)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_tensor).sum().item()  
            epoch_correct += correct
            epoch_total += y_tensor.size(0)

        avg_loss = epoch_loss / (num_samples // BATCH_SIZE)
        avg_accuracy = (epoch_correct / epoch_total) * 100  

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

    torch.save(model.state_dict(), "character_recognition/model.pth")
