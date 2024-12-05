import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pickle
import string
import time
from contextlib import contextmanager

# Profiler context manager
@contextmanager
def profile_section(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'[{name}] execution time: {end_time - start_time:.6f} seconds')

# Include both lowercase and uppercase letters
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "

# Create char_map and reverse mapping
char_map = {char: idx for idx, char in enumerate(characters)}
rev_char_map = {idx: char for char, idx in char_map.items()}

# Custom Dataset
class StrokeDataset(Dataset):
    def __init__(self, x, y, char_map):
        self.x = x
        self.y = y
        self.char_map = char_map
        self.rev_char_map = {v: k for k, v in char_map.items()}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_sequence = [
            [[float(val) for val in point] for point in stroke]  # Process each point in the stroke
            for stroke in self.x[idx]  # Iterate over strokes
        ]
        flattened_input_sequence = [point for stroke in input_sequence for point in stroke]
        target_sequence = [self.char_map[char] for char in self.y[idx]]
        return torch.tensor(flattened_input_sequence, dtype=torch.float32), torch.tensor(target_sequence, dtype=torch.long)

# Collate function for padding
def collate_fn(batch):
    with profile_section('Collate function'):
        inputs, targets = zip(*batch)
        input_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
        target_lengths = torch.tensor([len(seq) for seq in targets], dtype=torch.long)
        padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        concatenated_targets = torch.cat(targets)
    return padded_inputs, input_lengths, concatenated_targets, target_lengths

# New Model using Convolutions
class StrokeCNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(StrokeCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)  # Convert (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.transpose(1, 2)  # Convert back (batch_size, input_dim, seq_len) -> (batch_size, seq_len, hidden_dim)
        outputs = self.fc(x)
        return outputs

# Training
def train_model(model, dataloader, optimizer, criterion, num_epochs, scaler, device):
    print("Model training started")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting epoch {epoch + 1}")
        epoch_start_time = time.time()
        for i, (inputs, input_lengths, targets, target_lengths) in enumerate(dataloader):
            print(f"Batch {i} started")
            with profile_section('Data transfer'):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device)
                input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(inputs, input_lengths)
                logits = logits.log_softmax(2)
                loss = criterion(logits.permute(1, 0, 2), targets, input_lengths, target_lengths)

            with profile_section('Backward and Step'):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            print(f"Batch {i} ended")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} finished, Loss: {total_loss / len(dataloader)}, Epoch Time: {epoch_end_time - epoch_start_time:.6f} seconds")

if __name__ == "__main__":
    with profile_section('Loading data'):
        with open('y_data.pkl', 'rb') as file:
            y = pickle.load(file)
            print('y loaded')
        with open('x_data.pkl', 'rb') as file:
            x = pickle.load(file)
            print('x loaded')

    # Dataset and DataLoader
    dataset = StrokeDataset(x, y, char_map)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=4)

    # Model, optimizer, and loss
    input_dim = 3  # x, y, time
    hidden_dim = 128  # Increased from 64
    num_classes = len(char_map) + 1  # Add 1 for the blank label in CTC
    model = StrokeCNNModel(input_dim, hidden_dim, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    ctc_loss = nn.CTCLoss(blank=num_classes - 1)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # Train with profiling
    train_model(model, dataloader, optimizer, ctc_loss, num_epochs=10, scaler=scaler, device=device)

    # Save and load model
    torch.save(model.state_dict(), 'ctc_v2_model.pth')
    model.load_state_dict(torch.load('ctc_v2_model.pth'))