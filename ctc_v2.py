import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pickle
import string
import time
from contextlib import contextmanager

#torch.autograd.set_detect_anomaly(True)
# Profiler context manager
@contextmanager
def profile_section(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'[{name}] execution time: {end_time - start_time:.6f} seconds')

# Include both lowercase and uppercase letters
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation  +" " 

# Create char_map and reverse mapping
#char_map = {char: idx for idx, char in enumerate(characters)}
#rev_char_map = {idx: char for char, idx in char_map.items()}

# Custom Dataset
def modify_string(s):
    if not s:
        return s  # Handle empty string case
    
    result = []
    i = 0
    while i < len(s):
        # Start of a potential sequence
        char = s[i]
        sequence_start = i
        
        # Find the length of the sequence of identical characters
        while i < len(s) and s[i] == char:
            i += 1
        sequence_length = i - sequence_start
        
        if sequence_length > 1:
            # Modify every other character in the sequence to be a dollar sign
            for j in range(sequence_length):
                if j % 2 == 0:
                    result.append(char)
                else:
                    result.append('$')
        else:
            # Single character sequence, add the character itself
            result.append(char)
            
    return ''.join(result)
class StrokeDataset(Dataset):
    def __init__(self, x, y, char_map):
        self.x = x
        self.y = y
        self.char_map = char_map
        self.rev_char_map = {v: k for k, v in char_map.items()}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        #input_sequence = [
        #    [[float(val) for val in point] for point in stroke]  # Process each point in the stroke
        #    for stroke in self.x[idx]  # Iterate over strokes
        #]
        #flattened_input_sequence = [point for stroke in input_sequence for point in stroke]
        input_sequence = self.x[idx]
        flattened_input_sequence = [np.ravel(stroke) for stroke in input_sequence]
        flattened_input_sequence = [[float(val) for val in stroke] for stroke in flattened_input_sequence]
        target_sequence = [self.char_map[char] for char in self.y[idx]] 
        #print(self.y[idx])
        #target_sequence = [self.char_map[char] for char in modify_string(self.y[idx])]
        return torch.tensor(flattened_input_sequence, dtype=torch.float32), torch.tensor(target_sequence, dtype=torch.long)

# Collate function for padding
def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    target_lengths = torch.tensor([len(seq) for seq in targets], dtype=torch.long)

    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    concatenated_targets = torch.cat(targets)

    return padded_inputs, input_lengths, concatenated_targets, target_lengths

# Model
class StrokeCTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(StrokeCTCModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.fc(outputs)
        return outputs
"""# Collate function for padding
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
        return outputs"""
# Training

def smooth_logits(logits, smooth_factor, num_classes):
    """
    Apply label smoothing directly to logits.
    
    :param logits: Tensor of shape [time, batch, num_classes+1] (logits before log softmax)
    :param smooth_factor: Smoothing factor
    :param num_classes: Number of classes (excluding the blank token)
    :return: Smoothed logits
    """
    confidence_value = 1.0 - smooth_factor
    smoothing_value = smooth_factor / num_classes
    # Apply inverse smoothing to logits (away from center logits)
    logits = logits * confidence_value + smoothing_value
    return logits
def train_model(model, dataloader, optimizer, criterion, num_epochs,smooth_factor=0.1):
    lossstored = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        i = 0
        fails = 0
        for inputs, input_lengths, targets, target_lengths in dataloader:
            # print(i)
            optimizer.zero_grad()
            # Forward pass
            logits = model(inputs, input_lengths)
            logits = logits.log_softmax(2)
            logits = logits.permute(1, 0, 2)
            epsilon = 0.1
            num_classes = logits.size(-1)

            # Smooth targets: assuming targets in the format [batch_size, seq_length]
            smoothed_logits = smooth_logits(logits, smooth_factor, num_classes)

            # CTC Loss
            loss = criterion(smoothed_logits, targets, input_lengths, target_lengths)
            """if torch.isnan(loss).any() or torch.isinf(loss).any():
                #print(f"Iteration {i}: loss contains NaN or inf values")
                #print(f"input lengths - target lengths")
                #print(input_lengths - target_lengths)
                print('b', end="", flush=True)
                fails +=1
                continue
            else:"""
            print('a', end="", flush=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # print(loss.item())
            i+=1
        print()
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(dataloader)-fails)} Fails: { fails}")
        lossstored.append(total_loss/(len(dataloader)-fails))
        if epoch%10==9 and epoch>0:
            save_path = 'train_model_epoch_'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / (len(dataloader)-fails),
            }, f'{save_path}{epoch + 1}.pt')
    return lossstored
"""# Training
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
                print(loss)

            with profile_section('Backward and Step'):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            print(f"Batch {i} ended")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} finished, Loss: {total_loss / len(dataloader)}, Epoch Time: {epoch_end_time - epoch_start_time:.6f} seconds")
"""
if __name__ == "__main__":
    with profile_section('Loading data'):
        with open('y_train.pkl', 'rb') as file:
            y = pickle.load(file)
            print('y loaded')
        with open('x_train.pkl', 'rb') as file:
            x = pickle.load(file)
            print('x loaded')
    characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "

    # Create char_map and reverse mapping
    char_map = {char: idx for idx, char in enumerate(characters)}
    rev_char_map = {idx: char for char, idx in char_map.items()}
    
    # Dataset and DataLoader
    dataset = StrokeDataset(x, y, char_map)
    dataloader = DataLoader(dataset, batch_size=15, collate_fn=collate_fn)

    # Model, optimizer, and loss
    input_dim = 10  # x, y, time
    hidden_dim = 196
    num_classes = len(char_map) + 1  # Add 1 for the blank label in CTC
    model = StrokeCTCModel(input_dim, hidden_dim, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ctc_loss = nn.CTCLoss(blank=num_classes - 1)

    # Train
    losses = train_model(model, dataloader, optimizer, ctc_loss, num_epochs=50)
    with open('lossctc2.pkl', 'wb') as f:
        pickle.dump(losses, f)
        
    #print(model(x[3]))
    #print(y[3])
    torch.save(model, 'ctc_v2.pth')
