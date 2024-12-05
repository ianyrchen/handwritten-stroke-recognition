import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import string
import pickle
import time 
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
                [[float(val) for val in point] for point in stroke]
                for stroke in self.x[idx]
            ]
        flattened_input_sequence = [point for stroke in input_sequence for point in stroke]
        
        target_sequence = [self.char_map[char] for char in self.y[idx]]
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

# Training
def train_model(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        total_loss = 0
        i = 0
        for inputs, input_lengths, targets, target_lengths in dataloader:
            start_time = time.time()  # Start timing
            
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs, input_lengths)
            logits = logits.log_softmax(2)

            # CTC Loss
            loss = criterion(logits.permute(1, 0, 2), targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_time = time.time() - start_time

            print(f"Batch {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f} - Time: {batch_time:.4f} seconds")
            i+=1

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

        save_path = 'train_model_epoch_'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
        }, f'{save_path}{epoch + 1}.pt')

# Example Usage
if __name__ == "__main__":
    # Data
    with open('x_data.pkl', 'rb') as file:
        x = pickle.load(file)
    with open('y_char_data.pkl', 'rb') as file:
        y = pickle.load(file)
    print("xy loaded")
    
    characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "

    # Create char_map and reverse mapping
    char_map = {char: idx for idx, char in enumerate(characters)}
    rev_char_map = {idx: char for char, idx in char_map.items()}
    
    # Dataset and DataLoader
    dataset = StrokeDataset(x, y, char_map)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Model, optimizer, and loss
    input_dim = 3  # x, y, time
    hidden_dim = 128
    num_classes = len(char_map) + 1  # Add 1 for the blank label in CTC
    model = StrokeCTCModel(input_dim, hidden_dim, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    ctc_loss = nn.CTCLoss(blank=num_classes - 1)

    # Train
    train_model(model, dataloader, optimizer, ctc_loss, num_epochs=10)


    torch.save(model, 'ctc_model.pth')

