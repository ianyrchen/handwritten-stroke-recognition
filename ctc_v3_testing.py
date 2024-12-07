import torch
import string
from torch.nn.utils.rnn import pad_sequence
from ctc_v3 import StrokeCTCModel
import pickle

checkpoint_path = 'train_model_epoch_6.pt'
checkpoint = torch.load(checkpoint_path)

input_dim = 3  # x, y, time
hidden_dim = 128
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "
num_classes = len(characters) + 1  # +1 for the blank label in CTC
char_map = {char: idx for idx, char in enumerate(characters)}
rev_char_map = {idx: char for char, idx in char_map.items()}

model = StrokeCTCModel(input_dim, hidden_dim, num_classes)

# Load saved weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() 



with open('x_data.pkl', 'rb') as file:
    x = pickle.load(file)
with open('y_char_data.pkl', 'rb') as file:
    y = pickle.load(file)
print("xy loaded")


idx = 0
example_input = [
                [[float(val) for val in point] for point in stroke]
                for stroke in x[idx]
            ]
flattened_example_input = [point for stroke in example_input for point in stroke]
example_tensor = torch.tensor(flattened_example_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
example_length = torch.tensor([len(flattened_example_input)], dtype=torch.long)


print("Example tensor shape:", example_tensor.shape)
print("Example length:", example_length)



# forward
with torch.no_grad():
    logits = model(example_tensor, example_length)
    logits = logits.log_softmax(2)  
    
print("Logits shape:", logits.shape)
print("Top logits for debugging:", logits[0, :5, :])  



predictions = logits.argmax(2).squeeze(0).cpu().numpy()  

print("Predictions:", predictions)


blank_idx = num_classes - 1
decoded_output = []
previous_idx = -1
for idx in predictions:
    if idx != previous_idx and idx != blank_idx:
        decoded_output.append(rev_char_map[idx])
    previous_idx = idx

decoded_string = ''.join(decoded_output)
print(f"Decoded string: {decoded_string}")
