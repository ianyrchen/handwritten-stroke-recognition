import torch
import string
import threading
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ctc_v2 import StrokeCTCModel
import pickle

# Load the model checkpoint
checkpoint_path = 'ntrain_model_epoch_30.pt'
checkpoint = torch.load(checkpoint_path)

# Initialize the model with the same dimensions as during training
input_dim = 20  # x, y, time
hidden_dim = 196
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation +" "
num_classes = len(characters) + 1  # +1 for the blank label in CTC
char_map = {char: idx for idx, char in enumerate(characters)}
rev_char_map = {idx: char for char, idx in char_map.items()}
rev_char_map[95] = " "

model = StrokeCTCModel(input_dim, hidden_dim, num_classes)

# Load the saved model weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode




def mainloop():
    threading.Timer(1.0, mainloop).start()
    with open('bezwb.pkl', 'rb') as file:
        x = pickle.load(file)
    #with open('x_bezier_data.pkl', 'rb') as file:
    #    xd = pickle.load(file)
    #with open('y_char_data.pkl', 'rb') as file:
    #    y = pickle.load(file)
    #print("xy loaded")
    
    # Flatten strokes into a single sequence
    idx = 0
    #example_input = [
    #                [[float(val) for val in point] for point in stroke]
    #                for stroke in x[idx]
    #            ]
    example_input = x[idx]
    print(example_input)
    #example_input = x[idx][40:100] + x[idx+1][30:100] + x[idx+2][20:300]
    flattened_input_sequence = [np.ravel(stroke) for stroke in example_input]
    flattened_example_input = [[float(val) for val in stroke] for stroke in flattened_input_sequence]
    
    
    #flattened_example_input = [point for stroke in example_input for point in stroke]
    example_tensor = torch.tensor(flattened_example_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    example_length = torch.tensor([len(flattened_example_input)], dtype=torch.long)
    
    
    print("Example tensor shape:", example_tensor.shape)
    print("Example length:", example_length)
    
    
    
    # Forward pass
    with torch.no_grad():
        logits = model(example_tensor, example_length)
        logits = logits.log_softmax(2)  # Apply log softmax (if not done inside the model)
    
    # print("Logits shape:", logits.shape)
    # print("Top logits for debugging:", logits[0, :5, :])  # Inspect the first few timesteps
    
    
    # Decode the output using greedy decoding
    predictions = logits.argmax(2).squeeze(0).cpu().numpy()  # Get the most likely class index at each time step
    
    # print("Predictions:", predictions)
    
    
    # Convert predictions to string, ignoring repeated characters and the blank label
    blank_idx = num_classes - 1
    decoded_output = []
    previous_idx = -1
    for uidx in predictions:
        #if uidx != previous_idx and uidx != blank_idx:
        if uidx != blank_idx:
            decoded_output.append(rev_char_map[uidx])
        previous_idx = uidx
    
    # Join characters to form the final string
    decoded_string = ''.join(decoded_output)
    #decoded_string = [decoded_string[i] if i==0 or decoded_string[i]!="$" else decoded_string[i-1] for i in range(len(decoded_string))]
    print(f"Decoded string: {''.join(decoded_string)}")
    #print(f"Actual string: {y[idx]}, {y[idx+1], y[idx+2]}")
mainloop()
