import torch
import numpy as np
from ctc_v2 import StrokeCTCModel
import string
import pickle
from Levenshtein import distance as levenshtein_distance

def load_data(file_path):
    """Loads data from a .pkl file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " " 
char_map = {char: idx for idx, char in enumerate(characters)}
rev_char_map = {idx: char for char, idx in char_map.items()}
num_classes = len(characters) + 1  # +1 for the blank label in CTC
 
def load_model(model_path):
    """Loads a PyTorch model from a file."""
    checkpoint = torch.load(model_path)
    # Initialize the model with the same dimensions as during training
    input_dim = 10  # x, y, time
    hidden_dim = 196
    model = StrokeCTCModel(input_dim, hidden_dim, num_classes)

    # Load the saved model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def evaluate_model(model, x_val, y_val):
    """
    Evaluates the model's accuracy using Levenshtein distance.
    Returns the average Levenshtein distance across all validation examples.
    """
    model.eval()  # Set model to evaluation mode
    total_distance = 0
    num_samples = len(x_val)

    with torch.no_grad():
        for x, y_true in zip(x_val, y_val):
            # Convert input to a tensor if needed
            flattened_input_sequence = [np.ravel(stroke) for stroke in x]
            flattened_example_input = [[float(val) for val in stroke] for stroke in flattened_input_sequence]
            #flattened_example_input = [point for stroke in example_input for point in stroke]
            example_tensor = torch.tensor(flattened_example_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            example_length = torch.tensor([len(flattened_example_input)], dtype=torch.long)
            logits = model(example_tensor, example_length)
            logits = logits.log_softmax(2)
            predictions = logits.argmax(2).squeeze(0).cpu().numpy()
            blank_idx = num_classes - 1
            decoded_output = []
            previous_idx = -1
            for uidx in predictions:
                if uidx != blank_idx:
                    decoded_output.append(rev_char_map[uidx])
                    # previous_idx = uidx
            decoded_string = ''.join(decoded_output)


            #x_tensor = torch.tensor(x).unsqueeze(0)  # Assuming input is a single instance
            #y_pred = model(x_tensor)

            # Convert predictions to strings if necessary
            #y_pred_str = y_pred.argmax(dim=-1).tolist() if y_pred.dim() > 1 else y_pred.tolist()
            #y_true_str = y_true if isinstance(y_true, str) else str(y_true)

            # Compute Levenshtein distance
            total_distance += levenshtein_distance(str(y_true), str(decoded_string)) / len(y_true)

    # Return the average Levenshtein distance
    return total_distance / num_samples

def main(model_path, x_val_path, y_val_path):
    # Load the validation datasets
    x_val = load_data(x_val_path)
    y_val = load_data(y_val_path)

    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    avg_distance = 1-evaluate_model(model, x_val, y_val)

    print(f"Average Levenshtein distance based accuracy on validation set: {avg_distance:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model accuracy using Levenshtein distance.")
    parser.add_argument("model_path", type=str, help="Path to the saved model (model.pt)")
    parser.add_argument("x_val_path", type=str, help="Path to x_val.pkl")
    parser.add_argument("y_val_path", type=str, help="Path to y_val.pkl")

    args = parser.parse_args()

    main(args.model_path, args.x_val_path, args.y_val_path)

