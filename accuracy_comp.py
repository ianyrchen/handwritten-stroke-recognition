import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import string
from ctc_v2 import StrokeDataset, collate_fn, StrokeCTCModel

# Functions for decoding
def greedy_decoder(logits, rev_char_map):
    """
    Perform greedy decoding on logits.
    Args:
        logits: Logits from the model (T, B, C)
        rev_char_map: Reverse character map {index: char}
    Returns:
        List of decoded strings
    """
    predictions = torch.argmax(logits, dim=-1)  # Get the index with the highest probability
    sequences = []
    for batch in predictions.permute(1, 0):  # Iterate over batch
        decoded = []
        prev_char = None
        for char_idx in batch:
            if char_idx != prev_char and char_idx != len(rev_char_map):  # Avoid consecutive repeats and blanks
                decoded.append(rev_char_map[char_idx.item()])
            prev_char = char_idx
        sequences.append(''.join(decoded))
    return sequences

# Custom Levenshtein distance function
def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    Args:
        s1: First string
        s2: Second string
    Returns:
        Distance (int)
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Custom WER calculation
def calculate_wer(target, prediction):
    """
    Calculate Word Error Rate (WER).
    Args:
        target: Reference sentence
        prediction: Hypothesized sentence
    Returns:
        WER (float)
    """
    target_words = target.split()
    prediction_words = prediction.split()
    distance = levenshtein_distance(target_words, prediction_words)
    return distance / len(target_words) if target_words else 0.0

# Load dataset
with open('ymod.pkl', 'rb') as file:
    y = pickle.load(file)
with open('x_bezier_data.pkl', 'rb') as file:
    x = pickle.load(file)

characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "
char_map = {char: idx for idx, char in enumerate(characters)}
rev_char_map = {idx: char for char, idx in char_map.items()}

dataset = StrokeDataset(x, y, char_map)
dataloader = DataLoader(dataset, batch_size=15, collate_fn=collate_fn)



# Custom Normalized Levenshtein Accuracy calculation
def normalized_levenshtein_accuracy(target, prediction):
    """
    Calculate the Normalized Levenshtein Accuracy.
    Args:
        target: Reference sentence
        prediction: Hypothesized sentence
    Returns:
        Accuracy (float)
    """
    distance = levenshtein_distance(target, prediction)
    max_length = max(len(target), len(prediction))
    return 1 - (distance / max_length) if max_length > 0 else 1.0

# Modified evaluation function to include Normalized Levenshtein Accuracy
def evaluate_model(model, dataloader, rev_char_map):
    model.eval()
    total_edit_distance = 0
    total_chars = 0
    total_wer = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths in dataloader:
            logits = model(inputs, input_lengths)
            predictions = logits.log_softmax(2).detach()
            decoded_sequences = greedy_decoder(predictions, rev_char_map)

            target_sequences = []
            idx = 0
            for length in target_lengths:
                target_sequence = targets[idx:idx + length].tolist()
                target_sequences.append(''.join(rev_char_map[char] for char in target_sequence))
                idx += length

            # Calculate edit distance, WER, and Normalized Levenshtein Accuracy
            for pred, target in zip(decoded_sequences, target_sequences):
                total_edit_distance += levenshtein_distance(pred, target)
                total_chars += len(target)
                total_wer += calculate_wer(target, pred)
                total_accuracy += normalized_levenshtein_accuracy(target, pred)
                total_samples += 1

    avg_cer = total_edit_distance / total_chars  # Character Error Rate
    avg_wer = total_wer / total_samples         # Word Error Rate
    avg_accuracy = total_accuracy / total_samples  # Normalized Levenshtein Accuracy
    return avg_cer, avg_wer, avg_accuracy

# Compare saved models
saved_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = {}

for epoch in saved_epochs:
    save_path = f'train_model_epoch_{epoch}.pt'
    checkpoint = torch.load(save_path)
    model = StrokeCTCModel(input_dim=10, hidden_dim=196, num_classes=len(char_map) + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')  # Ensure model is on CPU for evaluation

    print(f"Evaluating model saved at epoch {epoch}...")
    avg_cer, avg_wer, avg_accuracy = evaluate_model(model, dataloader, rev_char_map)
    results[epoch] = {'CER': avg_cer, 'WER': avg_wer, 'Accuracy': avg_accuracy}

# Display results
print("\nModel Accuracy Comparison:")
for epoch, metrics in results.items():
    print(f"Epoch {epoch}: CER = {metrics['CER']:.4f}, WER = {metrics['WER']:.4f}, Accuracy = {metrics['Accuracy']:.4f}")
