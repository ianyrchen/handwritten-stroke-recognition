import pickle
import random
import argparse

def load_data(file_path):
    """Loads data from a .pkl file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_data(data, file_path):
    """Saves data to a .pkl file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def split_data(x_data, y_data, seed, r):
    """Splits x_data and y_data into training and validation sets."""
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    # Set the random seed
    random.seed(seed)

    # Create a list of indices and shuffle them
    indices = list(range(len(x_data)))
    random.shuffle(indices)

    # Calculate the split point
    split_point = int(len(indices) * r)

    # Split the indices into training and validation sets
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Create the training and validation datasets
    x_train = [x_data[i] for i in train_indices]
    y_train = [y_data[i] for i in train_indices]
    x_val = [x_data[i] for i in val_indices]
    y_val = [y_data[i] for i in val_indices]

    return x_train, y_train, x_val, y_val

def main(x_data_path, y_data_path, seed, r):
    # Load the datasets
    x_data = load_data(x_data_path)
    y_data = load_data(y_data_path)

    # Split the datasets
    x_train, y_train, x_val, y_val = split_data(x_data, y_data, seed, r)

    # Save the split datasets
    save_data(x_train, 'x_train.pkl')
    save_data(y_train, 'y_train.pkl')
    save_data(x_val, 'x_val.pkl')
    save_data(y_val, 'y_val.pkl')

    print("Datasets have been split and saved:")
    print("x_train.pkl, y_train.pkl, x_val.pkl, y_val.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split datasets into training and validation sets.")
    parser.add_argument("x_data_path", type=str, help="Path to x_data.pkl")
    parser.add_argument("y_data_path", type=str, help="Path to y_data.pkl")
    parser.add_argument("seed", type=int, help="Random seed for splitting")
    parser.add_argument("r", type=float, help="Proportion of data for training (e.g., 0.9 for 90%)")

    args = parser.parse_args()

    main(args.x_data_path, args.y_data_path, args.seed, args.r)

