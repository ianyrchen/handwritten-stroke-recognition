import pickle
import numpy as np

def load_data(file_path):
    """Loads data from a .pkl file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_data(data, file_path):
    """Saves data to a .pkl file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def normalize_strokes(training_examples):
    """Normalizes the points in each training example by the maximum height of a stroke."""
    normalized_examples = []
    for example in training_examples:
        # Find the maximum height in the example
        max_height = max([max([float(point[1]) for point in stroke]) for stroke in example])
        normalized_example = [[[float(x) / max_height, float(y) / max_height] for x, y, _ in stroke] for stroke in example]
        normalized_examples.append(normalized_example)
    return normalized_examples
def decompose_stroke(stroke, tolerance=0.9, min_sublist_length=4):
    """
    Decomposes a stroke into multiple smooth segments based on dot product continuity.

    Args:
        stroke (list): A list of (x, y) points.
        tolerance (float): Minimum dot product threshold for smoothness.
        min_sublist_length (int): Minimum size of contiguous subsequences to form a stroke.

    Returns:
        list: A list of decomposed strokes.
    """
    if len(stroke) < 2:
        return [stroke]  # A stroke with fewer than 2 points cannot be decomposed

    # Calculate finite differences
    differences = [np.array(stroke[i + 1]) - np.array(stroke[i]) for i in range(len(stroke) - 1)]

    # Normalize the difference vectors
    normalized_differences = [
        diff / np.linalg.norm(diff) if np.linalg.norm(diff) > 0 else np.zeros_like(diff)
        for diff in differences
    ]

    # Calculate the dot product of adjacent vectors
    dot_products = [
        np.dot(normalized_differences[i], normalized_differences[i + 1])
        for i in range(len(normalized_differences) - 1)
    ]

    # Identify contiguous subsequences where dot product exceeds the tolerance
    decomposed_strokes = []
    current_stroke = []
    for i, dp in enumerate(dot_products):
        if dp >= tolerance:
            current_stroke.append(stroke[i])
        else:
            if len(current_stroke) >= min_sublist_length:
                # Add the contiguous sublist as a new stroke
                current_stroke.append(stroke[i])  # Add the last point
                decomposed_strokes.append(current_stroke)
            current_stroke = []

    # Handle the last segment
    if len(current_stroke) >= min_sublist_length:
        current_stroke.append(stroke[-1])  # Add the last point of the original stroke
        decomposed_strokes.append(current_stroke)

    # If no subsequences satisfy the criteria, return the original stroke
    if not decomposed_strokes:
        return [stroke]

    return decomposed_strokes



def decompose_training_examples(training_examples):
    """Decomposes strokes in training examples into smooth strokes."""
    decomposed_examples = []
    for example in training_examples:
        decomposed_example = []
        for stroke in example:
            decomposed_example.extend(decompose_stroke(stroke))
        print(f"Increased the size by {len(example)/len(decomposed_example)}")
        decomposed_examples.append(decomposed_example)
    return decomposed_examples

def translate_strokes(training_examples):
    """Translates strokes in training examples relative to the last point of the previous stroke."""
    translated_examples = []
    for example in training_examples:
        translated_example = []
        origin = [0, 0]
        for stroke in example:
            translated_stroke = [[x - origin[0], y - origin[1]] for x, y in stroke]
            origin = translated_stroke[-1]
            translated_example.append(translated_stroke)
        translated_examples.append(translated_example)
    return translated_examples

def process_stroke(stroke):
    """Extracts [x0, y0, dx0, dy0, x1, y1, dx1, dy1] from a stroke."""
    if len(stroke) < 3:
        raise ValueError("A stroke must have at least two points.")
        return None
    x0, y0 = stroke[0]
    x1, y1 = stroke[-1]
    dx0, dy0 = np.array(stroke[1]) - np.array(stroke[0])
    dx0, dy0 = dx0 / (np.linalg.norm([dx0, dy0])+1e-8), dy0 / (np.linalg.norm([dx0, dy0])+ 1e-8)
    dx1, dy1 = np.array(stroke[-1]) - np.array(stroke[-2])
    dx1, dy1 = dx1 / (np.linalg.norm([dx1, dy1])+1e-8), dy1 / (np.linalg.norm([dx1, dy1])+1e-8)
    return [x0, y0, dx0, dy0, x1, y1, dx1, dy1]

def convert_to_final_format(training_examples):
    """Converts strokes in training examples to the final format."""
    final_examples = []
    for example in training_examples:
        final_example = [process_stroke(stroke) for stroke in example if len(stroke)>3]
        final_examples.append(final_example)
    return final_examples

def main(x_data_path, save_path):
    # Load the data
    training_examples = load_data(x_data_path)
    
    # Remove the time parameter and normalize strokes
    training_examples = normalize_strokes(training_examples)
    
    # Decompose strokes into smooth segments
    training_examples = decompose_training_examples(training_examples)
    
    # Translate strokes
    training_examples = translate_strokes(training_examples)
    
    # Convert to the final format
    training_examples = convert_to_final_format(training_examples)
    
    # Save the processed data
    save_data(training_examples, save_path)
    print(f"Processed data saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process training examples for handwriting.")
    parser.add_argument("x_data_path", type=str, help="Path to x_data.pkl")
    parser.add_argument("save_path", type=str, help="Path to save the processed data")

    args = parser.parse_args()
    main(args.x_data_path, args.save_path)

