import pickle
import numpy as np
import threading
def load_data(file_path):
    """Loads data from a .pkl file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_data(data, file_path):
    """Saves data to a .pkl file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
import numpy as np

def adjust_points(points, num_samples=10):
    """
    Adjust the number of points to exactly num_samples by either downsampling or adding intermediate points.

    Args:
        points (list): A list of (x, y) tuples.
        num_samples (int): Desired number of points.

    Returns:
        list: Adjusted points with exactly num_samples points.
    """
    n = len(points)
    points = [(float(x), float(y)) for x, y,_ in points]
    if n == num_samples:
        return points

    if n > num_samples:
        # Downsample by selecting evenly spaced indices
        indices = np.linspace(0, n - 1, num_samples, dtype=int)
        return [points[i] for i in indices]

    # Add intermediate points to reach num_samples
    adjusted_points = points[:]
    while len(adjusted_points) < num_samples:
        new_points = []
        for i in range(len(adjusted_points) - 1):
            new_points.append(adjusted_points[i])
            # Insert intermediate point
            mid_x = (adjusted_points[i][0] + adjusted_points[i + 1][0]) / 2
            mid_y = (adjusted_points[i][1] + adjusted_points[i + 1][1]) / 2
            new_points.append((mid_x, mid_y))
        new_points.append(adjusted_points[-1])  # Add the last point
        adjusted_points = new_points

    # Downsample to exactly num_samples if we've added too many points
    indices = np.linspace(0, len(adjusted_points) - 1, num_samples, dtype=int)
    return np.array([adjusted_points[i] for i in indices])
def maximize_distance_with_local_search(points, k, neighborhood_size=3):
    """
    Selects k points from n+1 sorted points to maximize the sum of distances between adjacent points
    using a gradient ascent-inspired approach with local search optimization.

    Args:
        points (list): A list of (x, y) coordinates sorted in time order.
        k (int): The number of points to select.
        neighborhood_size (int): Number of neighbors to consider for swaps.

    Returns:
        list: Subset of k points maximizing the sum of distances between adjacent points.
    """
    n = len(points)
    if n==1:
        return [points[0]]*k 
    if k > n or k < 2:
        raise ValueError("Invalid value for k.")

    # Initialize with evenly spaced points
    selected_indices = np.linspace(0, n - 1, k, dtype=int).tolist()

    def compute_total_distance(indices):
        """Compute the total distance for the given indices."""
        return sum(
            np.sqrt(np.linalg.norm(points[indices[i + 1]] - points[indices[i]]))
            for i in range(len(indices) - 1))

    def get_neighbors(idx, n, size):
        """Return a range of indices near idx, bounded by 0 and n-1."""
        return range(max(0, idx - size), min(n, idx + size + 1))

    while True:
        improved = False
        current_distance = compute_total_distance(selected_indices)

        # Attempt swaps with local neighbors
        for i in range(k):
            current_idx = selected_indices[i]
            neighbors = get_neighbors(current_idx, n, neighborhood_size)

            for neighbor in neighbors:
                if neighbor in selected_indices:
                    continue  # Skip already-selected points

                # Create a new selection by swapping
                new_indices = selected_indices[:]
                new_indices[i] = neighbor
                new_indices.sort()

                new_distance = compute_total_distance(new_indices)
                if new_distance > current_distance:
                    selected_indices = new_indices
                    current_distance = new_distance
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break  # Exit if no improvement is found

    return [points[i] for i in selected_indices]
"""
def maximize_distance_sum(points, k):
    ""
    Select k points from points to maximize the sum of distances between adjacent points.

    Args:
        points (list): A list of (x, y) tuples, assumed to be ordered.
        k (int): The number of points to select.

    Returns:
        list: Subset of k points that maximize the sum of distances.
    ""
    n = len(points)
    if n==1:
        return [points[0]]*k
    if k > n and n!=1:
        raise ValueError("k cannot be greater than the number of points.")

    def distance(p1, p2):
        ""Calculate Euclidean distance between two points.""
        #return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        return np.linalg.norm(p2-p1)

    # Initialize DP table
    dp = [[float('-inf')] * k for _ in range(n)]
    prev = [[-1] * k for _ in range(n)]  # To reconstruct the solution

    # Base case: selecting the first point
    for i in range(n):
        dp[i][0] = 0  # Sum of distances is 0 when selecting 1 point

    # Fill DP table
    for j in range(1, k):  # Number of points selected
        for i in range(j, n):  # Current point
            for p in range(j - 1, i):  # Previous point
                current_distance = dp[p][j-1] + distance(points[p], points[i])
                if current_distance > dp[i][j]:
                    dp[i][j] = current_distance
                    prev[i][j] = p

    # Backtrack to find selected points
    selected_indices = []
    current = n - 1
    for j in range(k - 1, -1, -1):
        selected_indices.append(current)
        current = prev[current][j]

    selected_indices.reverse()
    return [points[i] for i in selected_indices]
"""

def normalize_strokes(training_examples):
    """Normalizes the points in each training example by the maximum height of a stroke."""
    normalized_examples = []
    for example in training_examples:
        # Find the maximum height in the example
        max_height = max([max([float(point[1]) for point in stroke]) for stroke in example])
        normalized_example = [[[float(x) / max_height, float(y) / max_height] for x, y in stroke] for stroke in example]
        normalized_examples.append(normalized_example)
    return normalized_examples
def decompose_stroke(stroke, tolerance=0.8, min_sublist_length=8):
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
        diff / (np.linalg.norm(diff) + 1e-8)  #if np.linalg.norm(diff) > 0 else np.zeros_like(diff)
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
        origin = np.array([0, 0])
        for stroke in example:
            translated_stroke = stroke - origin
            origin = translated_stroke[-1]
            translated_example.append(translated_stroke)
        translated_examples.append(translated_example)
    return translated_examples

def process_stroke(stroke,k):
    """Extracts [x0, y0, dx0, dy0, x1, y1, dx1, dy1] from a stroke."""
    if len(stroke) < 4:
        raise ValueError("A stroke must have at least two points.")
        return None
    n = len(stroke)
    ret = []
    for i in range(k):
        ret.extend(stroke[round(i*(n-1)/(k-1))])
    return ret
    x0, y0 = stroke[0]
    x1, y1 = stroke[-1]
    dx0, dy0 = stroke[round(n/(k-1))]
    #dx0, dy0 = np.array(stroke[1]) - np.array(stroke[0])
    #dx0, dy0 = dx0 / (np.linalg.norm([dx0, dy0])+1e-8), dy0 / (np.linalg.norm([dx0, dy0])+ 1e-8)
    #dx1, dy1 = np.array(stroke[-1]) - np.array(stroke[-2])
    #dx1, dy1 = dx1 / (np.linalg.norm([dx1, dy1])+1e-8), dy1 / (np.linalg.norm([dx1, dy1])+1e-8)
    return [x0, y0, dx0, dy0, x1, y1, dx1, dy1]

def convert_to_final_format(training_examples):
    """Converts strokes in training examples to the final format."""
    final_examples = []
    for example in training_examples:
        final_example = [process_stroke(stroke,4) for stroke in example if len(stroke)>3]
        final_examples.append(final_example)
    return final_examples

def main(x_data_path, save_path):
    # Load the data
    training_examples = load_data(x_data_path)
    print("loaded path")
    # Remove the time parameter and normalize strokes

    training_examples = [[adjust_points(stroke, num_samples=10) if len(stroke)>1 else [(float(stroke[0][0]), float(stroke[0][1]))]*10 for stroke in example] for example in training_examples]
    
    print("adjusted")
    #training_examples = normalize_strokes(training_examples)
    # Decompose strokes into smooth segments
    #training_examples = decompose_training_examples(training_examples)
    print("normalized")
    # Translate strokes
    training_examples = translate_strokes(training_examples)
    print("translated")
    # Convert to the final format
    #training_examples = convert_to_final_format(training_examples)
    training_examples = [[np.ravel(stroke) for stroke in example] for example in training_examples ]
    #training_examples = [[np.ravel(maximize_distance_with_local_search(stroke, 5, neighborhood_size=4)) for stroke in example] for example in training_examples ]
    print("optimized")
    # Save the processed data
    save_data(training_examples, save_path)
    print(f"Processed data saved to {save_path}")

def mainloop():
    try:
        training_examples = load_data('whiteboardtest.pkl') 
        # Remove the time parameter and normalize strokes

        training_examples = [[adjust_points(stroke, num_samples=10) if len(stroke)>1 else [(float(stroke[0][0]), float(stroke[0][1]))]*10 for stroke in example] for example in training_examples]
        #training_examples = normalize_strokes(training_examples)
    
        # Decompose strokes into smooth segments
        #training_examples = decompose_training_examples(training_examples)
    
        # Translate strokes
        training_examples = translate_strokes(training_examples)
    
        # Convert to the final format

        training_examples = [[np.ravel(stroke) for stroke in example] for example in training_examples ]
        #training_examples = [[np.ravel(maximize_distance_sum(stroke, 8)) for stroke in example] for example in training_examples ]
        save_path = 'bezwb.pkl'
        save_data(training_examples, save_path)
        print(f"Processed data saved to {save_path}")
    except:
        pass
    threading.Timer(0.5, mainloop).start()
mainloop()

"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process training examples for handwriting.")
    parser.add_argument("x_data_path", type=str, help="Path to x_data.pkl")
    parser.add_argument("save_path", type=str, help="Path to save the processed data")

    args = parser.parse_args()
    main(args.x_data_path, args.save_path)
"""
