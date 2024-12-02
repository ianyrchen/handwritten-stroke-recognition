import matplotlib.pyplot as plt
import torch 

def plot_tracetxt(filepath):
    # Initialize lists to store x and y coordinates
    x_coords = []
    y_coords = []

    # Read the file and process data
    with open(filepath, 'r') as file:
        for line in file:
            y, x, _ = map(int, line.strip().split(','))  # Split the line and ignore the z-coordinate
            x_coords.append(x)
            y_coords.append(y)

    # Plot the points and connect them with line segments
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')

    # Add labels and grid
    plt.title('Character Trace Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # https://data.mendeley.com/datasets/ms3sbpbrgp/1
    filepath = './character_recognition/preview/S01/0/trace_20292.txt'
    filepath = './character_recognition/preview/S01/1/trace_26537.txt'
    filepath = './character_recognition/preview/S01/2/trace_21494.txt'
    #filepath = './character_recognition/preview/S01/3/trace_22098.txt'
    
    #plot_tracetxt(filepath)


    dataset_data, dataset_labels = torch.load('character_recognition/dataset_data_labels.pth')
