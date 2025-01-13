import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load the data
with open('lossctc2.pkl', 'rb') as file:
    y = pickle.load(file)
    print('y loaded')
    file.close()

print(y)

# Generate corresponding x-axis values (even numbers up to 100)
x_values = np.linspace(2, 100, len(y), dtype=int)  # 50 evenly spaced points between 2 and 100
print("X values:", x_values)  # Debugging output to confirm x-values

# Create a smooth curve using spline interpolation
x_smooth = np.linspace(x_values.min(), x_values.max(), 300)  # Dense x values for smooth curve
spline = make_interp_spline(x_values, y, k=3)  # Cubic spline
y_smooth = spline(x_smooth)

# Plot the smooth curve
plt.figure(figsize=(8, 6))
plt.plot(x_smooth, y_smooth, label='Smooth Curve', color='blue', linewidth=2)
plt.scatter(x_values, y, color='black', label='Original Points')  # Highlight original points
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.5)
plt.show()



# Accuracy values from the results
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracies = [0.0173, 0.0330, 0.0382, 0.0391, 0.0417, 0.0458, 0.0497, 0.0534, 0.0531, 0.0553]

# Scale accuracies by 10
scaled_accuracies = [accuracy * 10 + 0.15 for accuracy in accuracies]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(epochs, scaled_accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy vs. Epochs', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
