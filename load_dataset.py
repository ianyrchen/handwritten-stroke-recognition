import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# writer_id ranges from 0-169
# drawing_id ranges from 0-(164ish, varies)

def plot_2d(x, y):
  plt.scatter(x, y, color='b')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

def plot_3d(x, y, z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(x, y, z, color='b')  

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()
  
if __name__ == "__main__":
  for writer_id in range(0, 170):
    for drawing_id in range(0, 170):
      filename = f"BRUSH/{writer_id}/{drawing_id}"
      if os.path.isfile(filename):
          with open(filename, 'rb') as f:
              [sentence, drawing, label] = pickle.load(f)
              print(sentence)
              
              x = drawing[:, 0]
              y = drawing[:, 1]
              y = -y
              z = drawing[:, 2]

              plot_2d(x, y) # when you close a plot why is there a new different plot of the same sentence popping up
              # plot_3d(x, y, z) # got no clue what the 0/1 labeling in z is supposed to be

              # all labels are same for a sentence?
              #print(label[0])
              break



