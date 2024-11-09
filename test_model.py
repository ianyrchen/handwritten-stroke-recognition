import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model('my_model.h5')

with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
    file.close()
with open('x_bezier_data.pkl', 'rb') as file:
    x = pickle.load(file)
    file.close()

xx = np.array(x[0]).reshape((1, 47, 5, 2))
print(xx)
prediction = model.predict(xx)
prediction = prediction[0]
print(prediction)

indices_of_largest = np.argsort(prediction)[-25:][::-1]
print("Indices of the n largest elements:", indices_of_largest)

print("The n largest elements:", prediction[indices_of_largest])
for i in indices_of_largest:
    print(features[i])