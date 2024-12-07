import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, TimeDistributed, Conv1D, LSTM, Dense, Dropout, Reshape, Bidirectional
from tensorflow.keras.models import Model

def model_init(sequence_length, sub_step_dim1, sub_step_dim2, hidden_dim, num_layers, output_dim):
    inputs = Input(shape=(sequence_length, sub_step_dim1, sub_step_dim2))

    x = TimeDistributed(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))(inputs)
    x = TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))(x)

    # reshape to (batch_size, sequence_length, features) for LSTM
    x = Reshape((sequence_length, sub_step_dim1 * 32))(x)  

    for _ in range(num_layers):
        x = Bidirectional(LSTM(hidden_dim, return_sequences=True))(x)
        x = Dropout(0.15)(x)

    x = LSTM(hidden_dim)(x)
    outputs = Dense(output_dim, activation='sigmoid')(x) 

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    with open('y_tfidf_data.pkl', 'rb') as file:
        y = pickle.load(file)
        print('y loaded')
        file.close()
    with open('x_bezier_data.pkl', 'rb') as file:
        x = pickle.load(file)
        print('x loaded')
        file.close()

    x = np.asarray(x).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=5)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=5)

    sequence_length = 47            
    sub_step_dim1 = 5               # control points
    sub_step_dim2 = 2               
    hidden_dim = 128                
    num_layers = 2                 
    output_dim = 1000               

    model = model_init(sequence_length, sub_step_dim1, sub_step_dim2, hidden_dim, num_layers, output_dim)

    # train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=32
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    model.save('my_model.h5')  

