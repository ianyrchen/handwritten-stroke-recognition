import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, TimeDistributed, Conv1D, LSTM, Dense, Dropout, Reshape, Bidirectional
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# Parameters
input_dim = 128  # Dimension of input feature vectors
num_classes = 75  # Number of output characters (e.g., 26 letters + space + blank + special)
batch_size = 4
epochs = 10
# Load data
with open('y_data.pkl', 'rb') as file:
    y = pickle.load(file)
    print('y loaded')
    file.close()
with open('x_bezier_data.pkl', 'rb') as file:
    x = pickle.load(file)
    print('x loaded')
    file.close()


def pad(input_data, target_data):
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(input_data, padding="post", dtype="float32")
    padded_targets = tf.keras.preprocessing.sequence.pad_sequences(target_data, padding="post", dtype="int32")
    # Create Lengths Arrays
    input_lengths = np.array([seq.shape[0] for seq in input_data])
    label_lengths = np.array([len(seq) for seq in target_data])
    
    # Ensure Correct Shapes
    input_lengths = input_lengths[:, np.newaxis]
    label_lengths = label_lengths[:, np.newaxis]
    return padded_inputs, input_lengths, padded_targets, label_lengths

x = np.asarray(x).astype(np.float32)
max_len = max(len(seq) for seq in y)
y_padded = [seq + [0] * (max_len - len(seq)) for seq in y]  # Padding with zeros
y = np.asarray(y_padded).astype(np.float32)

train_inputs, x_temp, train_labels, y_temp = train_test_split(x, y, test_size=0.3, random_state=5)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=5)

train_inputs, train_input_lengths, train_labels, train_label_lengths = pad(train_inputs, train_labels)
print("Input lengths:", train_input_lengths)
print("Label lengths:", train_label_lengths)

# Generate Mock Data for Training
"""max_input_len = 200
max_label_len = 50
train_inputs, train_labels, train_input_lengths, train_label_lengths = generate_mock_data(
    batch_size, max_input_len, max_label_len, input_dim, num_classes
)"""
# Model Definition
def create_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(None, 5, 2,1), name="inputs")  # Update input shape
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
    )(inputs)  # Process (5, 2) dimensions
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)  # Flatten the spatial dimensions
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Dense(num_classes + 1, activation="softmax")(x)  # +1 for blank token
    model = tf.keras.Model(inputs, x)
    return model
    """
def create_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(None, input_dim), name="inputs")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(inputs)
    x = tf.keras.layers.Dense(num_classes + 1, activation="softmax")(x)  # +1 for blank token
    model = tf.keras.Model(inputs, x)
    return model"""

# Create Model
model = create_model(input_dim, num_classes)
model.summary()

# Training with Custom Loss and Training Step
@tf.function
def ctc_loss(y_true, y_pred, input_lengths, label_lengths):
    return tf.reduce_mean(
        tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_lengths, label_lengths)
    )

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Training Step
@tf.function
def train_step(inputs, labels, input_lengths, label_lengths):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = ctc_loss(labels, logits, input_lengths, label_lengths)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training Loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step in range(len(train_inputs) // batch_size):
        batch_inputs = train_inputs[step * batch_size : (step + 1) * batch_size]
        batch_labels = train_labels[step * batch_size : (step + 1) * batch_size]
        batch_input_lengths = train_input_lengths[step * batch_size : (step + 1) * batch_size]
        batch_label_lengths = train_label_lengths[step * batch_size : (step + 1) * batch_size]
        
        loss = train_step(batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths)
        print(f"  Step {step + 1}, Loss: {loss.numpy():.4f}")

print("Training complete!")
"""
def pad(input_data, target_data):
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(input_data, padding="post", dtype="float32")
    padded_targets = tf.keras.preprocessing.sequence.pad_sequences(target_data, padding="post", dtype="int32")
    # Create Lengths Arrays
    input_lengths = np.array([seq.shape[0] for seq in input_data])
    label_lengths = np.array([len(seq) for seq in target_data])
    
    # Ensure Correct Shapes
    input_lengths = input_lengths[:, np.newaxis]
    label_lengths = label_lengths[:, np.newaxis]
    return padded_inputs, input_lengths, paddet_targets, label_lengths


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
    # Load data
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

    # Model params
    sequence_length = 47            
    sub_step_dim1 = 5               # control points
    sub_step_dim2 = 2               
    hidden_dim = 128                # dimension of the lstm layers
    num_layers = 2                  # number of lstm layers
    output_dim = 1000               # output dimension for multi-label classification

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
"""
