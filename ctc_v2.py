import tensorflow as tf
import numpy as np
import pickle
import string
import time
from contextlib import contextmanager

# Profiler context manager
@contextmanager
def profile_section(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'[{name}] execution time: {end_time - start_time:.6f} seconds')

# Include both lowercase and uppercase letters
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "

# Create char_map and reverse mapping
char_map = {char: idx for idx, char in enumerate(characters)}
rev_char_map = {idx: char for char, idx in char_map.items()}

# Custom Dataset loader
def load_data():
    with open('x_data.pkl', 'rb') as file:
        x = pickle.load(file)
    with open('y_char_data.pkl', 'rb') as file:
        y = pickle.load(file)
    return x, y

x, y = load_data()

# Custom Dataset
def stroke_dataset(x, y, char_map):
    def gen():
        for i in range(len(x)):
            input_sequence = [
                [[float(val) for val in point] for point in stroke]
                for stroke in x[i]
            ]
            flattened_input_sequence = [point for stroke in input_sequence for point in stroke]
            target_sequence = [char_map[char] for char in y[i]]
            yield np.array(flattened_input_sequence, dtype=np.float32), np.array(target_sequence, dtype=np.int32)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

# Create dataset
dataset = stroke_dataset(x, y, char_map)

# Padded batching to handle variable-length sequences
dataset = dataset.padded_batch(
    batch_size=32,
    padded_shapes=([None, 3], [None]),  # Define shapes for input and target
    padding_values=(0.0, 0)  # Padding values for inputs and targets
)

# Add prefetch for performance optimization
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# New Model using Convolutions
class StrokeCNNModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(StrokeCNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu')
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense(x)
        return x

# CTC Loss function
def ctc_loss_fn(y_true, y_pred, input_length, target_length):
    y_pred = tf.nn.log_softmax(y_pred)
    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logits=y_pred,
        label_length=target_length,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=num_classes - 1
    )
    return tf.reduce_mean(loss)

# Training
@tf.function
def train_step(inputs, input_lengths, targets, target_lengths, model, optimizer, loss_object, train_loss_metric):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        logits_time_len = tf.fill([tf.shape(inputs)[0]], tf.shape(logits)[1])
        loss = loss_object(targets, logits, logits_time_len, input_lengths)

    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    train_loss_metric.update_state(loss)

def train_model(model, dataset, optimizer, loss_object, num_epochs):
    train_loss_metric = tf.keras.metrics.Mean()
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}')
        for batch, (inputs, targets) in enumerate(dataset):
            print(batch)
            input_lengths = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            target_lengths = tf.fill([tf.shape(targets)[0]], tf.shape(targets)[1])
            train_step(inputs, input_lengths, targets, target_lengths, model, optimizer, loss_object, train_loss_metric)
            print(f'Batch {batch} - Loss: {train_loss_metric.result().numpy()}')
        print(f'Epoch {epoch + 1} - Loss: {train_loss_metric.result().numpy()}')
        train_loss_metric.reset_states()

if __name__ == "__main__":
    batch_size = 32
    num_classes = len(char_map) + 1
    hidden_dim = 128
    input_dim = 3  # x, y, time

    model = StrokeCNNModel(input_dim, hidden_dim, num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    num_epochs = 10

    train_model(model, dataset, optimizer, ctc_loss_fn, num_epochs)

    model.save('ctc_v2_model')
    loaded_model = tf.keras.models.load_model('ctc_v2_model', compile=False)
