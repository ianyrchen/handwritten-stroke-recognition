import torch
import torch.nn as nn
from train_character_model import TraceClassifier
from print import plot_tracetxt
import os
import string


def normalize_example(example, mean, std):
    """
    example is a tensor (seq_len, 2), where 2 is (x, y).
    """
    normalized = (example - mean) / std
    return normalized.to(torch.float32) 

def model_pred(filepath, model, ans, mean, std):
    x_data = []
    with open(filepath, 'r') as f:
        data_points_str = f.read().split('\n')[:-1] 
        data_points = []
        for line in data_points_str:
            coordinates = line.split(',')
            x, y = float(coordinates[0]), float(coordinates[1])
            data_points.append([x, y])  
        x_data.extend(data_points)

    input_tensor = torch.tensor(x_data, dtype=torch.float32)

    # Normalize
    # input_tensor = normalize_example(input_tensor, mean, std)

    # shape is (1, seq_len, input_dim)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)  

    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class_index = torch.argmax(probabilities, dim=1).item() 

    # labels are 0-9 + a-z
    label_map = {str(i): i for i in range(10)} 
    label_map.update({c: i + 10 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}) 
    index_to_label = {v: k for k, v in label_map.items()}

    #get predicted label
    predicted_label = index_to_label[predicted_class_index]
    # print(predicted_label)
    if predicted_label == ans:
        return True
    else:
        return False




model = TraceClassifier(input_dim=2, hidden_dim=128, output_dim=36)
model.load_state_dict(torch.load("character_recognition/model.pth"))
model.eval()  

mean, std = torch.load('normalization_params.pth')  

labels = [str(i) for i in range(10)]
labels += list(string.ascii_lowercase)

grand_total = 0
grand_total_correct = 0

for test_label in labels:
    file_path = './character_recognition/HCMYO-A/data/S01/' + test_label + '/trace_'
    ext = '.txt'

    correct = 0
    tot = 0
    for id in range(0, 50000):
        f = file_path + str(id) + ext
        if os.path.isfile(f):
            res = model_pred(filepath=f, model=model, ans=test_label, mean=mean, std=std)
            if res:
                correct += 1
            tot += 1

    print(test_label + " correct: ", correct)
    print(test_label + " total: ", tot)

    grand_total += tot
    grand_total_correct += correct

print("accuracy: ", grand_total_correct/grand_total)
