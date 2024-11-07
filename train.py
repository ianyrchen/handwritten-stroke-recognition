import pickle
import tensorflow

if __name__ == "__main__":
    with open('x_data.pkl', 'rb') as file:
        x = pickle.load(file)
        print('x loaded')
    with open('y_data.pkl', 'rb') as file:
        y = pickle.load(file)
        print('y loaded')

    print(len(x))
    print(len(y))