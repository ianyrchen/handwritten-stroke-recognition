import pickle
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    with open('y_data.pkl', 'rb') as file:
        y = pickle.load(file)
        print('y loaded')
        file.close()

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(y)
    tfidf_dense = tfidf_matrix.toarray()

    print("TF-IDF Vectors:\n", tfidf_dense)
    print("\nFeature Names (Vocabulary):\n", vectorizer.get_feature_names_out())
    print("\nShape of TF-IDF Matrix:", tfidf_dense.shape)
    
    features = vectorizer.get_feature_names_out()
    with open('features.pkl', 'wb') as file:
        pickle.dump(features, file)
        file.close()
    with open('y_tfidf_data.pkl', 'wb') as file:
        pickle.dump(tfidf_dense, file)
        file.close()
    
    
    
    # need to pad # of strokes per output
    # need to pad # of points per stroke
    '''
    max_points = 0
    for strokes in x:
        for stroke in strokes:
            max_points = max(max_points, len(stroke))
    max_strokes = 0
    min_strokes = float('inf')
    for strokes in x:
        max_strokes = max(max_strokes, len(strokes))
        min_strokes = min(min_strokes, len(strokes))
    print('max_points', max_points)
    print('max_strokes', max_strokes)
    print('min_strokes', min_strokes)
    
    
    for strokes in x:
        for i in range(max_strokes - len(strokes)):
            strokes.append([])
    print('a')
    count = 0
    for strokes in x:
        print(count)
        for stroke in strokes:
            for i in range(max_points - len(stroke)):
                stroke.append([0,0,float('inf')])
        count += 1
    print('b')
    with open('x_basicpadded_data.pkl', 'wb') as file:
        pickle.dump(x, file)
        file.close()
    '''