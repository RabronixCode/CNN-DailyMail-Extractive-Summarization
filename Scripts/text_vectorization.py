import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    # Convert the sparse matrix to a dense array (optional)
    return tfidf_matrix.toarray()

    # Print results
    #print("Feature Names:", feature_names)
    #print("TF-IDF Matrix:")
    #print(tfidf_array)

    

