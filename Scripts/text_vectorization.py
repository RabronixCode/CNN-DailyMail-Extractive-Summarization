import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# This method returns same vocab size matrix for every array of sentences
def tfidf(articles, a_docs, highlights, h_docs):
    vectorizer = TfidfVectorizer()
    docs = a_docs + h_docs
    vectorizer.fit(docs) # Same vocabulary for all
    articles_transformed = [vectorizer.transform(article).toarray() for article in articles]
    highlights_transformed = [vectorizer.transform(highlight).toarray() for highlight in highlights]
    
    return articles_transformed, highlights_transformed


    
