import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('dataset.csv')

def create_index_to_response(df):
    index_to_response = {}
    for index, row in df.iterrows():
        index_to_response[index] = row['Response']
    return index_to_response

def load_chatbot():
    index_to_response = create_index_to_response(df)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Question'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    while True:
        user_input = input("You: ")
        input_tfidf = tfidf_vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)
        closest_match_index = np.argmax(similarity_scores)
        response = index_to_response[closest_match_index]
        print("AkhilgenixAI: " + response)

load_chatbot()