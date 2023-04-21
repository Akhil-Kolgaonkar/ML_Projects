import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_path = 'dataset2.csv'

def load_faq():
    return pd.read_csv(faq_path)

def save_faq(df):
    df.to_csv(faq_path, index=False)

def add_to_faq(new_question, new_answer):
    df = pd.read_csv(faq_path)
    df = pd.concat([df, pd.DataFrame({'Question': [new_question], 'Answer': [new_answer]})], ignore_index=True)
    df.to_csv(faq_path, index=False)


def create_index_to_response(df):
    index_to_response = {}
    for index, row in df.iterrows():
        index_to_response[index] = row['Answer']
    return index_to_response

def train_chatbot(df):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(df['Question'])
    cosine_sim = cosine_similarity(question_vectors)
    index_to_response = create_index_to_response(df)
    return index_to_response, question_vectors, vectorizer, cosine_sim

def load_chatbot():
    df = load_faq()
    return train_chatbot(df)

def respond(user_input):
    index_to_response, question_vectors, vectorizer, cosine_sim = load_chatbot()
    input_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_tfidf, question_vectors)
    max_score = np.argmax(similarity_scores)
    if similarity_scores[0][max_score] == 0:
        new_question = input("Sorry, I do not understand. Please enter the answer to this question: ")
        add_to_faq(user_input, new_question)
        print("Thank you! I have learned a new question.")
    else:
        response_index = max_score
        response = index_to_response[response_index]
        print(response)


while True:
    user_input = input("Ask me a question: ")
    respond(user_input)
