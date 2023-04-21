import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_path = 'dataset2.csv'


def load_chatbot():
    df = pd.read_csv(faq_path)
    question_vectors, vectorizer = preprocess_data(df['Question'])
    return df, question_vectors, vectorizer


def preprocess_data(text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text)
    return vectors, vectorizer


def cosine_similarity_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def respond(user_input):
    df, question_vectors, vectorizer = load_chatbot()

    if user_input.lower().startswith("addfaq"):
        split_input = user_input.lower().split("addfaq")
        if len(split_input) != 2:
            return "Sorry, I didn't understand. Please try again."

        question = split_input[1].strip()
        answer = input("What should the new answer be?")

        # Check if question already exists
        if question in df['Question'].values:
            df.loc[df['Question'] == question, 'Answer'] = answer
            df.to_csv(faq_path, index=False)
            return "Answer updated for the question: " + question
        else:
            add_to_faq(question, answer)
            return "Thanks, I've added this question to my database!"

    input_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_tfidf, question_vectors)
    closest_question_index = np.argmax(similarity_scores)
    if similarity_scores[0][closest_question_index] < 0.1:
        return "Sorry, I'm not sure how to respond to that."
    else:
        return df.iloc[closest_question_index]['Answer']


def add_to_faq(new_question, new_answer):
    df = pd.read_csv(faq_path)
    new_row = pd.DataFrame({'Question': [new_question], 'Answer': [new_answer]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(faq_path, index=False)


while True:
    user_input = input("You: ")
    response = respond(user_input)
    print("AI: " + response)