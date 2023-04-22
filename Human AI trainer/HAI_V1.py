import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_path = 'test_dataset.txt'

def load_chatbot():
    df = pd.read_csv(faq_path, delimiter='\t', encoding='ISO-8859-1')
    df = df.dropna()  # Drop any rows with NaN values
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
        if df[df['Question'].str.lower() == question.lower()].shape[0] > 0:
            df.loc[df['Question'].str.lower() == question.lower(), 'Answer'] = answer
            df.to_csv(faq_path, index=False, sep='\t')
            return "Answer updated for the question: " + df[df['Question'].str.lower() == question.lower()]['Question'].values[0]
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
    df = pd.read_csv(faq_path, delimiter='\t', encoding='ISO-8859-1')

    # Check if question already exists (ignoring case)
    existing_questions = [str(q).lower() for q in df['Question'] if not pd.isna(q)]
    if new_question.lower() in existing_questions:
        idx = existing_questions.index(new_question.lower())
        df.loc[idx, 'Answer'] = new_answer
        df.to_csv(faq_path, index=False, sep='\t')
        print("Answer updated for the question:", df.loc[idx, 'Question'])
    else:
        new_row = {'Question': new_question, 'Answer': new_answer}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(faq_path, index=False, sep='\t')
        print("New question added to the FAQ!")

while True:
    user_input = input("You: ")
    response = respond(user_input)
    print("AI: " + response)
