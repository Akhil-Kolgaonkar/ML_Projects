import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_file = 'dataset2.csv'
df = pd.read_csv(faq_file)

# initialize the vectorizer and calculate the tfidf matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Question'].values.astype('U'))

# calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def create_index_to_response(df):
    index_to_response = {}
    for index, row in df.iterrows():
        index_to_response[index] = row['Answer']
    return index_to_response

index_to_response = create_index_to_response(df)


def add_to_faq(new_question, new_answer):
        global df
        new_row = {'Question': new_question, 'Answer': new_answer}
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        df.to_csv(faq_file, index=False)

def respond(user_input):
    global index_to_response, df
    input_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)[0]
    most_similar_index = similarity_scores.argsort()[-1]
    if similarity_scores[most_similar_index] == 0:
        print("I'm sorry, I don't understand.")
    else:
        print(index_to_response[most_similar_index])
        add_new = input("Do you want to add a new question and answer? (yes or no) ")
        if add_new == 'yes':
            new_question = input("What is the new question? ")
            new_answer = input("What is the answer to the new question? ")
            add_to_faq(new_question, new_answer)
            print("Thank you for adding a new question and answer!")
        else:
            print("Okay, let me know if you have any other questions!")

while True:
    user_input = input("What is your question? ")
    if user_input.lower() == 'quit':
        break
    respond(user_input)
