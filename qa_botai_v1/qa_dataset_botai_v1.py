import csv
import chardet
import random
from Levenshtein import distance

# Define function to load questions and answers from file
def load_qa_pairs(filename, encodings):
    # Try each encoding in the list until we can successfully read the file
    for encoding in encodings:
        try:
            with open(filename, mode='r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter='\t')
                questions = []
                answers = []
                for row in reader:
                    questions.append(row['Question'])
                    answers.append(row['Answer'])
                return questions, answers, encoding
        except UnicodeDecodeError:
            continue

    # If we can't read the file with any of the encodings, raise an exception
    raise ValueError(f"Unable to read file '{filename}' with encodings {encodings}")


# Load questions and answers from files
filenames = ['question_answer_pairs.txt', 'question_answer_pairs2.txt', 'question_answer_pairs3.txt']
encodings = ['utf-8', 'iso-8859-1', 'cp1252']
datasets = []

for filename in filenames:
    questions, answers, encoding = load_qa_pairs(filename, encodings)
    datasets.append((questions, answers, encoding))
    print(filename)

# Define chatbot function
def chatbot():
    print("Hello! I'm a chatbot. What's your name?")
    name = input()
    print(f"Nice to meet you, {name}!")
    print("no.of dataset:",len(datasets))

    while True:
        # Get user input
        print("What's your question? (or type 'quit' to exit)")
        user_input = input().lower()

        # Exit loop if user types 'quit'
        if user_input == 'quit':
            break

        # Find best match in all datasets
        best_match = None
        lowest_distance = float('inf')
        count=0
        for questions, answers, _ in datasets:
            count= count+1
            print("total no. of Ques.per datasets:",(count),len(questions))
            for question, answer in zip(questions, answers):
                dist = distance(user_input, question)
                if dist < lowest_distance:
                    best_match = answer
                    lowest_distance = dist

        # If no match found, use a random response
        if lowest_distance == len(user_input):
            responses = [
                "I'm sorry, I don't understand.",
                "Can you please rephrase that?",
                "I'm not sure what you mean.",
            ]
            best_match = random.choice(responses)

        # Print chatbot response
        print(best_match)


# Define function to compute similarity between two strings
def compute_similarity(s1, s2):
    # Replace with your preferred similarity metric
    # For example, you could use cosine similarity or Jaccard index
    return int(s1 == s2)


# Run chatbot
chatbot()
