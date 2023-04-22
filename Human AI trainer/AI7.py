import csv
import chardet
import random
from Levenshtein import distance

# Load questions and answers from file
questions = []
answers = []


# List of possible encodings to try
encodings = ['utf-8', 'iso-8859-1', 'cp1252']

# Open the question-answer pairs file with the appropriate encoding
for encoding in encodings:
    try:
        with open('test_dataset.txt', mode='r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                question = row['Question']
                answer = row['Answer']
                questions.append(question)
                answers.append(answer)
                print("Q:", question)
                print("A:", answer)
                print("-------")
                print("encodings:",encoding)
        break
    except UnicodeDecodeError:
        continue

# Define chatbot function
def chatbot():
    print("Hello! I'm a chatbot. What's your name?")
    name = input()
    print(f"Nice to meet you, {name}!")

    while True:
        # Get user input
        print("What's your question? (or type 'quit' to exit)")
        user_input = input().lower()

        # Exit loop if user types 'quit'
        if user_input == 'quit':
            break

        # Find best match in questions
        best_match = None
        lowest_distance = float('inf')
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


# Run chatbot
chatbot()
