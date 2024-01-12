from sentence_transformers import SentenceTransformer, util
import json
from nltk.stem import WordNetLemmatizer
import os
import subprocess
import shlex
import spacy
import sys
import pickle

# Set the working directory to the FastChat folder
fastchat_directory = '/home/mane/FastChat'
os.chdir(fastchat_directory)

# Load the lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load the SentenceTransformer model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cuda')

if (os.path.exists('/home/mane/Mane-Project/Model/embeddings.pkl')==False):
    print("here")
    # Load the FAQ data
    with open('/home/mane/Mane-Project/Model/faq_data.json', 'r') as file:
        data = json.load(file)

    # Extract questions and answers from the data
    filename = [entry['filename'] for entry in data]
    context = [entry['context'] for entry in data]
    questions = [entry.get('question', '') for entry in data]
    answers = [entry.get('answer', '') for entry in data]

    # Encode questions and answers
    filename_embeddings = model.encode(filename)
    #context_embeddings = model.encode(context)
    #question_embeddings = model.encode(questions)
    answer_embeddings = model.encode(answers)
    combined_context_questions = [context + " " + question for context, question in zip(context, questions)]
    combined_embeddings = model.encode(combined_context_questions, convert_to_tensor=True)

    #Store sentences & embeddings on disc
    with open('/home/mane/Mane-Project/Model/embeddings.pkl', "wb") as fOut:
        print("create pickles")
        pickle.dump({'combined_embeddings': combined_embeddings, 'questions' : questions, 'answers' : answers,'filename' : filename}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def openBert_embeddings():
#Load sentences & embeddings from disc
    with open('/home/mane/Mane-Project/Model/embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_combined_embeddings = stored_data['combined_embeddings']
        stored_questions = stored_data['questions']
        stored_answers = stored_data['answers'] 
        stored_filename = stored_data['filename']                           
    return stored_combined_embeddings, stored_questions, stored_answers, stored_filename


# Set the working directory to the FastChat folder
vicuna_model_path = 'lmsys/vicuna-7b-v1.5'
    
# Initialize spaCy for lemmatization
nlp = spacy.load('en_core_web_sm')
chat_history = []
last_answered_filename = None

# Define a function for lemmatization
def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

def provide_document_reference():
    if last_answered_filename:
        return f"The document reference for the last answer is: {last_answered_filename}"
    else:
        return "No document reference available."

def print_chat_history():
    print("\nChat History:")
    for entry in chat_history:
        print(f"{entry['user']}: {entry['message']}")


while True:
    user_input = input("\nAsk your question (type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break
    # After the user asks the next question
    if user_input.lower() == 'reference':
        document_reference = provide_document_reference()
        print(document_reference)
        continue


    while not user_input:
        print("Please enter a valid question.")
        user_input = input("Ask your question (type 'exit' to quit): ").strip()
    # Encode the user's input
    user_input_embedding = model.encode(preprocess_text(user_input), convert_to_tensor=True)
    combined_embeddings, questions, answers, filename = openBert_embeddings()
    
    # Calculate cosine similarity between the user's input and all questions in the dataset
    similarities = util.cos_sim(user_input_embedding, combined_embeddings)[0]
    most_similar_index = similarities.argmax()
    highest_similarity = similarities[most_similar_index]

    # Set the threshold values
    high_threshold = 0.8
    medium_threshold = 0.5

    # Construct the command to run Vicuna with the user's question
    # Check the similarity against different thresholds
    if highest_similarity >= high_threshold:
        # High similarity, directly provide the answer
        print(f"\nUSER: {user_input}")
        print(f"Most similar question: {questions[most_similar_index]}")
        print(f"ASSISTANT: {answers[most_similar_index]}")
        print(f"Similarity Score: {highest_similarity}")
        sys.stdout.flush()
        # Ask for user satisfaction
        user_satisfaction = input("Is this the information you were looking for? (yes/no): ").lower()
        
        if user_satisfaction == 'no':
            # Continue the interaction loop
            user_input += " " + input("Please provide more details or clarify your question: ")
            user_input_embedding = model.encode(preprocess_text(user_input), convert_to_tensor=True)
            similarities = util.cos_sim(user_input_embedding, combined_embeddings)[0]
            most_similar_index = similarities.argmax()
            print(f"\nUSER: {user_input}")
            print(f"Most similar question after clarification: {questions[most_similar_index]}")
            print(f"ASSISTANT: {answers[most_similar_index]}")
            print(f"Similarity Score after clarification: {similarities[most_similar_index]}")
            
        last_answered_filename = filename[most_similar_index]


    elif medium_threshold <= highest_similarity < high_threshold:
        # Medium similarity, ask for clarification
        print(f"The question is similar to: {questions[most_similar_index]}")
        print(f"Similarity Score: {highest_similarity}")
        sys.stdout.flush()
        user_confirmation = input("Is this the same question? (yes/no): ").lower()

        if user_confirmation == 'yes':
            # User confirms, provide the answer
            print(f"\nUSER: {user_input}")
            print(f"Most similar question: {questions[most_similar_index]}")
            print(f"ASSISTANT: {answers[most_similar_index]}")
            print(f"Similarity Score: {highest_similarity}")

            # Ask for user satisfaction
            sys.stdout.flush()
            user_satisfaction = input("Is this the information you were looking for? (yes/no): ").lower()
            if user_satisfaction == 'no':
                # Continue the interaction loop
                user_input += " " + input("Please provide more details or clarify your question: ")
                user_input_embedding = model.encode(preprocess_text(user_input), convert_to_tensor=True)
                similarities = util.cos_sim(user_input_embedding, combined_embeddings)[0]
                most_similar_index = similarities.argmax()
                print(f"User's input after clarification: {user_input}")
                print(f"Most similar question after clarification: {questions[most_similar_index]}")
                print(f"Corresponding answer: {answers[most_similar_index]}")
                print(f"Similarity Score after clarification: {similarities[most_similar_index]}")
        else:
            # User denies, ask for clarification
            user_input += " " + input("Please provide more details or clarify your question: ")
            user_input_embedding = model.encode(preprocess_text(user_input), convert_to_tensor=True)
            similarities = util.cos_sim(user_input_embedding, combined_embeddings)[0]
            # Low similarity, display similar questions
            print("I found a few similar questions. Here's the best match:")
            similar_questions = []
            for idx, sim_score in enumerate(similarities):
                if medium_threshold <= sim_score < high_threshold:
                    similar_questions.append((questions[idx], sim_score))

            for i, (similar_question, sim_score) in enumerate(similar_questions, start=1):
                print(f"{i}- {similar_question} (Score: {sim_score})")

            # Ask the user to choose a number
            sys.stdout.flush()
            user_choice = input("If your question is similar to one of these, type the number. Otherwise, press Enter to continue: ")

            if user_choice.isdigit() and 1 <= int(user_choice) <= len(similar_questions):
                selected_index = int(user_choice) - 1
                selected_question = similar_questions[selected_index][0]
                selected_score = similar_questions[selected_index][1]

                print(f"Selected question: {selected_question}")
                print(f"Corresponding answer: {answers[questions.index(selected_question)]}")
                print(f"Similarity Score: {selected_score}")
            else:
                # Low similarity, use Vicuna for answering
                print("Low similarity. Asking Vicuna for an answer...")

                # Set the working directory to the FastChat folder
                os.chdir('/home/mane/FastChat')

                # Construct the command to run Vicuna with the user's question
                vicuna_command = ['python3', '-m', 'fastchat.serve.cli', '--model-path', f'lmsys/vicuna-7b-v1.5']
                try:
                    # Use subprocess to run Vicuna and capture its output
                    # Use subprocess to run Vicuna and capture its output
                    vicuna_process = subprocess.Popen(vicuna_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    vicuna_out, vicuna_err = vicuna_process.communicate(input=user_input)

                    if vicuna_process.returncode == 0:
                        # Extract user's question and Vicuna's answer from the output
                        lines = vicuna_out.strip().split('\n')
                        # Assuming the first line contains Vicuna's answer and starts with 'ASSISTANT: '
                        vicuna_answer = lines[0].split('ASSISTANT: ')[-1].strip()

                        # Display the extracted information in the desired format
                        print(f"USER: {user_input}")
                        print(f"ASSISTANT: {vicuna_answer}")
                    else:
                        print(f"Error while running Vicuna: {vicuna_err}")

                except Exception as e:
                    print(f"Error while running Vicuna: {e}")
                    vicuna_answer = "Error while running Vicuna"
        last_answered_filename = filename[most_similar_index]

    else:
        # Low similarity, use Vicuna for answering
        print("Low similarity. Asking Vicuna for an answer...")

        # Set the working directory to the FastChat folder
        os.chdir('/home/mane/FastChat')

        # Construct the command to run Vicuna with the user's question
        vicuna_command = ['python3', '-m', 'fastchat.serve.cli', '--model-path', f'lmsys/vicuna-7b-v1.5']
        try:
            # Use subprocess to run Vicuna and capture its output
            vicuna_process = subprocess.Popen(vicuna_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            vicuna_out, vicuna_err = vicuna_process.communicate(input=user_input)

            if vicuna_process.returncode == 0:
                # Extract user's question and Vicuna's answer from the output
                lines = vicuna_out.strip().split('\n')
                # Assuming the first line contains Vicuna's answer and starts with 'ASSISTANT: '
                vicuna_answer = lines[0].split('ASSISTANT: ')[-1].strip()

                # Display the extracted information in the desired format
                print(f"USER: {user_input}")
                print(f"ASSISTANT: {vicuna_answer}")
            else:
                print(f"Error while running Vicuna: {vicuna_err}")

        except Exception as e:
            print(f"Error while running Vicuna: {e}")
            vicuna_answer = "Error while running Vicuna"