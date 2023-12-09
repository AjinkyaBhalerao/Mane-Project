from sentence_transformers import SentenceTransformer, util
import json
from nltk.stem import WordNetLemmatizer
import os
import subprocess
import spacy
import streamlit as st
import time

# Streamlit UI
st.title("MANE Chatbot")
st.text("Hello! I am Mane Bot. How can I help you today?")
user_image = "image.png"
chatbot_image = "chatbot.png"
chat_history = st.empty()
if 'chat_history_list' not in st.session_state:
    st.session_state.chat_history_list = []

# Load the lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load the SentenceTransformer model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cuda')

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
context_embeddings = model.encode(context)
question_embeddings = model.encode(questions)
answer_embeddings = model.encode(answers)

# Set the working directory to the FastChat folder
fastchat_directory = '/home/mane/FastChat'
vicuna_model_path = 'lmsys/vicuna-7b-v1.5'

user_input = st.text_input("Type your question: ")
while not user_input:
    user_input = st.text_input("Please enter a valid question: ").strip()

# Initialize spaCy for lemmatization
nlp = spacy.load('en_core_web_sm')

# Define a function for lemmatization
def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

if user_input:
    st.session_state.typing_message = st.text("Bot is typing...")
    time.sleep(2)

    # Encode the user's input
    user_input_embedding = model.encode(preprocess_text(user_input))

    # Calculate cosine similarity between the user's input and all questions in the dataset
    similarities = util.cos_sim(user_input_embedding, question_embeddings)[0]
    most_similar_index = similarities.argmax()
    highest_similarity = similarities[most_similar_index]

    # Set the threshold values
    high_threshold = 0.8
    medium_threshold = 0.5

    # Construct the command to run Vicuna with the user's question
    # Check the similarity against different thresholds
    if highest_similarity >= high_threshold:
        # High similarity, directly provide the answer
        st.write(f"User's input: {user_input}")
        st.write(f"Most similar question: {questions[most_similar_index]}")
        st.write(f"Corresponding answer: {answers[most_similar_index]}")
        st.write(f"Similarity Score: {highest_similarity}")

        # Ask for user satisfaction
        user_satisfaction = st.text_input("Is this the information you were looking for? (yes/no): ").lower()
        if user_satisfaction == 'no':
            # Continue the interaction loop
            user_input += " " + st.text_input("Please provide more details or clarify your question: ")
            user_input_embedding = model.encode(preprocess_text(user_input))
            similarities = util.cos_sim(user_input_embedding, question_embeddings)[0]
            most_similar_index = similarities.argmax()
            st.write(f"User's input after clarification: {user_input}")
            st.write(f"Most similar question after clarification: {questions[most_similar_index]}")
            st.write(f"Corresponding answer: {answers[most_similar_index]}")
            st.write(f"Similarity Score after clarification: {similarities[most_similar_index]}")

    else:
        if medium_threshold <= highest_similarity < high_threshold:
            # Medium similarity, ask for clarification
            st.write(f"The question is similar to: {questions[most_similar_index]}")
            st.write(f"Similarity Score: {highest_similarity}")
            user_confirmation = st.text_input("Is this the same question? (yes/no): ").lower()

            if user_confirmation == 'yes':
                # User confirms, provide the answer
                st.write(f"User's input: {user_input}")
                st.write(f"Most similar question: {questions[most_similar_index]}")
                st.write(f"Corresponding answer: {answers[most_similar_index]}")
                st.write(f"Similarity Score: {highest_similarity}")

                # Ask for user satisfaction
                user_satisfaction = st.text_input("Is this the information you were looking for? (yes/no): ").lower()
                if user_satisfaction == 'no':
                    # Continue the interaction loop
                    user_input += " " + st.text_input("Please provide more details or clarify your question: ")
                    user_input_embedding = model.encode(preprocess_text(user_input))
                    similarities = util.cos_sim(user_input_embedding, question_embeddings)[0]
                    most_similar_index = similarities.argmax()
                    st.write(f"User's input after clarification: {user_input}")
                    st.write(f"Most similar question after clarification: {questions[most_similar_index]}")
                    st.write(f"Corresponding answer: {answers[most_similar_index]}")
                    st.write(f"Similarity Score after clarification: {similarities[most_similar_index]}")
            else:
                # User denies, ask for clarification
                user_input += " " + st.text_input("Please provide more details or clarify your question: ")
                user_input_embedding = model.encode(preprocess_text(user_input))
                similarities = util.cos_sim(user_input_embedding, question_embeddings)[0]
                most_similar_index = similarities.argmax()
                st.write(f"User's input after clarification: {user_input}")
                st.write(f"Most similar question after clarification: {questions[most_similar_index]}")
                st.write(f"Corresponding answer: {answers[most_similar_index]}")
                st.write(f"Similarity Score after clarification: {similarities[most_similar_index]}")
        else:
            # Low similarity, use Vicuna for answering
            st.write("Low similarity. Asking Vicuna for an answer...")

            # Set the working directory to the FastChat folder
            os.chdir('/home/mane/FastChat')

            # Construct the command to run Vicuna with the user's question
            vicuna_command = ['python3', '-m', 'fastchat.serve.cli', '--model-path', f'lmsys/vicuna-7b-v1.5']
            try:
                # Use subprocess to run Vicuna and capture its output
                vicuna_process = subprocess.Popen(vicuna_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE, text=True)
                vicuna_out, vicuna_err = vicuna_process.communicate(input=user_input)

                if vicuna_process.returncode == 0:
                    # Extract user's question and Vicuna's answer from the output
                    lines = vicuna_out.strip().split('\n')
                    user_question = lines[0].split('USER: ')[-1].strip()
                    vicuna_answer = lines[-1].split('ASSISTANT: ')[-1].strip()

                    # Display the extracted information in the desired format
                    st.write(f"user: {user_question}")
                    st.write(f"chatbot: {vicuna_answer}")
                else:
                    st.write(f"Error while running Vicuna: {vicuna_err}")
            except Exception as e:
                st.write(f"Error while running Vicuna: {e}")
                vicuna_answer = "Error while running Vicuna"

            st.session_state.chat_history_list.append(("User", user_question))
            st.session_state.chat_history_list.append(("Chatbot", vicuna_answer))

            st.session_state.typing_message.empty()

    for q, a in st.session_state.chat_history_list:
        with st.container():
            if q == "User":
                st.image(user_image, width=30, use_column_width=False, output_format="png")
                st.markdown(f"**{q}:** {a}")
            else:
                st.image(chatbot_image, width=30, use_column_width=False, output_format="png")
                st.markdown(f"**{q}:** {a}")
