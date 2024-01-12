#streamlit run app.py


from sentence_transformers import SentenceTransformer, util
import json
from nltk.stem import WordNetLemmatizer
import os
import subprocess
import spacy
import streamlit as st
import pickle

# Load the SentenceTransformer model
    
USER_image = "/home/mane/Mane-Project/Model/image.png"
chatbot_image = "/home/mane/Mane-Project/Model/chatbot.png"

# Initialize spaCy for lemmatization
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cuda')

if (os.path.exists('/home/mane/Mane-Project/Model/embeddings.pkl')==False):
    
    # Load the FAQ data
    with open('/home/mane/Mane-Project/Model/faq_data.json', 'r') as file:
        data = json.load(file)
        
    # Load the lemmatizer for word normalization
    lemmatizer = WordNetLemmatizer()
    # Extract questions and answers from the data
    filename = [entry['filename'] for entry in data]
    context = [entry['context'] for entry in data]
    questions = [entry.get('question', '') for entry in data]
    answers = [entry.get('answer', '') for entry in data]

    # Encode questions and answers
    filename_embeddings = model.encode(filename)
    combined_context_questions = [context + " " + question for context, question in zip(context, questions)]
    combined_embeddings = model.encode(combined_context_questions)
    answer_embeddings = model.encode(answers)


    #Store sentences & embeddings on disc
    with open('embeddings.pkl', "wb") as fOut:
         pickle.dump({'combined_embeddings': combined_embeddings, 'questions' : questions, 'answers' : answers, 'filename' : filename}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


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
fastchat_directory = '/home/mane/FastChat'
vicuna_model_path = 'lmsys/vicuna-7b-v1.5'

# Streamlit UI
# Add custom CSS for chat bubbles and alignment
st.markdown("""
    <style>
    .chat-bubble {
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 60%;
        color: black; /* Change text color here */
    }
    .USER-bubble {
        background-color: #E1FFC7;
        margin-left: auto;
        text-align: right;
    }
    .bot-bubble {
        background-color: #D3D3D3;
        margin-right: auto;
        text-align: left;
    }
    .stImage img { /* Adjusted CSS selector for circular images */
        border-radius: 50%;
        height: 40px;
        width: 40px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("MANE Chatbot")
st.text("Hello!👋 I am Mane Bot. How can I help you today?")

if 'chat_history_list' not in st.session_state:
    st.session_state.chat_history_list = []

# Define states for different stages of the conversation
CONVERSATION_STATES = {
    'INITIAL': 'initial',
    'AWAITING_CONFIRMATION': 'awaiting_confirmation',
    'AWAITING_DETAILS': 'awaiting_details',
    'AWAITING_USER_INPUT': 'awaiting_USER_input',
    'AWAITING_NUMERIC_INPUT': 'awaiting_numeric_input',
}

# Initialize session state variables
if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']

last_answered_filename = None

def provide_document_reference():
    if last_answered_filename:
        return f"The document reference for the last answer is: {last_answered_filename}"
    else:
        return "No document reference available."
# Define a function for lemmatizationf
def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

def add_to_chat_history(role, message):
    st.session_state.chat_history_list.append((role, message))

def display_messages():
    for role, message in st.session_state.chat_history_list:
        if role == "USER":
            col1, col2, col3 = st.columns([1, 6, 1])
            with col3:
                st.image(USER_image, width=40, output_format="png", use_column_width=True)
            with col2:
                st.markdown(f"<div class='chat-bubble USER-bubble'>{message}</div>", unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                st.image(chatbot_image, width=40, output_format="png", use_column_width=True)
            with col2:
                st.markdown(f"<div class='chat-bubble bot-bubble'>{message}</div>", unsafe_allow_html=True)

def display_chat_history():
    for role, message in st.session_state.chat_history_list:
        if role == "USER":
            st.image(USER_image, width=40, output_format="png", use_column_width=True)
            st.markdown(f"<div class='chat-bubble USER-bubble'>{message}</div>", unsafe_allow_html=True)
        else:
            st.image(chatbot_image, width=40, output_format="png", use_column_width=True)
            st.markdown(f"<div class='chat-bubble bot-bubble'>{message}</div>", unsafe_allow_html=True)

# Main interaction loop
def handle_interaction(prompt):
    if st.session_state.conversation_state == CONVERSATION_STATES['INITIAL']:
        process_initial_prompt(prompt)
    elif st.session_state.conversation_state == CONVERSATION_STATES['AWAITING_CONFIRMATION']:
        handle_confirmation(prompt)
    elif st.session_state.conversation_state == CONVERSATION_STATES['AWAITING_DETAILS']:
        handle_details(prompt)  
    elif st.session_state.conversation_state == CONVERSATION_STATES['AWAITING_NUMERIC_INPUT']:
        if prompt.isdigit():
            handle_numeric_input(prompt)
        else:
            add_to_chat_history("ASSISTANT", "That's not a number. Please enter the number corresponding to your question.")

        
def process_initial_prompt(prompt):
    if 'last_answered_filename' in st.session_state and prompt.lower().strip() in ["what is the reference for the last answer?", "source of last answer?", "last answer reference?"]:
        if st.session_state.last_answered_filename:
            add_to_chat_history("ASSISTANT", f"The reference for the last answer is: {st.session_state.last_answered_filename}")
        else:
            add_to_chat_history("ASSISTANT", "I don't have a reference for the last answer.")
    else :
        USER_input_embedding = model.encode(preprocess_text(prompt))
        combined_embeddings, questions, answers, filename = openBert_embeddings()
        similarities = util.cos_sim(USER_input_embedding, combined_embeddings)[0]
        most_similar_index = similarities.argmax()
        highest_similarity = similarities[most_similar_index]

        high_threshold = 0.8
        medium_threshold = 0.5

        if highest_similarity >= high_threshold:
            # Answer found with high confidence
            add_to_chat_history("ASSISTANT", answers[most_similar_index])
            add_to_chat_history("ASSISTANT", "Is this the information you were looking for?")
            st.session_state.conversation_state = CONVERSATION_STATES['AWAITING_CONFIRMATION']
            last_answeredn_filename = filename[most_similar_index]
        elif medium_threshold <= highest_similarity < high_threshold:
            add_to_chat_history("ASSISTANT", answers[most_similar_index])
            add_to_chat_history("ASSISTANT", "Is this the information you were looking for ?")
            st.session_state.conversation_state = CONVERSATION_STATES['AWAITING_CONFIRMATION']
            last_answered_filename = filename[most_similar_index]
        else:
                # Set the working directory to the FastChat folder
                os.chdir('/home/mane/FastChat')

                # Construct the command to run Vicuna with the USER's question
                vicuna_command = ['python3', '-m', 'fastchat.serve.cli', '--model-path', f'lmsys/vicuna-7b-v1.5']
                try:
                    # Use subprocess to run Vicuna and capture its output
                    vicuna_process = subprocess.Popen(vicuna_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    vicuna_out, vicuna_err = vicuna_process.communicate(input=prompt)

                    if vicuna_process.returncode == 0:
                        # Extract USER's question and Vicuna's answer from the output
                        lines = vicuna_out.strip().split('\n')
                        vicuna_answer = lines[0].split('ASSISTANT: ')[-1].strip()
 
                        # Display the extracted information in the desired format
                        add_to_chat_history("ASSISTANT", vicuna_answer)
                    else:
                        add_to_chat_history("ASSISTANT", f"Error while running Vicuna: {vicuna_err}")
                except Exception as e:
                    add_to_chat_history("ASSISTANT", f"Error while running Vicuna: {e}")
                    vicuna_answer = "Error while running Vicuna"



def handle_confirmation(prompt):
    if prompt.lower().strip() in ["yes", "y"]:
        add_to_chat_history("ASSISTANT", "Great! Do you have any other questions?")
        st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']
    elif prompt.lower().strip() in ["no", "n"]:
        add_to_chat_history("ASSISTANT", "Can you provide more details or clarify your question?")
        st.session_state.conversation_state = CONVERSATION_STATES['AWAITING_DETAILS']

def handle_details(prompt):
    A = prompt
    combined_embeddings, questions, answers, filename = openBert_embeddings()
    if prompt:
        detailed_input = prompt + " " + A
        detailed_input_embedding = model.encode(preprocess_text(detailed_input))
        similarities = util.cos_sim(detailed_input_embedding, combined_embeddings)[0]
        most_similar_index = similarities.argmax()
        highest_similarity = similarities[most_similar_index]

        high_threshold = 0.8
        medium_threshold = 0.5
        if highest_similarity >= high_threshold:
            add_to_chat_history("ASSISTANT", answers[most_similar_index])
        elif medium_threshold <= highest_similarity < high_threshold:
            handle_medium_confidence(similarities, prompt)
        else:
            handle_low_confidence(detailed_input)

        st.session_state.additional_input = None
        #print("\nHeres the issue")
        #st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']

def handle_medium_confidence(similarities, prompt):
    high_threshold = 0.8
    medium_threshold = 0.5
    similar_questions = [
        (questions[idx], sim_score) for idx, sim_score in enumerate(similarities)
        if medium_threshold <= sim_score < high_threshold
    ]
    similar_questions.sort(key=lambda tup: tup[1], reverse=True)
    similar_questions = similar_questions[:5]  # Take the top 5 questions

    if similar_questions:
        message = "I found a few similar questions. Does any of these match your query?\n\n"
        for i, (similar_question, _) in enumerate(similar_questions, start=1):
            message += f"{i}. {similar_question}\n"
        message += "\nPlease enter the number of the question that matches your query, or enter 'None' if none match."

        add_to_chat_history("ASSISTANT", message)
        st.session_state.similar_questions = [
            (idx, question) for idx, (question, _) in enumerate(similar_questions)
        ]
        st.session_state.conversation_state = CONVERSATION_STATES['AWAITING_NUMERIC_INPUT']

def handle_numeric_input(prompt):
    print("Entered handle_numeric_input with prompt:", prompt)  # Debug print

    if 'similar_questions' not in st.session_state:
        print("similar_questions not in session_state")  # Debug print
        add_to_chat_history("ASSISTANT", "I'm sorry, I seem to have forgotten the context. Could you please repeat your query?")
        st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']
        return

    # Debug print to check what is in similar_questions
    print("similar_questions:", st.session_state.similar_questions)

    similar_questions = st.session_state.similar_questions
    similar_questions.sort(key=lambda tup: tup[1], reverse=True)
    similar_questions = similar_questions[0:5]

    if prompt.isdigit():
        selected_number = int(prompt)
        print("selected_number:", selected_number)  # Debug print
        if 1 <= selected_number <= len(similar_questions):
            selected_question_index = similar_questions[selected_number - 1][0]
            selected_question = similar_questions[selected_question_index][1]
            ind = questions.index(selected_question)
            selected_answer = answers[ind]
            
            add_to_chat_history("User", f"Selected: {selected_question}")
            add_to_chat_history("ASSISTANT", selected_answer)
            st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']
        else:
            add_to_chat_history("ASSISTANT", "Please enter a valid number from the list.")
    else:
        add_to_chat_history("ASSISTANT", "That's not a number. Please enter the number corresponding to your question.")
    
    # Reset the state back to initial and clear the stored similar questions
    st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']
    if 'similar_questions' in st.session_state:
        del st.session_state.similar_questions
  
def handle_low_confidence(prompt):
    # Set the working directory to the FastChat folder
    os.chdir('/home/mane/FastChat')

    # Construct the command to run Vicuna with the USER's question
    vicuna_command = ['python3', '-m', 'fastchat.serve.cli', '--model-path', f'lmsys/vicuna-7b-v1.5']

    try:
        # Use subprocess to run Vicuna and capture its output
        vicuna_process = subprocess.Popen(vicuna_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        vicuna_out, vicuna_err = vicuna_process.communicate(input=prompt)

        if vicuna_process.returncode == 0:
            # Extract USER's question and Vicuna's answer from the output
            lines = vicuna_out.strip().split('\n')
            vicuna_answer = lines[0].split('ASSISTANT: ')[-1].strip()
            add_to_chat_history("ASSISTANT", vicuna_answer)
        else:
            # Handle errors while running Vicuna
            add_to_chat_history("ASSISTANT", f"Error while running Vicuna: {vicuna_err}")

    except Exception as e:
        # Handle exceptions while running Vicuna
        add_to_chat_history("ASSISTANT", f"Error while running Vicuna: {e}")

    # Reset the state back to initial
    st.session_state.conversation_state = CONVERSATION_STATES['INITIAL']

def get_user_response_to_similar_questions(similar_questions, prompt):
    if similar_questions:
        message = "I found a few similar questions. Does any of these match your query?\n\n"
        for i, (similar_question, _) in enumerate(similar_questions, start=1):
            message += f"{i}. {similar_question}\n"
        message += "\nPlease enter the number of the question that matches your query, or enter 'None' if none match."
        add_to_chat_history("ASSISTANT", message)

        # Capture user input with a unique key
        user_input = st.text_input("Your response", key="numeric_response")

        if user_input:
            handle_user_selection(user_input, similar_questions)
    else:
        # No similar questions found, handle low confidence
        handle_low_confidence(prompt)


def handle_user_selection(user_input, similar_questions, prompt):
    if user_input.lower() == 'none':
        # Handle low confidence input
        handle_low_confidence(prompt)
    elif user_input.isdigit():
        selected_index = int(user_input) - 1
        if 0 <= selected_index < len(similar_questions):
            # Display the corresponding answer
            _, selected_answer = similar_questions[selected_index]
            add_to_chat_history("ASSISTANT", selected_answer)
        else:
            # User entered an invalid number
            add_to_chat_history("ASSISTANT", "Please enter a valid number from the list.")
    else:
        # User entered an invalid response
        add_to_chat_history("ASSISTANT", "Please enter a number or 'None'.")

prompt = st.chat_input("Say something", key="user_response")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
    add_to_chat_history("USER", prompt)
    handle_interaction(prompt)

# Display chat history
display_messages()