import json
import torch
import streamlit as st
from transformers import BertTokenizer, BertModel
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
with open('Model\modified_faq_data.json', 'r') as json_file:
    data = json.load(json_file)

# Streamlit UI
st.title("MANE Chatbot")
st.text("Hello! I am Mane Bot. How can I help you today?")
user_image = "image.png"
chatbot_image = "chatbot.png"
chat_history = st.empty()
if 'chat_history_list' not in st.session_state:
    st.session_state.chat_history_list = []

user_question = st.text_input("Ask a question:")

if user_question:
    st.session_state.typing_message = st.text("Bot is typing...")
    time.sleep(2)

    # Tokenize and encode the user's question
    user_question_tokens = tokenizer.encode(user_question, truncation=True, max_length=512, return_tensors='pt')
    user_question_embedding = model(input_ids=user_question_tokens).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity between the user's question and all questions in the JSON file
    similarities = []
    for item in data:
        question_tokens = tokenizer.encode(item['question'], truncation=True, max_length=512, return_tensors='pt')
        question_embedding = model(input_ids=question_tokens).last_hidden_state.mean(dim=1)

        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(user_question_embedding, question_embedding)
        similarities.append(cosine_sim.item())

    # Find the index of the question with the highest cosine similarity
    max_index = similarities.index(max(similarities))

    most_similar_answer = data[max_index]['answer']

    st.session_state.chat_history_list.append(("User", user_question))
    st.session_state.chat_history_list.append(("Chatbot", most_similar_answer))

    st.session_state.typing_message.empty()

for q, a in st.session_state.chat_history_list:
    with st.container():
        if q == "User":
            st.image(user_image, width=30, use_column_width=False, output_format="png")
            st.markdown(f"**{q}:** {a}")
        else:
            st.image(chatbot_image, width=30, use_column_width=False, output_format="png")
            st.markdown(f"**{q}:** {a}")

