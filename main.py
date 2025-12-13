import streamlit as st
from src.chatbot import EcommerceChatbot

st.set_page_config(page_title="E-commerce Chatbot", layout="wide")

st.title("🛒 E-commerce Chatbot Demo")

if 'bot' not in st.session_state:
    st.session_state['bot'] = EcommerceChatbot()

user_input = st.text_input("Ask me about products, prices, or place an order:")

if user_input:
    response = st.session_state['bot'].handle_message(user_input)
    st.markdown(f"**Bot:** {response}")
