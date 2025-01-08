import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import ast
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API")
client = OpenAI(api_key=api_key)

chat_key = os.getenv("LLM_KEY")

# Load the CSV file and process embeddings
df = pd.read_csv(r"chunks.csv")
df["Embedding"] = df["Embedding"].apply(ast.literal_eval)  # Convert embedding strings to lists

# Define functions as provided
def get_embedding_openai(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def cosine_similarity(query, vector):
    query_vector = get_embedding_openai(query)  # Get the embedding for the query
    dot_product = np.dot(query_vector, vector)  # Compute the dot product
    magnitudes = np.linalg.norm(query_vector) * np.linalg.norm(vector)  # Compute the product of magnitudes
    return dot_product / magnitudes  # Return cosine similarity

def most_similar(query, n_retrieve_required=2):
    # Calculate similarity for each embedding
    df["Similarity"] = df["Embedding"].apply(lambda vector: cosine_similarity(query, vector))
    
    # Sort the dataframe by similarity in descending order
    sorted_df = df.sort_values("Similarity", ascending=False).head(n_retrieve_required)
    
    # Check if the top two similarities are below the threshold of 0.8
    if sorted_df["Similarity"].iloc[:n_retrieve_required].min() < 0.8:
        return "KABi details only", None
    
    # Combine titles and texts for the top results
    title = " ".join(sorted_df["Title"])
    text = " ".join(sorted_df["Text"])
    
    return title, text

def chat(query):
    url = chat_key
    payload = {
        "model": "llama3.1",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant for KABi Company. Only answer questions directly related to the company. For any unrelated or off-topic questions, respond with a fun and playful answer in one short sentence.",
                "role": "user",
                "content": query
            }
        ],
        "options": {"temperature": 0},
        "stream": False
    }
    response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
    return response.json()["message"]["content"]

def RAG(query):
    title, text = most_similar(query)
    prompt = f"You are an assistant for KABi Company. Only answer questions directly related to the company. For any unrelated or off-topic questions, respond with a fun and playful answer in one short sentence. Now, answer this query:\n{query}. Use this information only to answer: information:\n{text}, \n\n---if there no information answer only as 'I am sorry I am here it assissit you about KABi'"
    answer = chat(prompt)
    return answer

# Streamlit App Layout
st.set_page_config(page_title="KABi Knowledge Explorer", page_icon="💡", layout="wide")

st.title("💡 KABi Knowledge Explorer")
st.write(
    """
    Welcome to **KABi Knowledge Explorer**! 
    Ask questions about KABi Technologies and get insightful answers powered by advanced AI.
    """
)

# Sidebar with recommended questions
st.sidebar.title("🎯 Recommended Questions")
st.sidebar.write("Click on any question below to use it directly:")

# Recommendations list
recommendations = [
    "When was KABi Technologies founded, and what is its mission?",
    "Where are KABi's headquarters and global offices located?",
    "What industries does KABi primarily serve?",
    "How does KABi integrate AI to revolutionize HR?",
    "What sets KABi apart from other HR technology companies?",
    "What is KABi’s approach to empowering businesses through digital transformation?",
    "What are the core values driving KABi’s operations?",
    "What products does KABi offer, and how do they address HR challenges?",
    "How does KABi ensure its solutions are client-centric?",
    "What are the key benefits of using KABi's solutions?",
    "How does KABi support businesses in achieving cultural fit during hiring?",
    "What role does innovation play in KABi's strategy?",
    "How does KABi ensure its products remain relevant in a competitive market?",
    "What feedback has KABi received from its global client base?",
    "How can clients and partners get in touch with KABi?"
]

# Add clickable questions
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

for i, rec in enumerate(recommendations):
    if st.sidebar.button(f"Q{i+1}: {rec}"):
        st.session_state["user_query"] = rec  # Update session state with the selected question

# Main Query Section
st.subheader("🔎 Ask Your Question")
user_query = st.text_input(
    "Type your question below:",
    placeholder="E.g., How does KABi integrate AI into HR?",
    value=st.session_state["user_query"],  # Automatically populate the field
)

# Process query
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                result = RAG(user_query)
                st.success("💬 Answer:")
                st.write(result)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a valid question.")

# Add some additional styling for the app
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
    }
    .stButton > button:hover {
        background-color: #45a049;
        color: white;
    }
    .sidebar .stButton > button {
        background-color: #0073e6;
        color: white;
        border-radius: 5px;
        padding: 5px;
        font-size: 12px;
    }
    .sidebar .stButton > button:hover {
        background-color: #005bb5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
