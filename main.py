from openai import OpenAI
import requests
import pandas as pd
from dotenv import load_dotenv
import os
import json
import numpy as np
import ast

# Load variables from .env file
load_dotenv()

# Access variables
api_key = os.getenv("OPENAI_API")

client = OpenAI(api_key=api_key)

chat_key = os.getenv("LLM_KEY")

df = pd.read_csv(r"chunks.csv")
df["Embedding"] = df["Embedding"].apply(ast.literal_eval) # get embedding as list

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

def most_similar(query, n_retrive_required=2):
    df["Similarity"] = df["Embedding"].apply(lambda vector: cosine_similarity(query, vector))
    sorted_df = df.sort_values("Similarity", ascending=False).head(n_retrive_required)  # Get the top 4 rows
    title = " ".join(sorted_df["Title"])  # Combine titles with a space
    text = " ".join(sorted_df["Text"])    # Combine texts with a space
    return title, text

def chat(query):
  url = chat_key
  payload = {
    "model": "llama3.1",
    "messages": [
      {
        "role": "user",
        "content": query
      }
    ],
    "options":{"temperature":0},
    "stream": False
  }
  response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
  return response.json()["message"]["content"]


def RAG(query):
  # get most similer
  title, text = most_similar(query)

  # prompt
  prompt = f"answer this query:\n{query}. Use this information only to answer :\n{text}"

  # get answer
  answer = chat(prompt)

  return answer

query_1 = "who KABi?"
print(RAG(query_1))