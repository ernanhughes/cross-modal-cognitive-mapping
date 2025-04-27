# insert_memory.py
# Insert user memory (text + embedding) into PostgreSQL with pgvector

import psycopg2
import psycopg2.extras
import openai  # or your local embedding model
import os
import torch
import ollama


from dotenv import load_dotenv
load_dotenv()


# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "cognitive_memory")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
cursor = conn.cursor()


def get_embedding(text):
    """Generates an embedding using Ollama"""
    try:
        embedding_data = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return torch.tensor(embedding_data["embedding"], dtype=torch.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Insert memory into database
def insert_memory(user_id, text_prompt):
    embedding = get_embedding(text_prompt)
    cursor.execute(
        """
        INSERT INTO memories (user_id, text_prompt, embedding)
        VALUES (%s, %s, %s)
        """,
        (user_id, text_prompt, embedding)
    )
    conn.commit()
    print("Memory inserted successfully.")

if __name__ == "__main__":
    user_id = 1
    text_prompt = "A peaceful home by the sea."
    insert_memory(user_id, text_prompt)

conn.close()
