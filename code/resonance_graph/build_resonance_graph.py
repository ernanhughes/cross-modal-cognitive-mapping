# build_resonance_graph.py
# Visualize memory resonance using network graphs

import psycopg2
import psycopg2.extras
import openai  # or your local embedding model
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# Load all memory embeddings
def load_memories():
    cursor.execute(
        "SELECT id, text_prompt, embedding FROM memories"
    )
    results = cursor.fetchall()
    ids = []
    texts = []
    embeddings = []
    for row in results:
        ids.append(row["id"])
        texts.append(row["text_prompt"])
        embeddings.append(row["embedding"])
    return ids, texts, np.array(embeddings)

# Build resonance graph
def build_graph(ids, texts, embeddings, threshold=0.8):
    G = nx.Graph()

    # Add nodes
    for idx, text in zip(ids, texts):
        G.add_node(idx, label=text)

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)

    # Add edges for pairs above the threshold
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            sim = similarities[i, j]
            if sim >= threshold:
                G.add_edge(ids[i], ids[j], weight=sim)

    return G

# Plot the graph
def plot_graph(G):
    pos = nx.spring_layout(G, k=0.5)
    labels = nx.get_node_attributes(G, 'label')
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, font_size=8, node_color='lightblue', edge_color='gray')
    plt.title("Cognitive Resonance Graph")
    plt.show()

if __name__ == "__main__":
    ids, texts, embeddings = load_memories()
    G = build_graph(ids, texts, embeddings, threshold=0.75)  # You can adjust the threshold!
    plot_graph(G)

conn.close()
