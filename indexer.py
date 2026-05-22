import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Cliente Chroma
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documentos"
)

# Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Función chunking
def chunk_text(text, chunk_size=300):
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    
    return chunks

DOCS_PATH = "./docs"

total_docs = 0
total_chunks = 0
total_tokens = 0

for filename in os.listdir(DOCS_PATH):

    if filename.endswith(".txt"):

        total_docs += 1

        path = os.path.join(DOCS_PATH, filename)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):

            tokens = len(encoding.encode(chunk))
            total_tokens += tokens

            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "chunk_id": idx
                }],
                ids=[f"{filename}_{idx}"]
            )

            total_chunks += 1

print("\n✅ INDEXACIÓN COMPLETADA")
print(f"📄 Documentos: {total_docs}")
print(f"🧩 Chunks: {total_chunks}")
print(f"🔠 Tokens: {total_tokens}")

# Estimación fake/simple
coste = total_tokens * 0.0000001

print(f"💸 Coste estimado: ${coste:.6f}")