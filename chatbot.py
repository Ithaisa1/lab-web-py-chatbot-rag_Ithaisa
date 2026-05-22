import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# LM Studio
client_llm = OpenAI(
    base_url=("http://localhost:1234/v1"),
    api_key="lm-studio"
)

# Chroma
client_chroma = chromadb.PersistentClient(path="./chroma_db")

collection = client_chroma.get_collection("documentos")

# Historial sesiones
historial = {}

SYSTEM_PROMPT = """
Eres un asistente RAG.

REGLAS:
- Responde SOLO usando el contexto proporcionado.
- Si no encuentras información relevante responde:
  'No tengo información sobre eso.'
- No inventes datos.
- Sé claro y breve.
"""

def chat(pregunta: str, session_id: str):

    # Buscar documentos similares
    resultados = collection.query(
        query_texts=[pregunta],
        n_results=3
    )

    documentos = resultados["documents"][0]
    metadatos = resultados["metadatas"][0]

    contexto = "\n\n".join(documentos)

    # Historial
    if session_id not in historial:
        historial[session_id] = []

    historial_chat = historial[session_id]

    mensajes = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Añadir historial
    mensajes.extend(historial_chat)

    # Prompt final
    mensajes.append({
        "role": "user",
        "content": f"""
CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}
"""
    })

    respuesta = client_llm.chat.completions.create(
        model="local-model",
        messages=mensajes,
        temperature=0.2
    )

    texto = respuesta.choices[0].message.content

    # Guardar historial
    historial_chat.append({
        "role": "user",
        "content": pregunta
    })

    historial_chat.append({
        "role": "assistant",
        "content": texto
    })

    fuentes = list(set([
        meta["source"] for meta in metadatos
    ]))

    return {
        "respuesta": texto,
        "fuentes": fuentes,
        "session_id": session_id,
        "fragmentos_usados": len(documentos)
    }