from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from chatbot import chat, historial
import time
import re
import logging

app = FastAPI()

# Logging
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO
)

# Rate limiting
request_logs = {}

MAX_REQUESTS = 10
WINDOW = 60

# Modelo request
class ChatRequest(BaseModel):
    pregunta: str
    session_id: str

# Detectar datos personales
def contiene_datos_personales(texto):

    email_regex = r'\S+@\S+'
    nombre_regex = r'\b(me llamo|mi nombre es)\b'

    if re.search(email_regex, texto):
        return True

    if re.search(nombre_regex, texto, re.IGNORECASE):
        return True

    return False

# Middleware rate limit
@app.middleware("http")
async def rate_limit(request: Request, call_next):

    ip = request.client.host
    now = time.time()

    if ip not in request_logs:
        request_logs[ip] = []

    request_logs[ip] = [
        t for t in request_logs[ip]
        if now - t < WINDOW
    ]

    if len(request_logs[ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Demasiadas peticiones"
        )

    request_logs[ip].append(now)

    response = await call_next(request)

    return response

# Endpoint chat
@app.post("/chat")
def chatbot_endpoint(data: ChatRequest):

    # Validación longitud
    if len(data.pregunta) > 500:
        raise HTTPException(
            status_code=400,
            detail="Pregunta demasiado larga"
        )

    # Aviso privacidad
    warning = None

    if contiene_datos_personales(data.pregunta):
        warning = "⚠️ Has incluido posibles datos personales."

    # Logging
    logging.info(
        f"Session: {data.session_id}"
    )

    respuesta = chat(
        data.pregunta,
        data.session_id
    )

    respuesta["warning"] = warning

    return respuesta

# Historial
@app.get("/chat/history/{session_id}")
def get_history(session_id: str):

    return historial.get(session_id, [])

# Documentos
@app.get("/documentos")
def documentos():

    return {
        "documentos_indexados": [
            meta["source"]
            for meta in historial.values()
        ]
    }