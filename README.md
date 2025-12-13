# cpu-llm-trainer

Para conectarte a tu nueva API con RAG, necesitas enviar una petición **POST** al endpoint que definimos en FastAPI (`/v1/chat`).

Aquí tienes la documentación técnica para consumirla desde diferentes entornos.

### 1\. Datos de Conexión

  * **URL Base:** `http://<IP-DE-TU-SERVIDOR>:7861`
  * **Endpoint:** `/v1/chat`
  * **Método:** `POST`
  * **Headers:** `Content-Type: application/json`

-----

### 2\. Ejemplo en Python (Para integrar en tu Backend)

Este es el script que usarías si estás construyendo otra app que consulta a esta IA.

```python
import requests
import json

# Configuración
API_URL = "http://localhost:7861/v1/chat"  # Cambia localhost por la IP de tu servidor si es remoto

# 1. El JSON sobre el que quieres preguntar (Tu Contexto)
datos_contexto = {
    "empresa": "TechSolutions",
    "politicas": {
        "devoluciones": "30 días sin costo",
        "envios": "Gratis en pedidos mayores a $50"
    },
    "productos": [
        {"id": 1, "nombre": "Laptop Pro", "precio": 1200, "stock": 5},
        {"id": 2, "nombre": "Mouse Gamer", "precio": 25, "stock": 0}
    ]
}

# 2. El cuerpo de la petición
payload = {
    "message": "¿Tienen stock del mouse gamer y cuánto cuesta?",
    "role": "Eres un asistente de ventas amable.",
    "data": datos_contexto,
    "history": [] # Opcional: Historial previo si es una conversación continua
}

# 3. Enviar la petición
try:
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        respuesta_ai = response.json()
        print("🤖 Respuesta:", respuesta_ai["response"])
    else:
        print("❌ Error:", response.text)
        
except Exception as e:
    print(f"Error de conexión: {e}")
```

-----

### 3\. Ejemplo con cURL (Terminal / Bash)

Útil para probar rápido desde la línea de comandos de tu servidor o computadora.

```bash
curl -X POST "http://localhost:7861/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Analiza estos datos y dime qué servicio es el más caro",
           "role": "Eres un analista financiero.",
           "data": {
             "servicios": [
               {"nombre": "Consultoría", "costo": 500},
               {"nombre": "Desarrollo", "costo": 1500},
               {"nombre": "Soporte", "costo": 200}
             ]
           }
         }'
```

-----

### 4\. Estructura del JSON (Payload)

El cuerpo que envíes **debe** respetar esta estructura (definida en tu `ChatRequest` de Pydantic):

| Campo | Tipo | Obligatorio | Descripción |
| :--- | :--- | :--- | :--- |
| `message` | `string` | **Sí** | La pregunta del usuario. |
| `data` | `dict` (JSON) | No | El JSON completo que el modelo usará como base de conocimiento (RAG). |
| `role` | `string` | No | Instrucción de comportamiento ("Eres un experto en..."). Default: Asistente útil. |
| `history` | `list` | No | Historial de chat previo `[{ "role": "user", "content": "hola" }, ...]` |

-----

### 5\. Tip Pro: Documentación Automática (Swagger UI)

Como usamos **FastAPI**, tienes una documentación interactiva generada automáticamente donde puedes probar la API sin escribir código.

1.  Abre tu navegador.
2.  Entra a: **`http://<TU_IP>:7861/docs`**
3.  Verás una interfaz azul (Swagger UI).
4.  Busca el endpoint `/v1/chat`, dale a "Try it out", pega el JSON y ejecuta.
