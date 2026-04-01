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

### 6. Cómo conectarse ahora (Con seguridad)
Ahora, si intentas usar el comando de antes, te dará error 403 Forbidden. Debes agregar el header X-API-Key.

### 6.1 Ejemplo con Python (Script Cliente)
#Python

import requests

API_URL = "http://38.51.69.71:7861/v1/chat"
MY_KEY = "super-secreto-2025"  # La misma que pusiste en docker-compose

payload = {
    "message": "Hola, ¿qué dice el JSON?",
    "data": {"info": "secreta"}
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": MY_KEY  # <--- AQUÍ VA LA CLAVE
}

try:
    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        print(response.json())
    else:
        print("Error:", response.status_code, response.text)
except Exception as e:
    print("Error de conexión")
    
### 6.1.2 Ejemplo con cURL
#Bash

curl -X POST "http://38.51.69.71:7861/v1/chat" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: super-secreto-2025" \
     -d '{"message": "Hola", "data": {}}'

-----

### 7\. Credenciales y ejemplos de ambos casos

Si tienes las dos APIs levantadas, estas son las credenciales:

| API | URL Base | Endpoint | API Key |
| :--- | :--- | :--- | :--- |
| Qwen | `http://38.51.69.71:7861` | `/v1/chat` | `super-secreto-2025` |
| Gemma 2 | `http://38.51.69.71:7862` | `/v1/chat` | `super-secreto-gemma2-2026` |

#### 7.0 Requisito extra para Gemma 2

El modelo `google/gemma-2-2b-it` es un repositorio restringido en Hugging Face. Antes de levantar `docker-compose.gemma2.yml` debes:

```bash
export HF_TOKEN=tu_token_de_hugging_face
```

Ese token debe pertenecer a una cuenta con acceso aprobado al repositorio `google/gemma-2-2b-it`. Si no lo haces, la API fallarÃ¡ al iniciar al descargar `config.json`.

#### 7.0.1 Descargar Gemma 2 a disco local y reutilizarla

La API de Gemma 2 ahora prioriza el uso de modelos locales montados en `./models`. Descarga una vez y luego el contenedor cargarÃƒÆ’Ã‚Â¡ desde disco:

```powershell
$env:HF_TOKEN = "hf_tu_token"
docker compose -f docker-compose.gemma2.yml run --rm inference-api-gemma2 python /app/download_models.py
```

Eso descarga:

- `google/gemma-2-2b-it` en `./models/gemma-2-2b-it`
- `all-MiniLM-L6-v2` en `./models/all-MiniLM-L6-v2`

Luego puedes iniciar la API normal:

```powershell
docker compose -f docker-compose.gemma2.yml up --build
```

Si `./models/gemma-2-2b-it` existe, la API usarÃƒÆ’Ã‚Â¡ esa copia local antes de intentar ir a Hugging Face.

#### 7.1 cURL para Qwen

```bash
curl -X POST "http://38.51.69.71:7861/v1/chat" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: super-secreto-2025" \
     -d '{
           "message": "Resume el JSON",
           "role": "Eres un asistente útil.",
           "data": {
             "empresa": "TechSolutions",
             "estado": "activo"
           },
           "history": []
         }'
```

#### 7.2 cURL para Gemma 2

```bash
curl -X POST "http://38.51.69.71:7862/v1/chat" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: super-secreto-gemma2-2026" \
     -d '{
           "message": "¿Qué productos tienen stock?",
           "role": "Eres un asistente de ventas.",
           "data": {
             "productos": [
               {"nombre": "Laptop Pro", "stock": 5},
               {"nombre": "Mouse Gamer", "stock": 0}
             ]
           },
           "history": []
         }'
```

#### 7.3 Python reutilizable para cualquiera de las dos

```python
import requests

def consultar_api(api_url, api_key, message, data):
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    payload = {
        "message": message,
        "role": "Eres un asistente útil.",
        "data": data,
        "history": [],
    }
    response = requests.post(api_url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()

respuesta_qwen = consultar_api(
    "http://38.51.69.71:7861/v1/chat",
    "super-secreto-2025",
    "Resume el JSON",
    {"empresa": "TechSolutions"}
)

respuesta_gemma2 = consultar_api(
    "http://38.51.69.71:7862/v1/chat",
    "super-secreto-gemma2-2026",
    "¿Qué producto tiene stock?",
    {"productos": [{"nombre": "Laptop Pro", "stock": 5}]}
)

print(respuesta_qwen)
print(respuesta_gemma2)
```

#### 7.4 Swagger UI

  * Qwen: `http://38.51.69.71:7861/docs`
  * Gemma 2: `http://38.51.69.71:7862/docs`


https://huggingface.co/google/gemma-2-2b-it
