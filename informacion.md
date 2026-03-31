Para desplegar Gemma 2 (9B) como una API local optimizada para CPU usando Docker, la solucion mas eficiente es utilizar **Ollama**. Esta herramienta envuelve el motor de inferencia `llama.cpp` (que es el estandar para correr modelos en procesadores Intel), gestiona automaticamente la cuantizacion adecuada para tus 16 GB de RAM y expone una API REST lista para consumir.

No necesitas descargar el modelo manualmente desde una web; el contenedor se encargara de obtener la version oficial en formato GGUF directamente desde el registro de Ollama. Nota: si en el futuro necesitas el archivo crudo para otros fines, los repositorios de Hugging Face como `bartowski/gemma-2-9b-it-GGUF` son una fuente comun.

A continuacion, tienes la estructura paso a paso para automatizar el despliegue.

### Paso 1: Crear la estructura de archivos

Crea una carpeta para tu proyecto y dentro de ella genera los siguientes tres archivos:

**1. `entrypoint.sh`**
Este script iniciara el servidor interno y descargara Gemma 2 automaticamente la primera vez que levantes el contenedor.

```bash
#!/bin/bash

/bin/ollama serve &
OLLAMA_PID=$!

echo "Iniciando servidor..."
while ! curl -s http://localhost:11434 > /dev/null; do
    sleep 1
done

echo "Descargando Gemma 2 (9B)..."
ollama pull gemma2:9b

echo "Modelo listo. API activa en el puerto 11434."

wait $OLLAMA_PID
```

**2. `Dockerfile`**
Empaqueta la imagen oficial con tu script de automatizacion.

```dockerfile
FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
```

**3. `docker-compose.yml`**
Orquesta el contenedor, mapea los puertos y crea un volumen para no tener que descargar el modelo en cada reinicio.

```yaml
services:
  gemma-api:
    build: .
    container_name: gemma_cpu_api
    ports:
      - "11434:11434"
    volumes:
      - ./model_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped
```

### Paso 2: Desplegar la API

Abre tu terminal en la carpeta donde creaste los archivos y ejecuta:

```bash
docker compose up --build -d
```

Puedes ver el progreso de la descarga del modelo ejecutando `docker logs -f gemma_cpu_api`. Una vez que veas el mensaje "Modelo listo. API activa", tu entorno estara funcionando.

### Paso 3: Consumir tu nueva API

La API de Ollama tiene sus propios endpoints. Aqui tienes como hacerle consultas.

**Ejemplo basico con cURL**

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma2:9b",
  "messages": [
    {
      "role": "user",
      "content": "Escribe una funcion en Python para invertir un string."
    }
  ],
  "stream": false
}'
```

**Ejemplo integrado en Python**

```python
import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "gemma2:9b",
    "prompt": "Explica de forma concisa que es una API REST.",
    "stream": False
}

response = requests.post(url, json=payload, timeout=120)
data = response.json()

print(data["response"])
```

El motor detectara automaticamente que estas en un entorno sin GPU y ejecutara el modelo sobre CPU.

### Paso 4: Credenciales para las dos APIs FastAPI del proyecto

Si ademas vas a consumir las APIs RAG de este repo, usa estas credenciales:

| API | URL | API Key |
| :--- | :--- | :--- |
| Qwen | `http://localhost:7861/v1/chat` | `super-secreto-2025` |
| Gemma 2 | `http://localhost:7862/v1/chat` | `super-secreto-gemma2-2026` |

**Ejemplo con cURL para Qwen**

```bash
curl -X POST "http://localhost:7861/v1/chat" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: super-secreto-2025" \
     -d '{
           "message": "Hola, que dice el JSON?",
           "role": "Eres un asistente util.",
           "data": {"info": "secreta"},
           "history": []
         }'
```

**Ejemplo con cURL para Gemma 2**

```bash
curl -X POST "http://localhost:7862/v1/chat" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: super-secreto-gemma2-2026" \
     -d '{
           "message": "Que productos tienen stock?",
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

**Ejemplo en Python para cualquiera de las dos**

```python
import requests

def consultar_api(api_url, api_key, payload):
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    response = requests.post(api_url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()

payload_qwen = {
    "message": "Resume el JSON",
    "role": "Eres un asistente util.",
    "data": {"empresa": "TechSolutions"},
    "history": [],
}

payload_gemma2 = {
    "message": "Que producto tiene stock?",
    "role": "Eres un asistente de ventas.",
    "data": {"productos": [{"nombre": "Laptop Pro", "stock": 5}]},
    "history": [],
}

print(consultar_api(
    "http://localhost:7861/v1/chat",
    "super-secreto-2025",
    payload_qwen
))

print(consultar_api(
    "http://localhost:7862/v1/chat",
    "super-secreto-gemma2-2026",
    payload_gemma2
))
```
