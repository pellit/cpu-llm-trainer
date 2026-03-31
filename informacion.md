Para desplegar Gemma 2 (9B) como una API local optimizada para CPU usando Docker, la solución más eficiente es utilizar **Ollama**. Esta herramienta envuelve el motor de inferencia `llama.cpp` (que es el estándar para correr modelos en procesadores Intel), gestiona automáticamente la cuantización adecuada para tus 16 GB de RAM y expone una API REST lista para consumir.

No necesitas descargar el modelo manualmente desde una web; el contenedor se encargará de obtener la versión oficial en formato GGUF directamente desde el registro de Ollama. *(Nota: Si en el futuro necesitas el archivo crudo para otros fines, los repositorios de Hugging Face como `bartowski/gemma-2-9b-it-GGUF` son la fuente principal).*

A continuación, tienes la estructura paso a paso para automatizar el despliegue.

### Paso 1: Crear la estructura de archivos

Crea una carpeta para tu proyecto y dentro de ella genera los siguientes tres archivos:

**1. `entrypoint.sh`**
Este script iniciará el servidor interno y descargará Gemma 2 automáticamente la primera vez que levantes el contenedor.
```bash
#!/bin/bash

# Iniciar el servidor de la API en segundo plano
/bin/ollama serve &
OLLAMA_PID=$!

# Esperar a que la API local responda
echo "Iniciando servidor..."
while ! curl -s http://localhost:11434 > /dev/null; do
    sleep 1
done

# Descargar el modelo (esto tomará unos minutos la primera vez dependiendo de tu internet)
echo "Descargando Gemma 2 (9B)..."
ollama pull gemma2:9b

echo "¡Modelo listo! API activa en el puerto 11434."

# Mantener el contenedor corriendo
wait $OLLAMA_PID
```

**2. `Dockerfile`**
Empaqueta la imagen oficial con tu script de automatización.
```dockerfile
FROM ollama/ollama:latest

# Instalar curl para la comprobación de salud en el entrypoint
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar y dar permisos al script de inicio
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Exponer el puerto estándar de la API
EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
```

**3. `docker-compose.yml`**
Orquesta el contenedor, mapea los puertos y crea un volumen para no tener que descargar el modelo (que pesa unos 5.5 GB) cada vez que reinicies.
```yaml
version: '3.8'

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
      # OLLAMA_NUM_PARALLEL=1 # Descomenta esto si notas la CPU al 100% y la PC se congela
    restart: unless-stopped
```

### Paso 2: Desplegar la API

Abre tu terminal en la carpeta donde creaste los archivos y ejecuta:

```bash
docker-compose up --build -d
```

Puedes ver el progreso de la descarga del modelo ejecutando `docker logs -f gemma_cpu_api`. Una vez que veas el mensaje *"¡Modelo listo! API activa"*, tu entorno estará funcionando.

### Paso 3: Consumir tu nueva API

La API es compatible con la estructura de OpenAI y tiene sus propios endpoints. Aquí tienes cómo hacerle consultas.

**Ejemplo básico con cURL (desde la terminal):**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma2:9b",
  "messages": [
    {
      "role": "user",
      "content": "Escribe una función en Python para invertir un string."
    }
  ],
  "stream": false
}'
```

**Ejemplo integrado en un script de Python:**
```python
import requests
import json

url = "http://localhost:11434/api/generate"

payload = {
    "model": "gemma2:9b",
    "prompt": "Explica de forma concisa qué es una API REST.",
    "stream": False # Cambia a True si quieres recibir la respuesta token por token
}

response = requests.post(url, json=payload)
data = response.json()

print(data["response"])
```

El motor detectará automáticamente que estás en un entorno sin GPU y derivará toda la carga matemática a los hilos de tu procesador Intel i5, utilizando la memoria RAM asignada para alojar el modelo cuantizado.