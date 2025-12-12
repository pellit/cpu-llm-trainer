# Usamos Python ligero
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema básicas (git y compiladores C++)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Instalar PyTorch versión CPU (Esto ahorra ~3GB de espacio y evita errores de CUDA)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Clonar LLaMA-Factory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git

WORKDIR /app/LLaMA-Factory

# 3. Instalar dependencias de LLaMA-Factory
# "metrics" añade soporte para evaluar, pero mantenemos el resto mínimo
RUN pip install -e .[metrics]

# Variable para evitar que busque GPU
ENV CUDA_VISIBLE_DEVICES=-1
ENV OMP_NUM_THREADS=8 

# Comando de inicio
CMD ["llamafactory-cli", "webui"]
