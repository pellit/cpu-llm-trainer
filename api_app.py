import uvicorn
import torch
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# --- 1. CONFIGURACIÓN Y CARGA DE MODELO ---
ADAPTER_PATH = "/app/LLaMA-Factory/saves/tu_modelo_entrenado"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("⏳ Iniciando carga del modelo en CPU...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Adaptadores LoRA cargados.")
except:
    print("ℹ️ Usando modelo base sin adaptadores.")

model.eval()

# --- 2. LÓGICA DE GENERACIÓN (NÚCLEO) ---
def core_generate(message: str, role_instruction: str, context_json: Dict, history: list = []):
    """
    Función pura que toma los inputs y devuelve texto.
    """
    
    # Convertir el JSON a string formateado
    json_str = json.dumps(context_json, indent=2, ensure_ascii=False)
    
    # Construir el System Prompt Dinámico
    full_system_prompt = f"""
    {role_instruction}
    
    CONTEXTO DE DATOS (JSON):
    {json_str}
    
    INSTRUCCIONES:
    1. Responde basándote EXCLUSIVAMENTE en el contexto proporcionado arriba.
    2. Si la respuesta no está en el JSON, di que no tienes esa información.
    """
    
    # Construir historial de mensajes
    messages = [{"role": "system", "content": full_system_prompt}]
    
    # Manejo de historial (API vs Gradio)
    if history and isinstance(history[0], list): # Formato Gradio
        for u, b in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})
    elif history and isinstance(history[0], dict): # Formato OpenAI
        messages.extend(history)
        
    messages.append({"role": "user", "content": message})

    # Tokenizar y Generar
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, # Aumentado para respuestas más completas
            temperature=0.3,    # Bajo para reducir alucinaciones
            do_sample=True,
            top_p=0.9
        )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- 3. DEFINICIÓN DE LA API (FastAPI) ---
app = FastAPI(title="LLM CPU API")

class ChatRequest(BaseModel):
    message: str
    role: str = "Eres un asistente útil que analiza datos JSON."
    data: Dict[str, Any] = {}
    history: List[Dict[str, str]] = []

@app.post("/v1/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_text = core_generate(
            message=req.message,
            role_instruction=req.role,
            context_json=req.data,
            history=req.history
        )
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. DEFINICIÓN DE LA UI (Gradio Actualizado) ---
def gradio_wrapper(message, history
