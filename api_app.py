import uvicorn
import torch
import json
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
    Se usa tanto por la API como por la UI.
    """
    
    # Convertir el JSON a string para que el LLM lo lea
    json_str = json.dumps(context_json, indent=2, ensure_ascii=False)
    
    # Construir el System Prompt Dinámico
    full_system_prompt = f"""
    {role_instruction}
    
    CONTEXTO DE DATOS (JSON):
    {json_str}
    
    Responde basándote en este contexto.
    """
    
    # Construir historial de mensajes
    messages = [{"role": "system", "content": full_system_prompt}]
    
    # Añadir historia previa (si existe)
    # Nota: para la API, history viene como lista de dicts. Para Gradio, requiere conversión.
    if history and isinstance(history[0], list): # Formato Gradio antiguo [[u,b], [u,b]]
        for u, b in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})
    elif history and isinstance(history[0], dict): # Formato OpenAI/API
        messages.extend(history)
        
    # Añadir mensaje actual
    messages.append({"role": "user", "content": message})

    # Tokenizar y Generar
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.4, # Bajo para ser fiel al JSON
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

# --- 4. DEFINICIÓN DE LA UI (Gradio) ---
def gradio_wrapper(message, history, role_input, json_input_str):
    try:
        # Intentar parsear el JSON que escribe el usuario en la UI
        json_data = json.loads(json_input_str) if json_input_str else {}
    except:
        json_data = {"error": "JSON inválido en la caja de texto"}
        
    return core_generate(message, role_input, json_data, history)

with gr.Blocks() as ui:
    gr.Markdown("## 🧪 Testeo de API LLM Local")
    
    with gr.Row():
        with gr.Column(scale=1):
            role_box = gr.Textbox(label="Rol / System Prompt", value="Eres un asistente experto en analizar estos datos.")
            json_box = gr.Code(label="Contexto JSON", language="json", value='{\n "producto": "Laptop",\n "precio": 500\n}')
        with gr.Column(scale=2):
            chat_interface = gr.ChatInterface(
                fn=gradio_wrapper,
                additional_inputs=[role_box, json_box],
                type="messages" 
            )

# Montar Gradio dentro de FastAPI en la ruta /ui
app = gr.mount_gradio_app(app, ui, path="/ui")

# --- 5. EJECUCIÓN ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
