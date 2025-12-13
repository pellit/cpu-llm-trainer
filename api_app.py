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

# --- CONFIGURACIÓN ---
ADAPTER_PATH = "/app/LLaMA-Factory/saves/tu_modelo_entrenado"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("⏳ Iniciando carga del modelo en CPU...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16,  # Optimizado para RAM (bfloat16)
    device_map="cpu",
    low_cpu_mem_usage=True
)

try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Adaptadores LoRA cargados.")
except:
    print("ℹ️ Usando modelo base sin adaptadores.")

model.eval()

# --- LÓGICA DE GENERACIÓN ---
def core_generate(message: str, role_instruction: str, context_json: Dict, history: list = []):
    json_str = json.dumps(context_json, indent=2, ensure_ascii=False)
    
    full_system_prompt = f"""
    {role_instruction}
    
    CONTEXTO DE DATOS (JSON):
    {json_str}
    
    INSTRUCCIONES:
    1. Responde basándote EXCLUSIVAMENTE en el contexto proporcionado.
    2. Si la respuesta no está en el JSON, di que no tienes esa información.
    """
    
    messages = [{"role": "system", "content": full_system_prompt}]
    
    if history and isinstance(history[0], list): 
        for u, b in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})
    elif history and isinstance(history[0], dict):
        messages.extend(history)
        
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- API FASTAPI ---
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

# --- UI GRADIO ---
def gradio_wrapper(message, history, role_input, json_text, json_file):
    final_json = {}
    try:
        if json_file is not None:
            with open(json_file, 'r', encoding='utf-8') as f:
                final_json = json.load(f)
        elif json_text.strip():
            final_json = json.loads(json_text)
        else:
            final_json = {"aviso": "No se cargaron datos JSON"}
    except Exception as e:
        return f"❌ Error procesando JSON: {str(e)}"
        
    return core_generate(message, role_input, final_json, history)

with gr.Blocks(theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 🧠 Chat con tus Datos (JSON)")
    
    with gr.Row():
        with gr.Column(scale=1):
            role_box = gr.Textbox(label="Rol", value="Eres un analista experto.", lines=2)
            with gr.Tabs():
                with gr.TabItem("📁 Subir Archivo"):
                    file_box = gr.File(label="Cargar JSON", file_types=[".json"], type="filepath")
                with gr.TabItem("📝 Pegar Texto"):
                    json_box = gr.Code(label="Editor JSON", language="json", value='{}')

        with gr.Column(scale=2):
            chat_interface = gr.ChatInterface(
                fn=gradio_wrapper,
                additional_inputs=[role_box, json_box, file_box],
                type="messages"
            )

app = gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
