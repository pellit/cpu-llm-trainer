
import gradio as gr
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURACIÓN ---
ADAPTER_PATH = "/app/LLaMA-Factory/saves/tu_modelo_entrenado"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
JSON_PATH = "/app/data.json"  # Ruta donde montaremos el JSON en Docker

print("⏳ Cargando modelo y datos...")

# 1. Cargar el JSON de contexto
try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        print("✅ JSON cargado correctamente.")
except Exception as e:
    print(f"⚠️ Error cargando JSON: {e}")
    json_str = "{}"

# 2. Definir el System Prompt (La "Personalidad" y el Contexto)
SYSTEM_PROMPT = f"""
Eres un asistente de atención al cliente útil y amable.
Tu objetivo es responder preguntas basándote ESTRICTAMENTE en la siguiente información en formato JSON.
Si la respuesta no está en el JSON, di amablemente que no tienes esa información.

INFORMACIÓN DE CONTEXTO:
{json_str}
"""

# 3. Cargar Tokenizer y Modelo
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

# --- LÓGICA DEL CHAT ---
def generate_response(message, history):
    # 1. Construir la lista de mensajes
    # IMPORTANTE: El mensaje del sistema va PRIMERO
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 2. Añadir historia del chat (Gradio ahora usa formato lista de dicts con type="messages")
    messages.extend(history)
    
    # 3. Añadir el mensaje actual del usuario
    messages.append({"role": "user", "content": message})

    # 4. Aplicar plantilla de chat
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256, # Un poco más de espacio para responder
            temperature=0.5,    # Más bajo para que sea más fiel al JSON (menos alucinación)
            do_sample=True,
            top_p=0.9
        )

    # Decodificar solo la respuesta nueva
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- INTERFAZ GRADIO ---
demo = gr.ChatInterface(
    fn=generate_response,
    type="messages",  # Formato moderno de Gradio
    title="🤖 Asistente con Contexto JSON",
    description=f"Este asistente responde preguntas sobre: {json_str[:100]}...",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
