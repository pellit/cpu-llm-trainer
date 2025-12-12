
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# --- CONFIGURACIÓN ---
# La ruta donde guardaste tu modelo al exportar en LLaMA-Factory
# Si solo guardaste adaptadores, apunta a la carpeta del checkpoint (ej. saves/Qwen.../lora/checkpoint-100)
ADAPTER_PATH = "/app/LLaMA-Factory/saves/tu_modelo_entrenado" 
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" # Debe ser el mismo que usaste para entrenar

print("⏳ Cargando modelo en CPU... paciencia...")

# 1. Cargar Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# 2. Cargar Modelo Base
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32, # CPU prefiere float32
    device_map="cpu",
    low_cpu_mem_usage=True
)

# 3. Cargar tus cambios entrenados (Si existen)
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Adaptadores LoRA cargados correctamente.")
except Exception as e:
    print(f"ℹ️ No se encontraron adaptadores o ruta incorrecta, usando modelo base. Error: {e}")

model.eval() # Modo evaluación

# --- LÓGICA DEL CHAT ---
def generate_response(message, history):
    # Formatear el prompt como chat (User/Assistant)
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Aplicar la plantilla de chat del modelo (importante para que entienda instrucciones)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    # Generar respuesta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128, # Pocos tokens para que sea rápido en CPU
            temperature=0.7,
            do_sample=True
        )

    # Decodificar solo la parte nueva
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- INTERFAZ GRADIO ---
demo = gr.ChatInterface(
    fn=generate_response,
    title="🤖 Mi LLM Local (CPU)",
    description="Probando el modelo Qwen entrenado localmente.",
    examples=["¿Qué es Docker?", "Explícame la relatividad de forma simple."],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
