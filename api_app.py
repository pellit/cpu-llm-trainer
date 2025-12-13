import uvicorn
import torch
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# --- NUEVAS LIBRERÍAS PARA RAG ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURACIÓN ---
ADAPTER_PATH = "/app/LLaMA-Factory/saves/tu_modelo_entrenado"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2" # Modelo diminuto y rápido para buscar

print("⏳ Iniciando carga de modelos en CPU...")

# Cargar LLM (bfloat16 para ahorrar RAM)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16, 
    device_map="cpu",
    low_cpu_mem_usage=True
)
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ LLM cargado.")
except:
    print("ℹ️ LLM Base cargado.")
model.eval()

# Cargar Modelo de Embeddings (para buscar en el JSON)
print("⏳ Cargando motor de búsqueda...")
embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device="cpu")
print("✅ Motor de búsqueda listo.")

# --- 2. CLASE DE AYUDA PARA RAG (CHUNKING) ---
class RAGEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.current_json_str = ""

    def ingest_json(self, json_data: Dict):
        """Convierte el JSON en trozos buscables"""
        json_str = json.dumps(json_data, sort_keys=True)
        
        # Si el JSON no cambió, no re-calculamos (ahorra CPU)
        if json_str == self.current_json_str and self.embeddings is not None:
            return
        
        self.current_json_str = json_str
        self.chunks = []
        
        # ESTRATEGIA DE CHUNKING: Aplanar el JSON
        # Convertimos {"usuario": {"nombre": "Juan"}} en "usuario -> nombre: Juan"
        def flatten_json(y, parent_key=''):
            for k, v in y.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_json(v, new_key)
                elif isinstance(v, list):
                    # Para listas, guardamos cada elemento como un chunk
                    for i, item in enumerate(v):
                        self.chunks.append(f"{new_key}[{i}]: {json.dumps(item, ensure_ascii=False)}")
                else:
                    self.chunks.append(f"{new_key}: {v}")

        flatten_json(json_data)
        
        if not self.chunks:
            self.chunks = ["Sin datos disponibles."]

        print(f"📚 Indexando {len(self.chunks)} trozos de información...")
        self.embeddings = embedder.encode(self.chunks, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Busca los trozos más relevantes para la pregunta"""
        if self.embeddings is None or len(self.chunks) == 0:
            return "{}"
            
        # Crear embedding de la pregunta
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        
        # Calcular similitud
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Obtener los top_k índices más altos
        # Si hay pocos chunks, devolver todos
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [self.chunks[i] for i in top_indices]
        return "\n".join(results)

# Instancia global del motor RAG
rag = RAGEngine()

# --- 3. LÓGICA DE GENERACIÓN ---
def core_generate(message: str, role_instruction: str, context_json: Dict, history: list = []):
    
    # 1. INGESTIÓN: Procesar JSON (Solo si cambió)
    rag.ingest_json(context_json)
    
    # 2. RETRIEVAL: Buscar solo lo relevante para esta pregunta
    # Buscamos los 7 trozos más relevantes
    relevant_context = rag.retrieve(message, top_k=7)
    
    # 3. Prompt Augmentado
    full_system_prompt = f"""
    {role_instruction}
    
    INFORMACIÓN RELEVANTE ENCONTRADA (Fragmentos del JSON):
    ---
    {relevant_context}
    ---
    
    INSTRUCCIONES:
    1. Responde usando SOLO la información de arriba.
    2. Si la respuesta no está en los fragmentos, di que no sabes.
    """
    
    messages = [{"role": "system", "content": full_system_prompt}]
    
    # Historial limitado
    if history:
        if isinstance(history[0], list): 
            messages.extend([{"role": "user", "content": u}, {"role": "assistant", "content": b}] for u,b in history[-2:])
        elif isinstance(history[0], dict):
            messages.extend(history[-4:])
        
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    # Seguridad extra
    if inputs.input_ids.shape[1] > 4096:
        return "❌ Error: Pregunta demasiado compleja para la memoria actual."

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

# --- 4. API & UI ---
app = FastAPI(title="LLM RAG API")

class ChatRequest(BaseModel):
    message: str
    role: str = "Eres un asistente útil."
    data: Dict[str, Any] = {}
    history: List[Dict[str, str]] = []

@app.post("/v1/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        return {"response": core_generate(req.message, req.role, req.data, req.history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def gradio_wrapper(message, history, role_input, json_text, json_file):
    final_json = {}
    try:
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f: final_json = json.load(f)
        elif json_text.strip():
            final_json = json.loads(json_text)
    except:
        return "❌ JSON Inválido"
        
    return core_generate(message, role_input, final_json, history)

with gr.Blocks(theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 🧠 Chat Inteligente (RAG Vectorial)")
    with gr.Row():
        with gr.Column(scale=1):
            role_box = gr.Textbox(label="Rol", value="Eres un analista experto.")
            with gr.Tabs():
                with gr.TabItem("📁 Archivo"): file_box = gr.File(label="JSON", file_types=[".json"], type="filepath")
                with gr.TabItem("📝 Texto"): json_box = gr.Code(label="JSON", language="json", value='{}')
        with gr.Column(scale=2):
            gr.ChatInterface(fn=gradio_wrapper, additional_inputs=[role_box, json_box, file_box], type="messages")

app = gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
