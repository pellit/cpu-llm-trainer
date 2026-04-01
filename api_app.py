import uvicorn
import torch
import json
import numpy as np
import os
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# --- NUEVAS LIBRERÍAS PARA RAG ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURACIÓN ---
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/app/LLaMA-Factory/saves/tu_modelo_entrenado")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "all-MiniLM-L6-v2")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "")
APP_TITLE = os.getenv("APP_TITLE", "LLM RAG API Segura")
APP_PORT = int(os.getenv("APP_PORT", "7861"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# --- SEGURIDAD: Obtener clave del entorno (o usar una por defecto insegura) ---
API_KEY = os.getenv("API_KEY", "clave-segura-123")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

print(f"🔒 API Key configurada. Header requerido: {API_KEY_NAME}")
print("⏳ Iniciando carga de modelos en CPU...")

# Cargar LLM
def hf_auth_kwargs() -> Dict[str, str]:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def resolve_model_source(local_path: str, model_id: str, label: str):
    if local_path and os.path.isdir(local_path):
        print(f"Usando {label} local desde: {local_path}")
        return local_path, {}

    if local_path:
        print(f"No se encontro {label} local en {local_path}. Se intentara descargar '{model_id}'.")

    return model_id, hf_auth_kwargs()


def load_base_model():
    model_source, auth_kwargs = resolve_model_source(BASE_MODEL_PATH, BASE_MODEL_ID, "modelo base")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            **auth_kwargs
        )
        return tokenizer, model
    except OSError as exc:
        error_msg = str(exc).lower()
        if "gated repo" in error_msg or "access to model" in error_msg:
            if HF_TOKEN:
                hint = (
                    f"El token configurado en HF_TOKEN no tiene acceso al modelo '{BASE_MODEL_ID}'. "
                    "Acepta el acceso al repositorio en Hugging Face y vuelve a iniciar."
                )
            else:
                hint = (
                    f"El modelo '{BASE_MODEL_ID}' requiere autenticacion en Hugging Face. "
                    "Configura la variable HF_TOKEN con un token que tenga acceso al repositorio."
                )
            raise RuntimeError(hint) from exc
        raise


tokenizer, model = load_base_model()
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ LLM cargado.")
except:
    print("ℹ️ LLM Base cargado.")
model.eval()

# Cargar Embedder
print("⏳ Cargando motor de búsqueda...")
embedding_source, embedding_auth_kwargs = resolve_model_source(
    EMBEDDING_MODEL_PATH,
    EMBEDDING_MODEL_ID,
    "modelo de embeddings"
)
embedder = SentenceTransformer(embedding_source, device="cpu", **embedding_auth_kwargs)
print("✅ Motor de búsqueda listo.")

# --- 2. MOTOR RAG ---
class RAGEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.current_json_str = ""

    def ingest_json(self, json_data: Dict):
        json_str = json.dumps(json_data, sort_keys=True)
        if json_str == self.current_json_str and self.embeddings is not None:
            return
        
        self.current_json_str = json_str
        self.chunks = []
        
        def flatten_json(y, parent_key=''):
            for k, v in y.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_json(v, new_key)
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        self.chunks.append(f"{new_key}[{i}]: {json.dumps(item, ensure_ascii=False)}")
                else:
                    self.chunks.append(f"{new_key}: {v}")

        flatten_json(json_data)
        if not self.chunks: self.chunks = ["Sin datos."]
        
        # print(f"📚 Indexando {len(self.chunks)} trozos...") # Comentado para no ensuciar logs
        self.embeddings = embedder.encode(self.chunks, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 5) -> str:
        if self.embeddings is None or len(self.chunks) == 0: return "{}"
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = [self.chunks[i] for i in top_indices]
        return "\n".join(results)

rag = RAGEngine()

# --- 3. LÓGICA DE GENERACIÓN ---
def normalize_history(history: Optional[list]) -> List[Dict[str, str]]:
    if not history:
        return []

    normalized: List[Dict[str, str]] = []

    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            if role in {"user", "assistant", "system"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
            continue

        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, assistant_msg = entry
            if isinstance(user_msg, str):
                normalized.append({"role": "user", "content": user_msg})
            if isinstance(assistant_msg, str):
                normalized.append({"role": "assistant", "content": assistant_msg})

    return normalized[-4:]


def core_generate(message: str, role_instruction: str, context_json: Dict, history: Optional[list] = None):
    rag.ingest_json(context_json)
    relevant_context = rag.retrieve(message, top_k=7)
    
    full_system_prompt = f"""
    {role_instruction}
    
    INFORMACIÓN RELEVANTE (Fragmentos):
    ---
    {relevant_context}
    ---
    
    INSTRUCCIONES:
    1. Responde usando SOLO la información de arriba.
    2. Si no sabes, dilo.
    """
    
    messages = [{"role": "system", "content": full_system_prompt}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    if inputs.input_ids.shape[1] > 4096:
        return "❌ Error: Contexto demasiado largo."

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

# --- 4. API SEGURA (FastAPI) ---
app = FastAPI(title=APP_TITLE)

# Función de dependencia para validar la clave
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Credenciales inválidas. Falta header X-API-Key"
    )

class ChatRequest(BaseModel):
    message: str
    role: str = "Eres un asistente útil."
    data: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)

@app.post("/v1/chat")
# Inyectamos la seguridad aquí:
async def chat_endpoint(req: ChatRequest, api_key: str = Security(get_api_key)):
    try:
        return {"response": core_generate(req.message, req.role, req.data, req.history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. UI (Gradio) ---
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

with gr.Blocks() as ui:
    gr.Markdown("# 🔐 Chat RAG Seguro")
    with gr.Row():
        with gr.Column(scale=1):
            role_box = gr.Textbox(label="Rol", value="Eres un analista experto.")
            with gr.Tabs():
                with gr.TabItem("📁 Archivo"): file_box = gr.File(label="JSON", file_types=[".json"], type="filepath")
                with gr.TabItem("📝 Texto"): json_box = gr.Code(label="JSON", language="json", value='{}')
        with gr.Column(scale=2):
            gr.ChatInterface(fn=gradio_wrapper, additional_inputs=[role_box, json_box, file_box])

# Opcional: Proteger también la UI con usuario/pass
# app = gr.mount_gradio_app(app, ui, path="/ui", auth=("admin", "admin123"))
app = gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
