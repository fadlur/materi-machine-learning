"""
=============================================================
FASE 8 — MODUL 3: LLM ENGINEERING
=============================================================
LLM (Large Language Models) adalah skill #1 yang dicari di 2025-2026.
Untuk AI Engineer roles, LLM engineering adalah REQUIREMENT, bukan opsional.

Background backend + EE kamu sangat cocok untuk:
- LLM deployment & serving infrastructure
- RAG system architecture (database + API + caching)
- LLM agents & tool integration
- Cost optimization & monitoring

Durasi target: 5-7 hari
=============================================================
"""

import numpy as np

# ===========================================================
# 📖 BAGIAN 1: LLM Landscape 2025-2026
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     LLM LANDSCAPE FOR ENGINEERS                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  MODEL TIERS:                                            ║
║  ┌─────────────────────────────────────────────┐         ║
║  │ TIER 1: Frontier Models                     │         ║
║  │ GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro   │         ║
║  │ Use case: Complex reasoning, production apps │        ║
║  │ Cost: $$$, Latency: medium                  │         ║
║  ├─────────────────────────────────────────────┤         ║
║  │ TIER 2: Capable Open Models                 │         ║
║  │ Llama 3, Mistral, Qwen, DeepSeek           │         ║
║  │ Use case: Self-hosted, fine-tuned, cost-sensitive │    ║
║  │ Cost: $ (hosting), Latency: customizable    │         ║
║  ├─────────────────────────────────────────────┤         ║
║  │ TIER 3: Small & Fast                        │         ║
║  │ Phi-3, Gemma, Llama 3.2 1B/3B               │         ║
║  │ Use case: Edge deployment, simple tasks     │         ║
║  │ Cost: $, Latency: very fast                 │         ║
║  └─────────────────────────────────────────────┘         ║
║                                                          ║
║  KEY CONCEPTS FOR ENGINEERS:                             ║
║  1. Context Window — max tokens LLM bisa proses          ║
║  2. Tokenization — cara text dipecah jadi tokens         ║
║  3. Temperature — randomness (0=predictable, 1=creative) ║
║  4. System Prompt — instruksi global untuk LLM           ║
║  5. Function Calling — LLM panggil external tools        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 2: Prompt Engineering
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     PROMPT ENGINEERING PATTERNS                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. ZERO-SHOT                                            ║
║     "Classify this text as positive or negative: [text]" ║
║                                                          ║
║  2. FEW-SHOT                                             ║
║     "Here are examples:                                  ║
║      Text: 'Great product!' → Positive                   ║
║      Text: 'Terrible service' → Negative                 ║
║      Text: '[text]' → "                                  ║
║                                                          ║
║  3. CHAIN-OF-THOUGHT                                     ║
║     "Think step by step before answering."               ║
║     → LLM generates reasoning, then answer               ║
║                                                          ║
║  4. STRUCTURED OUTPUT                                    ║
║     "Respond in JSON format with keys: summary, sentiment║
║      confidence_score"                                   ║
║                                                          ║
║  5. SYSTEM PROMPT PATTERN                                ║
║     system = "You are an expert electrical engineer.     ║
║     Provide concise, technically accurate answers."      ║
║                                                          ║
║  ENGINEERING TIPS:                                       ║
║  - Be specific ("concise" vs "detailed")                 ║
║  - Use delimiters (###, """, XML tags)                   ║
║  - Specify output format explicitly                      ║
║  - Include constraints ("max 100 words")                 ║
║  - Version your prompts (track in git!)                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 3: RAG (Retrieval-Augmented Generation)
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     RAG SYSTEM ARCHITECTURE                              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Kenapa RAG?                                             ║
║  - LLM tidak punya knowledge spesifik domain             ║
║  - LLM bisa hallucinate (buat fakta palsu)               ║
║  - RAG = kasih context yang relevan sebelum generate     ║
║                                                          ║
║  ┌─────────────────────────────────────────────┐         ║
║  │           INGESTION PIPELINE                │         ║
║  │                                             │         ║
║  │  Documents ──▶ Chunking ──▶ Embedding     │         ║
║  │     │           │            │              │         ║
║  │     ▼           ▼            ▼              │         ║
║  │  PDF,TXT    500-1000    Vector (768-dim)  │         ║
║  │  Markdown   tokens      OpenAI/Sentence   │         ║
║  │  API docs               Transformers      │         ║
║  │                              │              │         ║
║  │                              ▼              │         ║
║  │                      ┌─────────────┐       │         ║
║  │                      │  Vector DB  │       │         ║
║  │                      │  ChromaDB   │       │         ║
║  │                      │  Pinecone   │       │         ║
║  │                      │  Weaviate   │       │         ║
║  │                      └─────────────┘       │         ║
║  └─────────────────────────────────────────────┘         ║
║                     │                                    ║
║  ┌──────────────────┼──────────────────────────┐         ║
║  │           QUERY PIPELINE                     │         ║
║  │                                              │         ║
║  │  User Query ──▶ Embedding ──▶ Similarity    │         ║
║  │     │              │           Search        │         ║
║  │     │              │               │         │         ║
║  │     │              │               ▼         │         ║
║  │     │              │        Top-k Chunks     │         ║
║  │     │              │               │         │         ║
║  │     └──────────────┼───────────────┘         │         ║
║  │                    ▼                          │         ║
║  │         ┌─────────────────┐                  │         ║
║  │         │  Augmented      │                  │         ║
║  │         │  Prompt         │                  │         ║
║  │         │                 │                  │         ║
║  │         │ Context: [chunks]│                 │         ║
║  │         │ Question: [query]│                 │         ║
║  │         │ Answer:          │                 │         ║
║  │         └────────┬────────┘                  │         ║
║  │                  │                           │         ║
║  │                  ▼                           │         ║
║  │              LLM Generate                    │         ║
║  │                  │                           │         ║
║  │                  ▼                           │         ║
║  │         Response + Source Citation           │         ║
║  └──────────────────────────────────────────────┘         ║
║                                                          ║
║  ADVANCED RAG TECHNIQUES:                                ║
║  - Re-ranking: cross-encoder untuk ranking ulang hasil   ║
║  - Hybrid search: vector + keyword (BM25)                ║
║  - Query expansion: generate synonym queries             ║
║  - Multi-modal RAG: images + text                        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 💻 CONTOH: Simple RAG Implementation
# ===========================================================

class SimpleRAG:
    """
    RAG system sederhana untuk memahami konsep.
    
    Di production, gunakan:
    - LangChain / LlamaIndex untuk orchestration
    - ChromaDB / Pinecone untuk vector store
    - OpenAI / Sentence-Transformers untuk embeddings
    """
    
    def __init__(self):
        self.documents = []      # Raw documents
        self.chunks = []         # Document chunks
        self.embeddings = []     # Vector embeddings
        self.chunk_sources = []  # Source mapping
    
    def chunk_document(self, text, chunk_size=500, overlap=50):
        """
        Split document into overlapping chunks.
        
        Strategy: Sliding window dengan overlap.
        Ini penting supaya context tidak terputus di tengah.
        """
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def simple_embedding(self, text):
        """
        Mock embedding function.
        
        Di production, gunakan:
        - openai.Embedding.create()
        - sentence_transformers.SentenceTransformer.encode()
        """
        # Simple hash-based embedding untuk demo
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(128)
    
    def cosine_similarity(self, a, b):
        """Compute cosine similarity antara dua vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    def add_documents(self, documents):
        """
        Add documents ke RAG system.
        
        Documents: list of dict {'id': str, 'content': str, 'metadata': dict}
        """
        for doc in documents:
            self.documents.append(doc)
            doc_chunks = self.chunk_document(doc['content'])
            
            for chunk in doc_chunks:
                self.chunks.append(chunk)
                self.embeddings.append(self.simple_embedding(chunk))
                self.chunk_sources.append({
                    'doc_id': doc['id'],
                    'metadata': doc.get('metadata', {})
                })
        
        print(f"📚 Added {len(documents)} documents, {len(self.chunks)} chunks")
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k relevant chunks untuk query.
        
        Returns:
        --------
        list of dict: {'chunk': str, 'score': float, 'source': dict}
        """
        query_embedding = self.simple_embedding(query)
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            results.append({
                'chunk': self.chunks[idx],
                'score': score,
                'source': self.chunk_sources[idx]
            })
        
        return results
    
    def generate_prompt(self, query, retrieved_chunks):
        """Generate augmented prompt dengan context."""
        context = "\n\n".join([
            f"[Document {i+1}] {r['chunk'][:200]}..."
            for i, r in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""Based on the following documents, answer the question.
If the answer is not in the documents, say "I don't have enough information."

Documents:
{context}

Question: {query}

Answer:"""
        
        return prompt


# ===========================================================
# 💻 DEMO: RAG System
# ===========================================================

print("\n" + "="*50)
print("DEMO: Simple RAG System")
print("="*50)

# Buat RAG system
rag = SimpleRAG()

# Tambahkan documents (simulasi manual EE)
docs = [
    {
        'id': 'power_quality_101',
        'content': """
        Power quality refers to the stability and purity of electrical power.
        Common power quality issues include voltage sags, swells, harmonics,
        and flicker. Voltage sags are short-duration reductions in voltage,
        typically caused by large loads starting or faults on the system.
        Harmonics are sinusoidal voltages or currents having frequencies
        that are integer multiples of the fundamental frequency.
        """,
        'metadata': {'category': 'power_quality', 'source': 'textbook'}
    },
    {
        'id': 'transformer_maintenance',
        'content': """
        Transformer maintenance includes regular oil analysis, thermal imaging,
        and dissolved gas analysis (DGA). DGA is one of the most important
        diagnostic tools for oil-filled transformers. Key gases to monitor
        include hydrogen, methane, ethane, ethylene, and acetylene.
        Increased levels of these gases indicate different types of faults:
        thermal faults, partial discharge, or arcing.
        """,
        'metadata': {'category': 'maintenance', 'source': 'manual'}
    },
    {
        'id': 'smart_grid_ml',
        'content': """
        Machine learning applications in smart grids include load forecasting,
        anomaly detection, and demand response optimization. Deep learning
        models such as LSTM and Transformer have shown superior performance
        for time series forecasting in power systems. Convolutional neural
        networks can be applied to power quality disturbance classification
        using time-frequency representations like spectrograms.
        """,
        'metadata': {'category': 'smart_grid', 'source': 'paper'}
    }
]

rag.add_documents(docs)

# Query
query = "What diagnostic tool is used for transformer faults?"
results = rag.retrieve(query, top_k=2)

print(f"\n🔍 Query: {query}")
print("\n📄 Retrieved Chunks:")
for i, r in enumerate(results):
    print(f"\n[{i+1}] Score: {r['score']:.4f}")
    print(f"    Source: {r['source']['doc_id']}")
    print(f"    Chunk: {r['chunk'][:150]}...")

# Generate augmented prompt
prompt = rag.generate_prompt(query, results)
print(f"\n📝 Augmented Prompt:\n{prompt[:500]}...")


# ===========================================================
# 📖 BAGIAN 4: LLM Fine-tuning dengan LoRA
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     LLM FINE-TUNING: LORA & QLORA                        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  KENAPA FINE-TUNE?                                       ║
║  - Domain-specific knowledge (engineering, legal, med)   ║
║  - Specific task (classification, NER, summarization)    ║
║  - Style/tone adaptation                                 ║
║                                                          ║
║  MASALAH FULL FINE-TUNING:                               ║
║  - GPT-3 punya 175B parameters → butuh 100+ GPU!        ║
║  - Cost: $$$$                                            ║
║  - Storage: model besar untuk setiap fine-tuned version  ║
║                                                          ║
║  SOLUSI: LORA (Low-Rank Adaptation)                      ║
║  ┌─────────────────────────────────────────────┐         ║
║  │  Instead of updating ALL weights:           │         ║
║  │  W_new = W_original + ΔW                   │         ║
║  │                                             │         ║
║  │  LoRA approximates: ΔW ≈ A × B            │         ║
║  │  where A (d × r), B (r × k), r << d,k     │         ║
║  │                                             │         ║
║  │  Result: Only train A and B!               │         ║
║  │  Parameters: 0.1% - 1% dari original       │         ║
║  │  Memory: bisa di consumer GPU (24GB)       │         ║
║  └─────────────────────────────────────────────┘         ║
║                                                          ║
║  QLORA (Quantized LoRA):                                 ║
║  - Base model di-quantize ke 4-bit                       ║
║  - LoRA adapter tetap 16-bit                             ║
║  - Bisa fine-tune 70B model di single 24GB GPU!         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 5: LLM Deployment
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     LLM DEPLOYMENT PATTERNS                              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  PATTERN 1: API WRAPPER (OpenAI/Anthropic)               ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Client  │───▶│ FastAPI │───▶│ OpenAI  │             ║
║  │         │◀───│ Server  │◀───│ API     │             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║  - Simple, reliable, managed                             ║
║  - Cost: per token, latency: 1-3s                       ║
║  - Best untuk: prototyping, low volume                   ║
║                                                          ║
║  PATTERN 2: SELF-HOSTED (vLLM/TGI)                       ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Client  │───▶│ FastAPI │───▶│ vLLM    │             ║
║  │         │◀───│ Server  │◀───│ Server  │             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║  - Full control, privacy                                 ║
║  - Cost: GPU rental, latency: 100-500ms                  ║
║  - Best untuk: high volume, sensitive data               ║
║                                                          ║
║  PATTERN 3: EDGE / LOCAL (llama.cpp/ollama)              ║
║  ┌─────────┐    ┌─────────┐                             ║
║  │ Client  │───▶│ Ollama  │                             ║
║  │         │◀───│ (local) │                             ║
║  └─────────┘    └─────────┘                             ║
║  - No internet, private                                  ║
║  - Cost: 0, latency: varies                              ║
║  - Best untuk: development, privacy-critical             ║
║                                                          ║
║  OPTIMIZATION TECHNIQUES:                                ║
║  - Quantization: FP16 → INT8 → INT4 (smaller, faster)    ║
║  - Batch processing: multiple requests together            ║
║  - KV Cache: reuse computation antar tokens              ║
║  - Streaming: send tokens as generated                   ║
║  - Prompt caching: cache common prefixes                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 🏋️ EXERCISE 1: Build RAG System for EE Domain
# ===========================================================
"""
Bangun RAG system untuk dokumentasi Teknik Elektro:

Requirements:
1. Document ingestion: PDF/TXT dari manual, datasheet, paper
2. Chunking strategy: semantic chunks (per section/paragraph)
3. Embedding: Sentence-Transformers (all-MiniLM-L6-v2)
4. Vector DB: ChromaDB (local) atau FAISS
5. Query: natural language questions about EE topics
6. Source citation: setiap jawaban harus cite sumber

Contoh queries:
- "Apa yang menyebabkan voltage sag?"
- "Bagaimana cara diagnose fault di transformer?"
- "Apa perbedaan THD dan TDD?"

Bonus:
- Gradio/Streamlit UI
- Conversation history (multi-turn)
- Re-ranking dengan cross-encoder
"""


# ===========================================================
# 🏋️ EXERCISE 2: Fine-tune Small LLM
# ===========================================================
"""
Fine-tune model kecil (Phi-3 / Llama 3.2 3B) untuk domain EE:

Dataset preparation:
1. Kumpulkan 100+ Q&A pairs tentang Teknik Elektro
2. Format: instruction tuning (Alpaca format)
   {
     "instruction": "Jelaskan apa itu power factor",
     "input": "",
     "output": "Power factor adalah rasio antara..."
   }

Fine-tuning dengan LoRA:
1. Load base model (4-bit quantized)
2. Add LoRA adapters (rank=16, alpha=32)
3. Train 3-5 epochs
4. Save adapter weights only

Evaluation:
- Compare: base model vs fine-tuned model
- Metrics: relevance, accuracy, technical depth
- Test dengan questions di luar training set
"""


# ===========================================================
# 🏋️ EXERCISE 3: LLM Agent
# ===========================================================
"""
Bangun simple agent dengan function calling:

Tools:
1. calculate_thd(harmonics) → return THD percentage
2. convert_dbmw(value) → convert to dBm/Watts
3. search_documentation(query) → RAG search
4. get_equipment_status(equipment_id) → mock API call

Agent logic:
- ReAct pattern: Reason → Act → Observe
- LLM decides which tool to use
- Chain multiple tools untuk complex tasks

Contoh conversation:
User: "Equipment T-001 status dan apa THD-nya?"
Agent: [call get_equipment_status] → [call calculate_thd] → answer
"""


# ===========================================================
# 🔥 CHALLENGE: Production LLM Application
# ===========================================================
"""
Bangun LLM application production-ready:

Requirements:
1. RAG dengan 1000+ dokumen EE
2. FastAPI backend dengan streaming responses
3. Chat interface (Streamlit/Gradio)
4. Conversation memory
5. Cost tracking (tokens used per request)
6. Rate limiting
7. Prompt versioning
8. Evaluation framework (benchmark Q&A set)

Architecture:
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Streamlit│───▶│ FastAPI  │───▶│ ChromaDB │
│  (UI)    │◀───│ (Backend│◀───│ (Vector) │
└──────────┘    └────┬─────┘    └──────────┘
                     │
                     ▼
              ┌──────────┐
              │ OpenAI / │
              │ Local LLM│
              └──────────┘

Deliverable:
- Code repository
- Docker Compose
- Benchmark report
- Demo video
"""


print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke projects/ untuk FLAGSHIP project")
print("="*50)
