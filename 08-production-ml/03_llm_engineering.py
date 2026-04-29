"""
=============================================================
FASE 8 - MODUL 3: LLM ENGINEERING
=============================================================
LLM (Large Language Models) merevolusi AI landscape di 2023-2026.
Sebagai ML Engineer, kamu perlu memahami:
- How to use LLMs effectively
- How to fine-tune untuk specific tasks
- How to deploy LLMs efficiently
- How to build LLM-powered applications

Koneksi Teknik Elektro:
- LLM = large-scale nonlinear system
- Prompt engineering = input signal conditioning
- Fine-tuning = system adaptation
- RAG = adaptive filter dengan external memory
- Quantization = reduced-precision signal processing

Durasi target: 4-5 jam
============================================================="""

import numpy as np

np.random.seed(42)


# ===========================================================
# BAGIAN 1: LLM Landscape
# ===========================================================
print("="*60)
print("BAGIAN 1: LLM LANDSCAPE")
print("="*60)

llm_landscape = """
TARGET MODELS:

Open Source (Self-Hosted):
  - LLaMA 2/3 (Meta): 7B, 13B, 70B parameters
  - Mistral: 7B, 8x7B (Mixture of Experts)
  - Falcon: 7B, 40B, 180B
  - Qwen: 7B, 14B, 72B (multilingual)
  - Gemma (Google): 2B, 7B

API-Based (Cloud):
  - GPT-4/4o (OpenAI): SOTA capabilities
  - Claude 3 (Anthropic): strong reasoning
  - Gemini (Google): multimodal

Specialized:
  - Code: CodeLlama, StarCoder, DeepSeek-Coder
  - Medical: Med-PaLM
  - Vision: LLaVA, GPT-4V

DETAIL:
- Open source models bisa di-self-host, lebih murah untuk high volume,
  tapi butuh infrastructure dan expertise.
- API-based models instant access, no maintenance, tapi ada cost per token.
- Specialized models lebih baik untuk domain tertentu.

TARGET KEY CONCEPTS:

Parameters:
  - 1B = 1 billion parameters
  - GPT-3: 175B parameters
  - GPT-4: estimated 1T+ parameters
  - Storage: 1B params ~ 2-4 GB (FP16)
  
  DETAIL:
  - Parameters = weights dalam neural network.
  - Semakin banyak parameters, semakin powerful model (generally).
  - Tapi juga semakin banyak memory dan compute yang dibutuhkan.

Context Window:
  - Max tokens yang bisa di-process
  - GPT-4: 128K tokens
  - Claude 3: 200K tokens
  - Llama 2: 4K tokens (base)
  
  DETAIL:
  - Context window = "working memory" dari LLM.
  - Longer context = bisa process dokumen lebih panjang.
  - Tapi longer context juga lebih expensive dan slower.

Tokens:
  - Unit of text (~ 0.75 words)
  - Pricing based on tokens
  - Input + output tokens dihitung
  
  DETAIL:
  - Tokenization: text di-split ke subword units.
  - English: ~100 tokens = ~75 words.
  - Pricing API-based models: per 1K tokens.

TARGET CAPABILITIES:

Text Generation:
  - Completion, summarization, translation
  - Question answering
  - Creative writing

Code:
  - Code generation, explanation, debugging
  - Refactoring, documentation
  - Test generation

Reasoning:
  - Chain-of-thought
  - Mathematical reasoning
  - Logical inference

Multimodal (GPT-4V, Gemini):
  - Image understanding
  - Video analysis
  - Audio processing
"""
print(llm_landscape)


# ===========================================================
# BAGIAN 2: Prompt Engineering Patterns
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 2: PROMPT ENGINEERING PATTERNS")
print("="*60)

prompt_patterns = """
TARGET ZERO-SHOT PROMPTING:
No examples, just instructions.

Example:
"Classify the sentiment of this review as positive, negative, or neutral:
Review: 'This product exceeded my expectations!'
Sentiment:"

DETAIL:
- Zero-shot = model menjawab tanpa melihat contoh.
- Works well untuk tasks yang sederhana dan umum.
- Tapi less reliable untuk tasks yang kompleks atau niche.

TARGET FEW-SHOT PROMPTING:
Provide examples untuk guide the model.

Example:
"Classify the sentiment:

Review: 'Amazing quality, highly recommend!'
Sentiment: Positive

Review: 'Terrible, waste of money'
Sentiment: Negative

Review: 'It works as expected'
Sentiment: Neutral

Review: 'Best purchase ever!'
Sentiment:"

DETAIL:
- Few-shot = model melihat beberapa contoh sebelum menjawab.
- Contoh membantu model understand format dan kriteria.
- 3-5 examples biasanya cukup.
- Examples harus diverse dan representative.

TARGET CHAIN-OF-THOUGHT (CoT):
Ask model untuk think step-by-step.

Example:
"Solve this math problem step by step:
If a train travels 60 km/h and needs to cover 240 km,
how long will it take?

Step 1: Identify known variables
Step 2: Identify the formula
Step 3: Calculate the answer"

DETAIL:
- CoT memaksa model untuk show reasoning process.
- Lebih accurate untuk tasks yang memerlukan reasoning.
- Bisa dikombinasikan dengan few-shot untuk lebih baik.
- Variants: Tree of Thoughts, Self-Consistency CoT.

TARGET ROLE PROMPTING:
Assign a role ke model.

Example:
"You are an expert electrical engineer with 20 years experience.
Explain how a three-phase induction motor works, focusing on:
1. The rotating magnetic field
2. Slip and its significance
3. Common failure modes"

DETAIL:
- Role prompting mempengaruhi tone dan depth dari response.
- Model akan "act" sesuai role yang diberikan.
- Berguna untuk domain-specific questions.

TARGET STRUCTURED OUTPUT:
Request specific output format.

Example:
"Analyze this power quality report and provide output in JSON:
{
  'voltage_sag_detected': boolean,
  'severity': 'low' | 'medium' | 'high',
  'recommended_action': string,
  'confidence': number (0-1)
}"

DETAIL:
- Structured output memudahkan parsing dan integration.
- JSON, XML, YAML adalah format yang umum.
- Sertakan schema dan type hints di prompt.
- Verify output structure setelah generation.

TARGET RETRIEVAL-AUGMENTED GENERATION (RAG):
Combine LLM dengan external knowledge.

Architecture:
  User Query -> Retrieve Documents -> Combine dengan Query -> LLM -> Answer

Components:
  - Vector database: store document embeddings
  - Retriever: find relevant documents
  - Generator: LLM yang generate answer dengan context
  
DETAIL:
- RAG mengatasi limitation LLM: knowledge cutoff dan hallucination.
- Documents di-chunk dan di-embed ke vector space.
- Retrieval menggunakan similarity search (cosine similarity).
- Re-ranking bisa meningkatkan relevance.

TARGET PROMPT CHAINING:
Break complex tasks into multiple prompts.

Example:
  Prompt 1: "Extract key metrics dari report ini"
  Prompt 2: "Analyze trends dari metrics: [output dari prompt 1]"
  Prompt 3: "Generate recommendations berdasarkan analysis: [output dari prompt 2]"
  
DETAIL:
- Prompt chaining = pipeline of LLM calls.
- Setiap step bisa di-verify sebelum lanjut.
- Lebih reliable untuk complex tasks.
- Bisa di-parallelize untuk independent steps.

Koneksi Teknik Elektro:
- Prompt = input signal dengan specific characteristics
- Few-shot = providing reference signals
- CoT = sequential processing (seperti filter cascade)
- RAG = system dengan external memory/storage
- Prompt chaining = multi-stage signal processing
"""
print(prompt_patterns)


# ===========================================================
# BAGIAN 3: RAG Implementation
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 3: RAG IMPLEMENTATION")
print("="*60)

class SimpleRAG:
    """
    Simplified Retrieval-Augmented Generation system.
    
    Parameters:
    -----------
    documents : list
        List of documents (strings).
    embedding_model : callable
        Function untuk compute embeddings.
    llm : callable
        Function untuk generate text.
        
    Notes:
    ------
    - RAG = combine retrieval dengan generation
    - Retrieval: find relevant documents dari knowledge base
    - Generation: generate answer menggunakan retrieved context
    - Benefit: accurate, up-to-date, verifiable
    - Ini adalah educational implementation. Production RAG
      menggunakan vector databases dan production LLM APIs.
    
    Koneksi Teknik Elektro:
    - Retrieval = matched filter (find similar signals)
    - Generation = signal synthesis (generate response)
    - Vector DB = filter bank (store reference signals)
    """
    
    def __init__(self, documents, embedding_model=None, llm=None):
        self.documents = documents
        self.embeddings = None
        self.embedding_model = embedding_model or self._simple_embedding
        self.llm = llm or self._simple_llm
        
        # Pre-compute embeddings
        self._index_documents()
    
    def _simple_embedding(self, text):
        """Simple bag-of-words embedding (untuk demo)."""
        words = text.lower().split()
        # Create simple vector (hash-based)
        vec = np.zeros(100)
        for word in words:
            idx = hash(word) % 100
            vec[idx] += 1
        # Normalize
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def _simple_llm(self, prompt):
        """Placeholder LLM (dalam praktik: OpenAI, local model, etc.)."""
        return f"[Generated response based on context]\nPrompt: {prompt[:100]}..."
    
    def _index_documents(self):
        """Compute dan store document embeddings."""
        self.embeddings = np.array([
            self.embedding_model(doc) for doc in self.documents
        ])
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k relevant documents.
        
        Parameters:
        -----------
        query : str
            Query string.
        top_k : int, default 3
            Jumlah documents untuk retrieve.
            
        Returns:
        --------
        list
            Top-k relevant documents.
            
        Notes:
        ------
        - Menggunakan cosine similarity untuk ranking.
        - Alternatif: Euclidean distance, dot product.
        - Re-ranking dengan cross-encoder bisa meningkatkan quality.
        """
        query_embedding = self.embedding_model(query)
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.documents[i] for i in top_indices]
    
    def generate(self, query, top_k=3):
        """
        Generate answer menggunakan RAG.
        
        Parameters:
        -----------
        query : str
            User query.
        top_k : int, default 3
            Jumlah context documents.
            
        Returns:
        --------
        str
            Generated answer.
            
        Notes:
        ------
        - Context di-limit untuk fit dalam context window.
        - Prompt engineering untuk instruct model menggunakan context.
        - Cite sources untuk verifiability.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Build prompt dengan context
        context = "\n\n".join(retrieved_docs)
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        return self.llm(prompt)


# Demo
print("\n=== RAG Demo ===")
documents = [
    "Three-phase induction motors are the most common type of AC motor.",
    "The rotating magnetic field in a three-phase motor is created by three windings.",
    "Slip is the difference between synchronous speed and rotor speed.",
    "Common failure modes include bearing failure and insulation breakdown.",
    "Power factor in induction motors is typically lagging due to reactive power.",
]

rag = SimpleRAG(documents)
query = "What causes induction motors to fail?"
retrieved = rag.retrieve(query, top_k=2)

print(f"\nQuery: {query}")
print(f"\nRetrieved documents:")
for i, doc in enumerate(retrieved):
    print(f"  {i+1}. {doc}")


# ===========================================================
# BAGIAN 4: Fine-tuning dan PEFT
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 4: FINE-TUNING DAN PEFT")
print("="*60)

finetuning_guide = """
TARGET FINE-TUNING STRATEGIES:

1. FULL FINE-TUNING
   - Update all parameters
   - Requires: GPU memory = 2-4x model size
   - Best untuk: large datasets, significant domain shift
   - Contoh: Llama-2-7B full fine-tune ~ 28-56 GB GPU
   
   DETAIL:
   - Full fine-tuning memberikan flexibility maksimal.
   - Tapi sangat expensive untuk large models.
   - Bisa menyebabkan catastrophic forgetting jika tidak hati-hati.

2. PARAMETER-EFFICIENT FINE-TUNING (PEFT):
   a) LoRA (Low-Rank Adaptation):
      - Add low-rank matrices ke attention weights
      - Update hanya matrices baru (2-4% dari total params)
      - Formula: W = W0 + BA, dimana B dan A adalah low-rank
      - Rank: typically 8, 16, 32, 64
      - Memory: ~MBs instead of GBs
      
      DETAIL:
      - LoRA membuat fine-tuning LLM accessible untuk consumer GPUs.
      - W0 (pre-trained weights) di-freeze.
      - Hanya A dan B yang di-train.
      - At inference, bisa merge adapters dengan base model.
      
   b) QLoRA (Quantized LoRA):
      - Quantize base model ke 4-bit
      - Add LoRA adapters
      - Memory: Llama-2-7B ~ 6 GB GPU
      - Very popular untuk consumer GPUs
      
      DETAIL:
      - Quantization mengurangi precision weights ke 4-bit.
      - NormalFloat (NF4) quantization lebih baik dari INT4.
      - Double quantization untuk quantize quantization constants.
      - Paged optimizers untuk handle memory spikes.
      
   c) Prefix Tuning:
      - Add trainable prefix tokens
      - Freeze base model
      - Only train prefix embeddings
      
   d) Prompt Tuning:
      - Similar ke prefix tuning
      - Add soft prompts (learnable embeddings)

TARGET INSTRUCTION TUNING:
- Format: Instruction -> Input -> Output
- Contoh:
  {
    "instruction": "Translate English to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  }
- Dataset: Alpaca, Dolly, OpenAssistant

DETAIL:
- Instruction tuning membuat model lebih baik mengikuti instructions.
- Format standar: instruction + input + output.
- Quality dataset lebih penting daripada quantity.

TARGET RLHF (Reinforcement Learning from Human Feedback):
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training (human preferences)
3. PPO (Proximal Policy Optimization)

DETAIL:
- SFT: fine-tune pada high-quality demonstrations.
- Reward Model: train model untuk predict human preferences.
- PPO: optimize policy untuk maximize expected reward.
- RLHF adalah kunci untuk ChatGPT-like capabilities.

TARGET QUANTIZATION:
- FP32 -> FP16 -> INT8 -> INT4
- Methods: GPTQ, AWQ, GGUF
- Impact: smaller model, faster inference, lower memory
- Tradeoff: slight accuracy loss

DETAIL:
- GPTQ: quantization berbasis gradient untuk minimize error.
- AWQ: activation-aware quantization, protects salient weights.
- GGUF: format untuk GGML, digunakan oleh llama.cpp.
- Quantization bisa 2-4x memory reduction dengan minimal accuracy loss.

Koneksi Teknik Elektro:
- Fine-tuning = system adaptation (like adaptive filters)
- LoRA = low-rank approximation (like model order reduction)
- Quantization = reduced-precision ADC/DAC
- RLHF = feedback control dengan human-in-the-loop
"""
print(finetuning_guide)


# ===========================================================
# BAGIAN 5: Deployment Patterns
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 5: DEPLOYMENT PATTERNS")
print("="*60)

deployment_patterns = """
TARGET DEPLOYMENT OPTIONS:

1. API WRAPPER (OpenAI, Anthropic)
   - Pros: no infrastructure, latest models, easy setup
   - Cons: cost, latency, data privacy, rate limits
   - Use case: prototyping, low-volume, non-sensitive data
   - Cost: ~$0.01-0.03 per 1K tokens
   
   DETAIL:
   - API wrapper adalah cara termudah untuk deploy LLM.
   - OpenAI API punya streaming support untuk real-time responses.
   - Rate limits perlu di-manage untuk high-volume applications.
   - Data privacy concern untuk sensitive data.

2. SELF-HOSTED (vLLM, TGI)
   - Pros: full control, no per-request cost, data privacy
   - Cons: infrastructure, expertise, model management
   - Use case: high-volume, sensitive data, custom models
   - Hardware: GPU required (A100, H100 untuk large models)
   
   DETAIL:
   - vLLM: high-throughput inference dengan PagedAttention.
     Throughput bisa 10-20x lebih tinggi dari naive serving.
   - TGI (Text Generation Inference): HuggingFace's serving solution.
     Supports streaming, quantization, dan safety.
   - Self-hosted memerlukan GPU infrastructure yang mahal.

3. EDGE DEPLOYMENT (ONNX, TensorRT, GGML)
   - Pros: low latency, no network dependency
   - Cons: limited model size, hardware constraints
   - Use case: mobile, IoT, real-time applications
   - Hardware: CPU atau edge GPU (Jetson)
   
   DETAIL:
   - Edge deployment menggunakan quantized models.
   - llama.cpp dengan GGUF format bisa run di CPU.
   - MobileLLM, Phi-3 adalah models yang optimized untuk edge.

TARGET INFERENCE OPTIMIZATION:

1. BATCHING
   - Dynamic batching: group requests untuk efficient GPU utilization
   - Continuous batching (vLLM): process tokens dari multiple requests
   
   DETAIL:
   - Batching meningkatkan throughput dengan amortize overhead.
   - Continuous batching (vLLM) lebih efisien dari static batching.
   - Tradeoff: latency increase untuk individual requests.

2. KV CACHE
   - Cache key-value pairs dari previous tokens
   - Avoid redundant computation
   - Critical untuk autoregressive generation
   
   DETAIL:
   - KV cache menyimpan intermediate computations.
   - Memory usage = O(batch_size * seq_len * d_model * num_layers * 2).
   - PagedAttention (vLLM) mengoptimalkan KV cache memory management.

3. SPECULATIVE DECODING
   - Use small draft model untuk predict next tokens
   - Verify dengan large model
   - Speedup: 2-3x
   
   DETAIL:
   - Draft model lebih cepat tapi less accurate.
   - Large model memverifikasi predictions dari draft model.
   - Effective untuk tasks dengan repetitive patterns.

4. QUANTIZATION
   - INT8/INT4 inference
   - Minimal accuracy loss
   - 2-4x memory reduction

TARGET SERVING INFRASTRUCTURE:

Architecture:
  Load Balancer -> API Gateway -> Model Servers (auto-scaling)
                          |
                   Cache (Redis) -> Vector DB (jika RAG)

Components:
  - API: FastAPI/Flask untuk REST, gRPC untuk high-performance
  - Queue: Redis/RabbitMQ untuk async processing
  - Cache: Redis untuk frequent queries
  - Monitoring: Prometheus + Grafana
  - Auto-scaling: Kubernetes HPA

TARGET COST OPTIMIZATION:

1. Right-size model:
   - 7B model often sufficient
   - 70B only jika absolutely necessary
   
2. Use caching:
   - Cache common queries
   - Redis untuk semantic cache
   
3. Optimize prompts:
   - Shorter prompts = fewer tokens
   - Use system prompts efficiently
   
4. Batch requests:
   - Higher throughput
   - Lower cost per request
   
5. Hybrid approach:
   - Small model untuk simple tasks
   - Large model untuk complex tasks
   - Router untuk distribute requests
   
   DETAIL:
   - Router bisa menggunakan heuristics atau classifier.
   - Simple tasks: factual Q&A, summarization.
   - Complex tasks: reasoning, code generation.
"""
print(deployment_patterns)


# ===========================================================
# LATIHAN 24: LLM Application Development
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun LLM-powered application
   - Mengimplementasikan RAG pipeline
   - Melakukan fine-tuning dengan LoRA
   - Deploy LLM untuk production

PANDUAN LANGKAH-LANGKAH:

STEP 1: Build RAG Application
-----------------------------
   Konteks: Technical Documentation Q&A untuk Power Systems
   
   a) Document ingestion:
      - Parse technical documents (PDF, HTML, Markdown)
      - Chunk documents (512-1024 tokens per chunk)
      - Compute embeddings (OpenAI, sentence-transformers)
      - Store di vector database (Chroma, Pinecone)
      
      DETAIL:
      - Chunking strategy: fixed-size, semantic, atau recursive.
      - Overlap antar chunks untuk maintain context.
      - Embeddings: OpenAI ada-002, sentence-transformers all-MiniLM-L6-v2.
      
   b) Retrieval:
      - Embed user query
      - Semantic search (cosine similarity)
      - Re-rank dengan cross-encoder
      - Return top-5 relevant chunks
      
      DETAIL:
      - Semantic search menggunakan vector similarity.
      - Re-ranking dengan cross-encoder lebih accurate.
      - Hybrid search: combine semantic + keyword search.
      
   c) Generation:
      - Build prompt dengan context
      - Generate answer dengan LLM
      - Cite sources (chunk references)
      
      DETAIL:
      - Prompt harus instruct model untuk menggunakan context.
      - Cite sources untuk verifiability dan trust.
      - Handle case jika context tidak cukup.
      
   d) Evaluation:
      - Test dengan 50+ questions
      - Metrics: relevance, accuracy, completeness
      - Human evaluation: rate 1-5


STEP 2: Fine-tuning dengan LoRA
-------------------------------
   a) Dataset preparation:
      - Collect domain-specific conversations
      - Format: instruction + input + output
      - Split: 80% train, 10% val, 10% test
      
      DETAIL:
      - Quality > quantity untuk fine-tuning dataset.
      - Format standar: Alpaca format atau ChatML format.
      
   b) LoRA configuration:
      - Base model: Llama-2-7B atau Mistral-7B
      - Rank: 16 atau 32
      - Target modules: q_proj, v_proj
      - Learning rate: 2e-4
      
      DETAIL:
      - Rank 16 biasanya cukup untuk most tasks.
      - Target modules: attention weights (q_proj, v_proj, k_proj, o_proj).
      - Alpha (scaling factor) biasanya = 2 * rank.
      
   c) Training:
      - QLoRA untuk memory efficiency
      - Batch size: 4-8
      - Epochs: 3-5
      - Evaluate: loss, perplexity
      
      DETAIL:
      - QLoRA memungkinkan training di consumer GPUs (RTX 3090, 4090).
      - Gradient accumulation untuk effective larger batch size.
      - Learning rate scheduler: cosine with warmup.
      
   d) Inference:
      - Merge adapters dengan base model (opsional)
      - Test: domain-specific questions
      - Compare: before vs after fine-tuning


STEP 3: Deployment
------------------
   a) API Development:
      - FastAPI endpoints: /chat, /rag, /health
      - Async processing untuk concurrent requests
      - Request/response validation (Pydantic)
      
      DETAIL:
      - FastAPI dengan async support untuk high concurrency.
      - Streaming responses untuk better UX.
      - Rate limiting untuk prevent abuse.
      
   b) Containerization:
      - Dockerfile untuk model dan API
      - Multi-stage build untuk optimize size
      - Docker Compose untuk local development
      
   c) Production deployment:
      - Kubernetes dengan GPU nodes
      - Horizontal Pod Autoscaler (HPA)
      - Ingress untuk load balancing
      - Monitoring dengan Prometheus/Grafana


STEP 4: Evaluation & Iteration
------------------------------
   a) Automated evaluation:
      - Benchmark dataset (50+ questions)
      - Metrics: accuracy, latency, cost per query
      - Regression testing untuk setiap deployment
      
   b) User feedback:
      - Thumbs up/down untuk responses
      - Feedback collection pipeline
      - Periodic retraining dengan feedback
      
   c) A/B testing:
      - Compare: different prompts, models, RAG strategies
      - Metrics: user satisfaction, task completion
      - Statistical significance testing


TIPS:
   - Start dengan OpenAI API untuk prototyping cepat
   - Use HuggingFace PEFT library untuk LoRA
   - ChromaDB = simple vector DB untuk development
   - vLLM = high-throughput inference engine
   - LangChain = framework untuk LLM applications

PERINGATAN COMMON MISTAKES:
   - RAG tanpa proper chunking (too large/small)
   - Fine-tuning tanpa enough data
   - No evaluation framework
   - Ignore latency requirements
   - No cost monitoring
   - Deploy tanpa guardrails (safety, bias)

TARGET EXPECTED OUTPUT:
   - Working RAG application
   - Fine-tuned model dengan domain knowledge
   - Production API dengan monitoring
   - Evaluation framework
   - Documentation dan deployment guide
"""


# ===========================================================
# CHALLENGE: Production LLM System
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun end-to-end LLM system untuk production
   - Mengintegrasikan dengan existing infrastructure
   - Implementasi comprehensive monitoring dan safety

PANDUAN LANGKAH-LANGKAH:

STEP 1: Problem Definition
--------------------------
   Pilih use case:
   
   a) Industrial Assistant:
      - Q&A untuk equipment manuals
      - Troubleshooting guidance
      - Maintenance scheduling advice
      
   b) Code Assistant:
      - Generate control system code
      - Explain algorithms
      - Debug PLC programs
      
   c) Report Generator:
      - Generate technical reports
      - Summarize inspection findings
      - Create maintenance recommendations


STEP 2: System Architecture
---------------------------
   a) Data Layer:
      - Document store (S3, GCS)
      - Vector database (Pinecone, Weaviate)
      - Conversation history (PostgreSQL)
      
   b) Application Layer:
      - API Gateway (rate limiting, auth)
      - RAG pipeline (retrieval + generation)
      - Fine-tuned model (LoRA adapters)
      - Prompt management (versioning)
      
   c) Serving Layer:
      - Model servers (vLLM/TGI)
      - Load balancer
      - Cache (Redis)
      - Queue (RabbitMQ untuk async)
      
   d) Monitoring Layer:
      - Performance metrics (latency, throughput)
      - Quality metrics (relevance, hallucination)
      - Safety metrics (toxicity, bias)
      - Cost tracking (per request, per user)


STEP 3: Safety & Guardrails
---------------------------
   a) Input validation:
      - Rate limiting (requests per minute)
      - Content filtering (block harmful inputs)
      - Size limits (max tokens)
      
   b) Output filtering:
      - Toxicity detection
      - PII detection dan redaction
      - Fact checking (jika possible)
      
   c) Fallback mechanisms:
      - Jika model fails -> heuristic response
      - Jika confidence low -> "I don't know"
      - Jika safety triggered -> block response
      
   d) Audit logging:
      - Log all requests dan responses
      - Track user interactions
      - Compliance requirements


STEP 4: Performance Optimization
--------------------------------
   a) Model optimization:
      - Quantization (INT8/INT4)
      - Pruning (remove unnecessary weights)
      - Distillation (smaller student model)
      
   b) Serving optimization:
      - Dynamic batching
      - KV cache optimization
      - Speculative decoding
      
   c) Infrastructure:
      - GPU cluster dengan auto-scaling
      - CDN untuk static assets
      - Edge caching untuk common queries


STEP 5: Evaluation & Improvement
--------------------------------
   a) Automated evaluation:
      - Benchmark suite (100+ test cases)
      - Continuous evaluation pipeline
      - Regression detection
      
   b) Human evaluation:
      - Expert review (domain experts)
      - User satisfaction surveys
      - A/B testing dengan real users
      
   c) Iteration:
      - Collect feedback
      - Identify failure modes
      - Retrain atau fine-tune
      - Deploy improvements


TIPS:
   - Use LangChain atau LlamaIndex untuk orchestration
   - Weights & Biases untuk experiment tracking
   - MLflow untuk model registry
   - Great Expectations untuk data validation
   - Guardrails AI untuk output validation

PERINGATAN COMMON MISTAKES:
   - Ignore safety dan guardrails
   - No monitoring untuk hallucinations
   - Underestimate infrastructure costs
   - No fallback untuk model failures
   - Ignore data privacy regulations
   - Deploy tanpa proper testing

TARGET EXPECTED OUTPUT:
   - Production LLM application
   - RAG dengan domain-specific knowledge
   - Safety guardrails dan monitoring
   - Performance optimization (latency < 2s)
   - Cost optimization (< $0.01 per query)
   - Comprehensive documentation

LLM Engineering adalah skill yang sangat dicari di 2025-2026!
Master ini dan kamu akan sangat valuable.
"""

print("\n" + "="*50)
print("SELESAI FASE 8!")
print("="*50)
print("""
Kamu sekarang bisa:
OK Design dan implement feature stores
OK Build model monitoring dan drift detection
OK Develop LLM-powered applications
OK Deploy production ML systems

SELESAI!

Kamu telah menyelesaikan seluruh kurikulum ML Engineer Track!

Next steps:
1. Review semua projects di folder projects/
2. Build portfolio dengan 3+ end-to-end projects
3. Apply untuk ML Engineer roles
4. Continue learning - ML field evolves rapidly!

Good luck!
""")
