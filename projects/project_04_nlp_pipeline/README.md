# Project 4: NLP Pipeline

## 🎯 Tujuan
Menerapkan NLP dengan Transformer untuk task classification/NER/summarization.
Tunjukkan kemampuan Hugging Face ecosystem + deployment.

## 🚀 Production-Ready Requirements
> **New:** Deploy sebagai API dengan inference optimization.

## Deliverables
1. **NLP Pipeline** — preprocessing, tokenization, model
2. **Fine-tuned Model** — BERT/RoBERTa/IndoBERT
3. **Evaluation** — task-specific metrics
4. **Deployment** — FastAPI + Docker (NEW)
5. **Demo** — Interactive text input (NEW)

## Dataset & Task
Pilih salah satu:
- **Sentiment Analysis** — Indonesian tweets/reviews (IndoBERT)
- **Text Classification** — news categorization
- **NER** — named entity recognition (person, org, location)
- **Custom:** Technical document classification (⭐ EE domain!)

## Checklist
### Data & Preprocessing
- [ ] Dataset loading & exploration
- [ ] Text preprocessing (cleaning, normalization)
- [ ] Tokenization dengan Hugging Face tokenizer
- [ ] Label encoding & class distribution analysis

### Model Training
- [ ] Load pretrained model (BERT / RoBERTa / IndoBERT)
- [ ] Fine-tuning dengan Trainer API atau custom loop
- [ ] Hyperparameter tuning (lr, batch_size, epochs)
- [ ] Evaluation metrics (accuracy, F1, precision, recall)
- [ ] Save best model & tokenizer
- [ ] Push to Hugging Face Hub (opsional)

### Production (NEW)
- [ ] **FastAPI serving** — `/predict` dengan text input
- [ ] **Batch inference** — `/predict/batch` endpoint
- [ ] **Docker containerization**
- [ ] **Gradio demo** — simple web interface
- [ ] **Model optimization** — ONNX export (opsional)

### LLM Extension (Opsional — untuk AI Engineer track)
- [ ] RAG atas corpus text
- [ ] Simple chatbot dengan context
- [ ] Integration dengan OpenAI API

## Hasil
*(Tulis hasil di sini setelah selesai)*

### Model Performance
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Baseline (TF-IDF + LR) | | | | |
| BERT-base | | | | |
| Fine-tuned BERT | | | | |

### API Example
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Ini adalah contoh teks untuk diklasifikasikan"}'
```

### Demo
- Gradio URL: (deploy ke Hugging Face Spaces)
- Streamlit URL: (deploy ke Streamlit Cloud)
