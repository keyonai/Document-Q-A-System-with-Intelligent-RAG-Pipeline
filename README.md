# Document Q&A System with Intelligent RAG Pipeline

My extern final project. Upload pharmaceutical PDFs and ask questions about them in plain English — the system figures out which documents are relevant, retrieves the right chunks, and generates answers with source citations.

## What it does

- Parses multi-page pharmaceutical PDFs and detects where each document starts and ends
- Classifies each page by document type (Certificate of Quality, Packaging Spec, Cover Letter, etc.)
- Builds a separate FAISS vector index per document type for smarter retrieval
- Routes each question to the right index before searching
- Generates answers using Qwen 2.5 7B (4-bit quantized) with page-level source attribution
- Serves everything through a Gradio web interface

## Models

| Role | Model |
|------|-------|
| LLM | Qwen 2.5 7B Instruct (4-bit quantized) |
| Embeddings | all-MiniLM-L6-v2 |

## Stack

- Python
- LlamaIndex, FAISS, sentence-transformers
- Transformers + bitsandbytes (4-bit quantization)
- PyMuPDF, PyPDF2, pytesseract
- Gradio
- Google Colab (T4 GPU)

## Run it

Open `Keyonai_W_Enhanced_Document_Q&A_System_with_Intelligent_RAG_Pipeline.ipynb` in Google Colab with a T4 GPU runtime.

```bash
pip install transformers torch bitsandbytes sentence-transformers faiss-cpu gradio llama-index PyPDF2 pymupdf pytesseract
```

Two sample pharmaceutical PDFs are included to test with.
