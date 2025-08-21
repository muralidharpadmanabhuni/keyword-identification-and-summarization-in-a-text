# Keyword Identification and Summarization in Text using Transformer Models

## 📌 Overview
This project implements an advanced system for **keyword extraction** and **text summarization** using deep learning, particularly **transformer-based NLP models**.  

We use:
- **DistilBERT** for keyword extraction  
- **T5 (Text-to-Text Transfer Transformer)** for abstractive summarization  

This combination allows us to generate **context-aware keywords** and **human-like summaries**, outperforming traditional methods such as TF-IDF, TextRank, and extractive-only models.

---

## ✨ Features
- 🔑 Accurate **keyword extraction** using transformer models  
- 📝 **Abstractive summarization** with T5  
- 📄 Supports multiple input formats: Text, PDF, DOCX  
- ⚡ Flask-based **REST API** for easy integration  
- 📊 Evaluation with **Precision, Recall, F1-score, ROUGE**  
- 🚀 Scalable, GPU/Cloud-ready implementation  

---

## 📚 Research Reference
This work is based on our B.Tech project and also supported by research published in:

- **“Keyword Identification and Summarization in a Text by Using Deep Learning Based Models”**  
  *2025 International Conference on Inventive Computation Technologies (ICICT)*  
  DOI: [10.1109/ICICT64420.2025.11005054](https://ieeexplore.ieee.org/document/11005054)  

Authors: Manam Bhavya Lakshmi, Murali Dhar Padmanabhuni, Bharath Satya Praveen Katragadda, Dhanusri Nakkina  and Sai Jyothi Sri Pusuluri

---

## ⚙️ System Architecture
1. **Input Handling** – Accepts text, PDF, or Word documents  
2. **Preprocessing** – Cleans and tokenizes input  
3. **Keyword Extraction** – DistilBERT identifies key terms  
4. **Summarization** – T5 generates coherent summaries  
5. **Output** – Displays extracted keywords and generated summary  

---

## 📊 Results
- **Keyword Extraction (DistilBERT)**  
  - Accuracy: **93.37%**  
  - Precision: **93.40%**  
  - Recall: **93.37%**  
  - F1-Score: **93.37%**  

- **Summarization (T5, ROUGE Scores)**  
  - ROUGE-1: **0.652**  
  - ROUGE-2: **0.404**  
  - ROUGE-L: **0.503**  

Our approach outperforms traditional models such as **TF-IDF + TextRank** and **BERTSUM**.

---

## 🛠️ Tech Stack
- **Language:** Python 3.8+  
- **Frameworks:** Flask, PyTorch, TensorFlow  
- **Libraries:** Hugging Face Transformers, NLTK, spaCy  
- **Document Handling:** PyMuPDF, python-docx, pdfminer.six  

---

## 📥 Installation
```bash
# Clone the repository
git clone https://github.com/muralidharpadmanabhuni/keyword-identification-and-summarization-in-a-text.git
cd keyword-identification-and-summarization-in-a-text

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
