from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import os
import PyPDF2
import docx
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
keyword_model_path = "./ner_model_output"
keyword_tokenizer = AutoTokenizer.from_pretrained(keyword_model_path)
keyword_model = AutoModelForTokenClassification.from_pretrained(keyword_model_path).to(device)

summary_model_path = "./xlsum_model_output"
summary_tokenizer = T5Tokenizer.from_pretrained(summary_model_path)
summary_model = T5ForConditionalGeneration.from_pretrained(summary_model_path).to(device)

keyword_extractor = pipeline(
    "ner",
    model=keyword_model,
    tokenizer=keyword_tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

STOPWORDS = {"as", "by", "is", "the", "to", "a", "it", "in", "of", "on", "for", "and", "with", "its",
             "this", "that", "but", "an", "be", "if", "or", "which", "all", "from", "are", "at", "has",
             "more", "not", "was", "were", "also", "other", "than", "data", "information", "based", "results"}

def clean_keywords(ner_results, original_text):
    keywords = []
    seen_words = set()
    original_words = set(re.findall(r'\b\w+\b', original_text.lower()))  

    for entity in ner_results:
        word = entity["word"].replace("##", "").strip().lower()
        if (entity["score"] > 0.7 and word not in STOPWORDS and len(word) > 2 and word in original_words
                and word not in seen_words):
            seen_words.add(word)
            keywords.append(word)

    return keywords[:15] if keywords else ["No significant keywords found"]

def extract_keywords(text):
    ner_results = keyword_extractor(text)
    return clean_keywords(ner_results, text)

def chunk_text(text, max_chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_summary(text, keywords):
    text_chunks = chunk_text(text, max_chunk_size=512)
    summary_chunks = []

    for chunk in text_chunks:
        input_text = f"Summarize focusing on: {', '.join(keywords)}. Text: {chunk}"
        inputs = summary_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        summary_ids = summary_model.generate(
            **inputs,
            max_length=150,
            min_length=50,
            num_beams=5,
            length_penalty=1.0,
            repetition_penalty=2.5
        )
        summary_chunks.append(summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    return " ".join(summary_chunks)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return " ".join([para.text for para in doc.paragraphs])

@app.route("/process", methods=["POST"])
def process_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    extracted_keywords = extract_keywords(text)
    summary = generate_summary(text, extracted_keywords)

    return jsonify({"words": extracted_keywords, "output": summary})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(filepath)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(filepath)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    extracted_keywords = extract_keywords(text)
    summary = generate_summary(text, extracted_keywords)

    return jsonify({"words": extracted_keywords, "output": summary})

if __name__ == "__main__":
    app.run(debug=True)
