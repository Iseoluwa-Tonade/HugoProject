import os
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use an environment variable for the API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_pptx(filepath):
    text = ""
    prs = Presentation(filepath)
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_text_from_docx(filepath):
    text = ""
    doc = Document(filepath)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        return extract_text_from_pdf(filepath)
    elif filepath.endswith('.pptx'):
        return extract_text_from_pptx(filepath)
    elif filepath.endswith('.docx'):
        return extract_text_from_docx(filepath)
    else:
        return ""

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected for uploading'}), 400
    
    filepaths = []
    for file in files:
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filepaths.append(filepath)
            
    return jsonify({'message': 'Files successfully uploaded', 'filepaths': filepaths})

@app.route('/query', methods=['POST'])
def query_files():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not uploaded_files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    latest_file = max([os.path.join(app.config['UPLOAD_FOLDER'], f) for f in uploaded_files], key=os.path.getctime)
    
    try:
        text = extract_text_from_file(latest_file)
        if not text:
            return jsonify({'answer': 'Could not extract text from the file.'})
            
        response = model.generate_content(f"Based on the following text, answer the question: '{question}'\n\nText: {text}")
        return jsonify({'answer': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Routes to serve the frontend files
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run()
