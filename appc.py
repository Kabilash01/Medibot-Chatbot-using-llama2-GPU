from flask import Flask, render_template, request, jsonify, url_for, session
from gtts import gTTS
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from accelerate import Accelerator
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '458f7b2e81ca56e59dc295bbb2311e79')  # Needed for session management

# Load environment variables
load_dotenv()

# Set up Pinecone API
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name = "medibot"

# Load the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Setup Prompt Template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

# Initialize Accelerator
accelerator = Accelerator()

# Configuration for the model
config = {
    'max_new_tokens': 512,
    'temperature': 0.8,
    'gpu_layers': 15  # Ensure this is set to a non-zero value for GPU usage
}

# Load the model
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    gpu_layers=config['gpu_layers'],  # This should ensure GPU usage if a compatible GPU is available
    config=config
)

# Prepare the model and configuration with Accelerator
llm, config = accelerator.prepare(llm, config)

# Setup RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Set this to a directory you have access to

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def save_chat_log(input_text, output_text):
    log_file = 'chat_log.txt'
    with open(log_file, 'a') as f:
        f.write(f"Input: {input_text}\n")
        f.write(f"Output: {output_text}\n")
        f.write("-" * 40 + "\n")

@app.route("/")
def index():
    # Initialize session history
    if 'history' not in session:
        session['history'] = []
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # Retrieve previous messages from session history
    history = session.get('history', [])

    # Append new message to history
    history.append({"user": msg})

    # Combine history into context for prompt
    context = "\n".join(f"User: {entry['user']}\nBot: {entry.get('bot', '')}" for entry in history)
    result = qa.invoke({"query": msg, "context": context})
    response_text = result["result"]

    # Add bot's response to history
    history[-1]["bot"] = response_text
    session['history'] = history

    # Convert the text response to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    audio_file = "response.mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
    tts.save(audio_path)

    # Save the chat log
    save_chat_log(msg, response_text)

    # Construct the URL for the audio file to send to the frontend
    audio_url = url_for('static', filename='uploads/' + audio_file)

    # Return both text and the audio file URL
    return jsonify({"text": response_text, "audio_url": audio_url})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
