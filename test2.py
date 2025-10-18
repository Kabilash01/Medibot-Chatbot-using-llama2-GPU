from flask import Flask, render_template, request, jsonify, url_for
from gtts import gTTS
from googletrans import Translator
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize translator
translator = Translator()

# Load environment variables from .env
load_dotenv()

# Set up Pinecone API keys from environment
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not set in environment.")
    raise ValueError("PINECONE_API_KEY is required")

index_name = "medibot"

# Load embeddings
logger.info("Loading embeddings...")
embeddings = download_hugging_face_embeddings()
logger.info("Embeddings loaded.")

# Load Pinecone vectorstore from existing index
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    logger.info("Loaded Pinecone vectorstore from existing index '%s'.", index_name)
except Exception as e:
    logger.exception("Failed to load Pinecone vectorstore: %s", e)
    raise

# Setup Prompt Template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Configuration for the model
config = {
    'max_new_tokens': 256,
    'context_length': 2048,
    'temperature': 0.7,
    'top_p': 0.95,
    'top_k': 40,
    'repetition_penalty': 1.1,
    'last_n_tokens': 64,
    'seed': -1,
    'batch_size': 8,
    'threads': -1,
    'stop': ['</s>', 'User:', 'Human:']
}

# Initialize the CTransformers LLM
logger.info("Loading LLM model...")
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    config=config
)
logger.info("LLM model loaded.")

# Setup RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
logger.info("RetrievalQA chain created.")

# Configure the upload folder (for generated audio)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_language(text):
    """Detect the language of input text"""
    try:
        detection = translator.detect(text)
        return detection.lang
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return 'en'  # Default to English

def translate_text(text, src_lang, dest_lang):
    """Translate text from source language to destination language"""
    try:
        if src_lang == dest_lang:
            return text
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
        return translation.text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def clean_response(text):
    """Clean up the response text to remove repetitive or garbled output"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if 'expected symptoms such as' in line.lower():
            parts = line.split('expected symptoms such as')
            if parts[0].strip():
                cleaned_lines.append(parts[0].strip())
            break
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    if result and not result[-1] in '.!?':
        last_period = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_period > 0:
            result = result[:last_period + 1]
    
    return result

def save_chat_log(input_text, output_text, language):
    """Save chat conversation to log file"""
    log_file = 'chat_log.txt'
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Language: {language}\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Output: {output_text}\n")
            f.write("-" * 40 + "\n")
    except Exception as e:
        logger.exception("Failed to write chat log: %s", e)

@app.route("/")
def index():
    """Render the chat interface"""
    return render_template('chats.html')

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages and return responses with multilingual support"""
    try:
        msg = request.form["msg"]
        logger.info(f"Received message: {msg}")
        
        # Detect the language of the input message
        detected_lang = detect_language(msg)
        logger.info(f"Detected language: {detected_lang}")
        
        # Translate input to English if not already in English
        msg_english = translate_text(msg, detected_lang, 'en')
        logger.info(f"Translated to English: {msg_english}")
        
        # Get the response from the QA system (in English)
        result = qa.invoke({"query": msg_english})
        response_text_english = result["result"]
        response_text_english = clean_response(response_text_english)
        logger.info(f"Generated English response: {response_text_english}")
        
        # Translate response back to the user's language
        response_text = translate_text(response_text_english, 'en', detected_lang)
        logger.info(f"Translated response to {detected_lang}: {response_text}")
        
        # Convert the text response to speech using gTTS in the detected language
        # Map language codes for gTTS compatibility
        tts_lang = detected_lang
        if detected_lang == 'zh-cn':
            tts_lang = 'zh-CN'
        elif detected_lang == 'zh-tw':
            tts_lang = 'zh-TW'
        
        tts = gTTS(text=response_text, lang=tts_lang)
        audio_file = "response.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
        tts.save(audio_path)
        
        # Save the chat log
        save_chat_log(msg, response_text, detected_lang)
        
        # Construct the URL for the audio file
        audio_url = url_for('static', filename='uploads/' + audio_file)
        
        # Return JSON response with language info
        return jsonify({
            "text": response_text,
            "audio_url": audio_url,
            "detected_language": detected_lang
        })
    
    except Exception as e:
        logger.exception(f"Error in chat: {str(e)}")
        return jsonify({
            "text": "I'm having trouble processing that. Please try again.", 
            "audio_url": None
        }), 500

@app.route("/get_supported_languages", methods=["GET"])
def get_supported_languages():
    """Return list of supported languages"""
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'ur': 'Urdu',
        'tr': 'Turkish',
        'nl': 'Dutch',
        'pl': 'Polish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'id': 'Indonesian'
    }
    return jsonify(languages)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)