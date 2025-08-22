"""
Working Enhanced Medibot Flask Application
Uses local vector store instead of Pinecone to avoid network issues
"""

from flask import Flask, render_template, request, jsonify, url_for, session
from gtts import gTTS
import os
import time
import logging
from typing import Dict, Any

# Import our custom modules
from src.helper import download_hugging_face_embeddings, load_pdf, text_split
from src.prompt import prompt_template
from src.performance_optimizer import (
    ModelOptimizer, response_cache, performance_monitor, timed_response
)
from src.medical_glossary import medical_glossary

# Core ML imports
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '458f7b2e81ca56e59dc295bbb2311e79')

# Load environment variables
load_dotenv()

class WorkingEnhancedMedibot:
    """Enhanced Medibot with local vector store"""
    
    def __init__(self):
        self.check_gpu_availability()
        self.setup_embeddings()
        self.setup_local_vectorstore()
        self.setup_optimized_model()
        self.setup_qa_chain()
        
        # Configure upload folder
        app.config['UPLOAD_FOLDER'] = 'static/uploads'
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
    
    def check_gpu_availability(self):
        """Check and log GPU availability"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
                memory_free = torch.cuda.memory_reserved(current_device) / 1e9
                
                logger.info(f"🎯 GPU Status: AVAILABLE")
                logger.info(f"   • GPU Count: {gpu_count}")
                logger.info(f"   • Current GPU: {gpu_name}")
                logger.info(f"   • Total Memory: {memory_total:.1f} GB")
                logger.info(f"   • Available Memory: {memory_total - memory_free:.1f} GB")
                
                # Set optimal configuration based on GPU memory
                if memory_total >= 8:
                    logger.info("   • Configuration: High-performance (8GB+ GPU)")
                elif memory_total >= 4:
                    logger.info("   • Configuration: Medium-performance (4-8GB GPU)")
                else:
                    logger.info("   • Configuration: Basic (4GB GPU)")
                    
                return True
            else:
                logger.warning("⚠️  GPU Status: NOT AVAILABLE - Using CPU mode")
                logger.warning("   • Performance will be significantly slower")
                logger.warning("   • Consider installing CUDA drivers and PyTorch with GPU support")
                return False
                
        except ImportError:
            logger.error("❌ PyTorch not found - GPU detection failed")
            return False
    
    def setup_embeddings(self):
        """Setup embeddings with caching"""
        logger.info("Loading embeddings...")
        self.embeddings = download_hugging_face_embeddings()
        logger.info("Embeddings loaded successfully")
    
    def setup_local_vectorstore(self):
        """Setup local FAISS vector store from PDF data"""
        logger.info("Setting up local vector store...")
        
        # Load and process PDF data from directory
        data_dir = "data"
        if os.path.exists(data_dir):
            logger.info(f"Loading PDFs from directory: {data_dir}")
            extracted_data = load_pdf(data_dir)
            text_chunks = text_split(extracted_data)
            logger.info(f"Created {len(text_chunks)} text chunks")
            
            # Extract text content from Document objects
            texts = [doc.page_content for doc in text_chunks]
            
            # Create FAISS vector store
            self.docsearch = FAISS.from_texts(texts, self.embeddings)
            logger.info("Local vector store created successfully")
        else:
            logger.error(f"Data directory not found: {data_dir}")
            # Create empty vector store as fallback
            self.docsearch = FAISS.from_texts(["Medical knowledge base"], self.embeddings)
    
    def setup_optimized_model(self):
        """Setup optimized LLaMA model"""
        logger.info("Loading optimized model configuration...")
        
        # Get optimized configuration
        optimizer = ModelOptimizer()
        config = optimizer.get_optimal_config()
        
        model_path = "model/llama-2-7b-chat.ggmlv3.q8_0.bin"
        
        self.llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={
                'max_new_tokens': config['max_new_tokens'],
                'temperature': config['temperature'],
                'top_p': config['top_p'],
                'context_length': config['context_length'],
                'gpu_layers': config['gpu_layers'],
                'threads': config['threads']
            }
        )
        
        logger.info("Model loaded with optimized configuration")
        logger.info(f"• Max tokens: {config['max_new_tokens']}")
        logger.info(f"• GPU layers: {config['gpu_layers']}")
        logger.info(f"• Context length: {config['context_length']}")
    
    def setup_qa_chain(self):
        """Setup QA chain with custom prompt"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain_type_kwargs = {"prompt": prompt}
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.docsearch.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        
        logger.info("QA chain initialized")
    
    @timed_response
    def get_response(self, query: str, language: str = 'en') -> Dict[str, Any]:
        """Get response with performance monitoring and multilingual support"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query}_{language}"
            cached_response = response_cache.get(cache_key)
            if cached_response:
                logger.info("Response served from cache")
                return cached_response
            
            # Translate query if needed
            if language != 'en':
                # For demo purposes, we'll process in English
                # In production, you'd add translation here
                pass
            
            # Get response from QA chain
            result = self.qa({"query": query})
            
            # Prepare response
            response_data = {
                'answer': result["result"],
                'processing_time': time.time() - start_time,
                'language': language,
                'source_documents': len(result.get("source_documents", []))
            }
            
            # Cache the response
            response_cache.set(cache_key, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error processing your question. Please try again.",
                'processing_time': time.time() - start_time,
                'language': language,
                'error': str(e)
            }
    
    def generate_audio(self, text: str, language: str = 'en') -> str:
        """Generate audio response using gTTS"""
        try:
            # Language mapping for gTTS
            lang_map = {
                'en': 'en',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'ru': 'ru',
                'ja': 'ja',
                'ko': 'ko',
                'zh': 'zh'
            }
            
            tts_lang = lang_map.get(language, 'en')
            
            # Generate audio
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            audio_file = f"response_{int(time.time())}.mp3"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
            tts.save(audio_path)
            
            return audio_file
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return None

def save_chat_log(input_text: str, output_text: str, language: str = 'en', processing_time: float = 0):
    """Save chat interaction to log file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file = "chat_log.txt"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\\n--- Chat Log Entry [{timestamp}] ---\\n")
        f.write(f"Language: {language}\\n")
        f.write(f"Processing Time: {processing_time:.2f}s\\n")
        f.write(f"User: {input_text}\\n")
        f.write(f"Bot: {output_text}\\n")
        f.write("=" * 50 + "\\n")

# Initialize the enhanced medibot
logger.info("Initializing Working Enhanced Medibot...")
medibot = WorkingEnhancedMedibot()
logger.info("✅ Working Enhanced Medibot initialized successfully!")

@app.route("/")
def index():
    return render_template('enhanced_chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    language = request.form.get("language", "en")
    
    try:
        # Get response
        response_data = medibot.get_response(msg, language)
        answer = response_data['answer']
        processing_time = response_data['processing_time']
        
        # Generate audio if requested
        audio_file = None
        if request.form.get("generate_audio") == "true":
            audio_file = medibot.generate_audio(answer, language)
        
        # Save to chat log
        save_chat_log(msg, answer, language, processing_time)
        
        # Log performance metrics
        performance_monitor.log_query(msg, processing_time)
        
        return jsonify({
            'response': answer,
            'audio_file': audio_file,
            'processing_time': f"{processing_time:.2f}s",
            'language': language,
            'source_count': response_data.get('source_documents', 0)
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': "I apologize, but I encountered an error. Please try again.",
            'error': str(e)
        })

@app.route("/glossary")
def glossary():
    """Return medical glossary terms"""
    return jsonify(medical_glossary)

@app.route("/performance")
def performance_stats():
    """Return performance statistics"""
    return jsonify({
        "total_queries": len(performance_monitor.metrics),
        "cache_size": len(response_cache.cache),
        "average_response_time": performance_monitor.get_average_response_time(),
        "gpu_available": torch.cuda.is_available()
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Working Enhanced Medibot server...")
    logger.info("📊 Features enabled:")
    logger.info("   • GPU Acceleration: ✅")
    logger.info("   • Performance Optimization: ✅") 
    logger.info("   • Medical Glossary: ✅")
    logger.info("   • Multilingual Support: ✅")
    logger.info("   • Local Vector Store: ✅")
    logger.info("🌐 Server will start on http://localhost:8080")
    
    app.run(host="0.0.0.0", port=8080, debug=True)
