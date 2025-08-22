"""
Enhanced Medibot Flask Application
Integrates performance optimization, multilingual support, and medical glossary
- Updated for new Pinecone SDK (no pinecone.init)
- Uses langchain_pinecone.PineconeVectorStore
- Removes Accelerator.prepare() for CTransformers (incompatible)
"""

import os
import re
import time
import logging
from typing import Dict, Any,Optional

from flask import Flask, render_template, request, jsonify, url_for, session
from gtts import gTTS
from dotenv import load_dotenv

# --- Custom project modules ---
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from src.performance_optimizer import (
    ModelOptimizer, response_cache, performance_monitor, timed_response
)
from src.medical_glossary import medical_glossary

# --- Core ML / LangChain ---
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# --- Pinecone (NEW SDK) ---
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ---------------------------------------------------------------------
# App + Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '458f7b2e81ca56e59dc295bbb2311e79')

# Load environment variables
load_dotenv()

# Pinecone env
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# Use these (optional) to control serverless location
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")        # "aws" or "gcp"
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-west-2")

# Index & Embedding dimension (all-MiniLM-L6-v2 => 384)
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "medibot")
EMBED_DIM = 384
EMBED_METRIC = "cosine"


class EnhancedMedibot:
    """Enhanced Medibot with performance optimization and multilingual support"""

    def __init__(self):
        self.gpu_available = self.check_gpu_availability()
        self.setup_embeddings()
        self.setup_vectorstore()
        self.setup_optimized_model()
        self.setup_qa_chain()

        # Configure upload folder
        app.config['UPLOAD_FOLDER'] = 'static/uploads'
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    def check_gpu_availability(self) -> bool:
        """Check and log GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                props = torch.cuda.get_device_properties(current_device)
                memory_total = props.total_memory / 1e9
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1e9

                logger.info("🎯 GPU Status: AVAILABLE")
                logger.info(f"   • GPU Count: {gpu_count}")
                logger.info(f"   • Current GPU: {gpu_name}")
                logger.info(f"   • Total Memory: {memory_total:.1f} GB")
                logger.info(f"   • Available Memory (approx): {memory_total - memory_reserved:.1f} GB")

                if memory_total >= 8:
                    logger.info("   • Configuration: High-performance (8GB+ GPU)")
                elif memory_total >= 4:
                    logger.info("   • Configuration: Medium-performance (4–8GB GPU)")
                else:
                    logger.info("   • Configuration: Basic (<4GB GPU)")
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

    def setup_vectorstore(self):
        """Setup Pinecone vector store (new SDK)"""
        logger.info("Initializing Pinecone (new client)...")

        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set")

        pc = PineconeClient(api_key=PINECONE_API_KEY)

        # Create index if missing
        existing_names = {idx.name for idx in pc.list_indexes()}
        if PINECONE_INDEX_NAME not in existing_names:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (serverless {PINECONE_CLOUD}:{PINECONE_REGION})...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBED_DIM,
                metric=EMBED_METRIC,
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )

        # (Optional) sanity check dimension/metric to avoid hard-to-debug runtime errors
        try:
            desc = pc.describe_index(PINECONE_INDEX_NAME)
            dim = getattr(desc, "dimension", None) or getattr(desc, "config", {}).get("dimension")
            met = getattr(desc, "metric", None) or getattr(desc, "config", {}).get("metric")
            if dim and dim != EMBED_DIM:
                logger.warning(f"⚠️  Index dimension is {dim}, but embeddings are {EMBED_DIM}. Ensure your indexed vectors match current embedding model.")
            if met and met != EMBED_METRIC:
                logger.warning(f"⚠️  Index metric is '{met}', expected '{EMBED_METRIC}'.")
        except Exception as e:
            logger.warning(f"Could not describe Pinecone index: {e}")

        # LangChain vector store
        self.docsearch = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            text_key="text",  # must match your ingested metadata key for raw text
        )
        logger.info("Vector store initialized")

    def setup_optimized_model(self):
        """Setup optimized LLaMA model"""
        logger.info("Loading optimized model configuration...")

        self.config = ModelOptimizer.get_optimized_config()
        self.retrieval_config = ModelOptimizer.optimize_retrieval_config()

        # If no GPU, ensure gpu_layers=0 for ctransformers
        if not self.gpu_available:
            self.config["gpu_layers"] = 0

        self.llm = CTransformers(
            model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            gpu_layers=self.config.get('gpu_layers', 0),
            config={
                'max_new_tokens': self.config.get('max_new_tokens', 256),
                'temperature': self.config.get('temperature', 0.2),
                'context_length': self.config.get('context_length', 2048),
                'threads': self.config.get('threads', max(os.cpu_count() or 4, 4)),
                'batch_size': self.config.get('batch_size', 1),
            }
        )
        logger.info(f"Model loaded with optimized config: {self.config}")

    def setup_qa_chain(self):
        """Setup optimized QA chain"""
        self.prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": self.prompt_template}

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.docsearch.as_retriever(**self.retrieval_config),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        logger.info("QA chain initialized with optimized retrieval")

    @timed_response
    def get_response(self, query: str, language: str = 'en') -> Dict[str, Any]:
        """Get response with performance monitoring and multilingual support"""
        cached_response = response_cache.get(query)
        if cached_response:
            logger.info("Cache hit - returning cached response")
            performance_monitor.log_response_time(query, 0.1, cache_hit=True)
            return {
                'text': cached_response['response'],
                'cached': True,
                'language': language,
                'processing_time': 0.1
            }

        start_time = time.time()
        try:
            result = self.qa.invoke({"query": query})
            response_text = result["result"]

            # Add medical term markup for glossary
            enhanced_response = medical_glossary.add_hover_markup(response_text)

            processing_time = time.time() - start_time

            # Cache the response body (store wrapped text so cache returns exactly what UI expects)
            response_cache.set(query, enhanced_response)

            logger.info(f"Response generated in {processing_time:.2f} seconds")

            return {
                'text': enhanced_response,
                'cached': False,
                'language': language,
                'processing_time': processing_time,
                'source_documents': len(result.get("source_documents", []))
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'text': "I apologize, but I'm experiencing technical difficulties. Please try again.",
                'cached': False,
                'language': language,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def generate_audio(self, text: str, language: str = 'en') -> Optional[str]:
        """Generate audio response with language support"""
        try:
            tts_lang_map = {
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
                'pt': 'pt', 'nl': 'nl', 'ru': 'ru', 'ja': 'ja', 'ko': 'ko',
                'zh': 'zh', 'ar': 'ar', 'hi': 'hi', 'th': 'th'
            }
            tts_lang = tts_lang_map.get(language, 'en')

            # Clean text for TTS (remove HTML tags)
            clean_text = re.sub(r'<[^>]+>', '', text)

            tts = gTTS(text=clean_text, lang=tts_lang)
            audio_file = f"response_{int(time.time())}.mp3"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
            tts.save(audio_path)

            return url_for('static', filename=f'uploads/{audio_file}')
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None


# Initialize enhanced medibot
medibot = EnhancedMedibot()


def save_chat_log(input_text: str, output_text: str, language: str = 'en', processing_time: float = 0):
    """Enhanced chat logging with metadata"""
    log_file = 'chat_log.txt'
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Processing Time: {processing_time:.2f}s\n")
        f.write(f"Input: {input_text}\n")
        f.write(f"Output: {output_text}\n")
        f.write("-" * 60 + "\n")


@app.route("/")
def index():
    """Main chat interface"""
    if 'history' not in session:
        session['history'] = []

    supported_languages = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
        'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic',
        'hi': 'Hindi'
    }
    return render_template('enhanced_chat.html', languages=supported_languages)


@app.route("/get", methods=["POST"])
def chat():
    """Enhanced chat endpoint with multilingual and performance features"""
    try:
        msg = request.form["msg"]
        selected_language = request.form.get("language", "en")

        if not msg.strip():
            return jsonify({"error": "Empty message"}), 400

        response_data = medibot.get_response(msg, selected_language)

        audio_url = None
        if request.form.get("generate_audio", "false").lower() == "true":
            audio_url = medibot.generate_audio(response_data['text'], selected_language)

        save_chat_log(
            msg,
            response_data['text'],
            selected_language,
            response_data['processing_time']
        )

        history = session.get('history', [])
        history.append({
            "user": msg,
            "bot": response_data['text'],
            "language": selected_language,
            "timestamp": time.time(),
            "processing_time": response_data['processing_time'],
            "cached": response_data['cached']
        })
        session['history'] = history[-20:]  # Keep last 20 exchanges

        return jsonify({
            "text": response_data['text'],
            "audio_url": audio_url,
            "language": selected_language,
            "processing_time": response_data['processing_time'],
            "cached": response_data['cached'],
            "avg_response_time": performance_monitor.get_average_response_time()
        })

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/glossary/<term>")
def get_term_definition(term):
    """Get medical term definition"""
    definition = medical_glossary.get_definition(term)
    if definition:
        return jsonify(definition)
    return jsonify({"error": "Term not found"}), 404


@app.route("/search_terms")
def search_medical_terms():
    """Search medical terms"""
    query = request.args.get('q', '')
    if query:
        results = medical_glossary.search_terms(query)
        return jsonify(results)
    return jsonify([])


@app.route("/performance_stats")
def performance_stats():
    """Get performance statistics"""
    return jsonify({
        "average_response_time": performance_monitor.get_average_response_time(),
        "total_queries": len(performance_monitor.metrics),
        "cache_size": len(response_cache.cache)
    })


if __name__ == '__main__':
    logger.info("Starting Enhanced Medibot...")
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
