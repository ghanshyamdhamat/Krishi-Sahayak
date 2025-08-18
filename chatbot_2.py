import streamlit as st
import speech_recognition as sr
import pyttsx3
import threading
import tempfile
import os
import time
import traceback
import logging
import queue
from datetime import datetime
import torch
from neo4j_files.store_farmer_profile import StoreFarmerProfile

from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperProcessor,WhisperForConditionalGeneration, AutoModelForSeq2SeqLM,SeamlessM4Tv2ForSpeechToText, AutoProcessor,SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from langchain_core.messages import HumanMessage
from farmer_clean_agent import Qwen3LLM  # Import your LLM instance
import warnings
import soundfile as sf
import io
import numpy as np
warnings.filterwarnings("ignore")
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from PIL import Image
from farmer_clean_agent import create_agent, process_query, FarmerProfile
import torch
import re
import nltk
import pycld2
try:
    # This line will raise a LookupError if 'punkt' is not found
    nltk.data.find('tokenizers/punkt')
except LookupError:  # <-- Corrected exception
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')

# Your other code that uses nltk can now safely assume 'punkt' is available
from nltk.tokenize import sent_tokenize


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import optional dependencies
try:
    import librosa
except ImportError:
    logger.warning("librosa not installed, some audio features will be unavailable")


try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    logger.warning("IndicTransToolkit not installed, Indian language support will be limited")


import sys
logger.info(f'{sys.executable}')


import ast

def extract_triples(text: str) -> list:
    """
    Use LLM to extract (subject, predicate, object) triples from text.
    Returns a list of triples: [(subject, predicate, object), ...]
    """
    triple_prompt = f"""Extract all factual (subject, predicate, object) triples from the following text. 
Each triple should capture a distinct relationship or fact stated or implied in the text. 
Be concise and use canonical forms for entities and relations.

Text:
{text}

Instructions:
- Only include triples that are clearly supported by the text.
- Use short, meaningful predicates (e.g., "grows", "located_in", "suffers_from").
- If the subject is the farmer, use "Farmer" as the subject.
- If the text mentions a location, crop, disease, or practice, use those as entities.
- Return the result as a valid Python list of triples, e.g.:
[("Farmer", "grows", "Wheat"), ("Farmer", "located_in", "Karnataka")]
"""
    llm = Qwen3LLM()
    triples_str = llm.invoke({"messages": [HumanMessage(content=triple_prompt)]})
    try:
        triples = ast.literal_eval(triples_str)
        if isinstance(triples, list):
            return triples
    except Exception as e:
        logger.error(f"Triple extraction failed: {e}")
    return []

def triples_to_cypher(triples: list, farmer_id: str) -> list:
    """
    Convert triples to Cypher MERGE statements for Neo4j.
    """
    cyphers = []
    for s, p, o in triples:
        # Sanitize predicate for Cypher relationship
        rel = p.upper().replace(" ", "_")
        cypher = (
            f"MATCH (f:Farmer {{id: '{farmer_id}'}}) "
            f"MERGE (x:Entity {{name: '{s}'}}) "
            f"MERGE (y:Entity {{name: '{o}'}}) "
            f"MERGE (x)-[:{rel}]->(y)"
        )
        cyphers.append(cypher)
    return cyphers

def update_neo4j_with_triples(profile_store, cypher_statements: list):
    """
    Run Cypher MERGE statements in Neo4j.
    """
    with profile_store.driver.session() as session:
        for cypher in cypher_statements:
            try:
                session.run(cypher)
            except Exception as e:
                logger.error(f"Cypher execution failed: {e}")



class TranslationPipeline:
    """
    Standalone translation pipeline for integration with RAG systems
    Handles input translation (indic→en) and output translation (en→indic)
    """
    def __init__(self):
        # AI4Bharat IndicTrans2 Translation models
        self.input_translator = None  # indic-en-1B for input translation
        self.input_tokenizer = None
        self.input_translator_name = "prajdabre/rotary-indictrans2-indic-en-1B" 
        
        self.output_translator = None  # en-indic-1B for output translation
        self.output_tokenizer = None
        self.output_translator_name = "prajdabre/rotary-indictrans2-en-indic-1B"
        
        self.indic_processor = None  # IndicTransToolkit processor
        
        self.user_language = "en"  # Store user's detected language for output translation
        
        # DON'T initialize models on startup - load only when needed
        logger.info("✅ Translation pipeline configured (models will load on demand)")
    
    def initialize_translation_models(self):
        """Initialize AI4Bharat IndicTrans2 translation models for input and output"""
        try:
            logger.info("Loading AI4Bharat IndicTrans2 translation models...")
            
            # Initialize IndicProcessor for preprocessing
            try:
                self.indic_processor = IndicProcessor(inference=True)
            except Exception as e:
                logger.warning(f"IndicProcessor not available: {e}")
                return False
            
            # Load input translator (Indian languages to English)
            logger.info("Loading input translator (indic-en-1B)...")
            self.input_tokenizer = AutoTokenizer.from_pretrained(
                self.input_translator_name,
                trust_remote_code=True
            )
            
            self.input_translator = AutoModelForSeq2SeqLM.from_pretrained(
                self.input_translator_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            if torch.cuda.is_available():
                self.input_translator = self.input_translator.to("cuda")
            
            print("✅ Input translator (indic-en-1B) loaded successfully!")
            
            # Load output translator (English to Indian languages)
            print("Loading output translator (en-indic-1B)...")
            self.output_tokenizer = AutoTokenizer.from_pretrained(
                self.output_translator_name,
                trust_remote_code=True
            )
            
            self.output_translator = AutoModelForSeq2SeqLM.from_pretrained(
                self.output_translator_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            if torch.cuda.is_available():
                self.output_translator = self.output_translator.to("cuda")
            
            print("✅ Output translator (en-indic-1B) loaded successfully!")
            print("🎉 All AI4Bharat IndicTrans2 translation models loaded successfully!")
            
        except Exception as e:
            print(f"❌ AI4Bharat IndicTrans2 model initialization failed: {e}")
            print("⚠️ Translation functionality will not be available. Original text will be used.")
            self.input_translator = None
            self.input_tokenizer = None
            self.output_translator = None
            self.output_tokenizer = None
            self.indic_processor = None
    
    # def detect_language(self, text):
    #     """Detect the language of input text"""
    #     try:
    #         if fasttext_model:
    #             label = fasttext_model.predict(text.strip().replace('\n', ' '))[0][0]
    #             detected_lang = label.replace('__label__', '')
    #             return detected_lang
    #         else:
    #             from langdetect import detect
    #             return detect(text)
    #     except Exception as e:
    #         print(f"Language detection failed: {e}")
    #         return "en"  # Default to English
    def detect_language(self, text):
        """Detect the language of input text using pycld2"""
        try:
            isReliable, textBytesFound, details = pycld2.detect(text)
            # details[0][1] gives the ISO 639-1 code (e.g., 'hi', 'en', etc.)
            detected_lang = details[0][1]
            return detected_lang
        except Exception as e:
            print(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    def get_ai4bharat_language_code(self, detected_lang):
        """Map detected language to AI4Bharat language codes"""
        # Mapping from langdetect codes to AI4Bharat codes
        language_mapping = {
            'hi': 'hi-IN',    # Hindi
            'bn': 'bn-IN',    # Bengali
            'gu': 'gu-IN',    # Gujarati
            'kn': 'kn-IN',    # Kannada
            'ml': 'ml-IN',    # Malayalam
            'mr': 'mr-IN',    # Marathi
            'or': 'or-IN',    # Odia
            'pa': 'pa-IN',    # Punjabi
            'ta': 'ta-IN',    # Tamil
            'te': 'te-IN',    # Telugu
            'ur': 'ur-IN',    # Urdu
            'en': 'en-IN',    # English
        }
        return language_mapping.get(detected_lang, 'en-IN')
    
    def translate_to_english(self, text, source_language_code=None):
        """
        Translate input text to English using AI4Bharat IndicTrans2 (indic-en-1B)
        Main method for RAG integration - takes any Indian language text and returns English
        """
        if not source_language_code:
            # Auto-detect if not provided
            detected_lang = self.detect_language(text)
            # logger.info(f"🔄  \n\n\nDetected language: {detected_lang} \n\n")
            source_language_code = self.get_ai4bharat_language_code(detected_lang)
            # logger.info(f"🔄 Using AI4Bharat code: {source_language_code}")
            self.user_language = source_language_code  # Store for output translation
        
        if not self.input_translator or not self.input_tokenizer or not self.indic_processor:
            print("AI4Bharat input translation model not available. Using original text.")
            return text
        
        # If already in English, no need to translate
        if source_language_code in ['en-IN', 'en']:
            return text
        
        try:
            # Map language codes for IndicTrans2
            indic_lang_map = {
                'hi-IN': 'hin_Deva',  # Hindi
                'bn-IN': 'ben_Beng',  # Bengali
                'gu-IN': 'guj_Gujr',  # Gujarati
                'kn-IN': 'kan_Knda',  # Kannada
                'ml-IN': 'mal_Mlym',  # Malayalam
                'mr-IN': 'mar_Deva',  # Marathi
                'or-IN': 'ory_Orya',  # Odia
                'pa-IN': 'pan_Guru',  # Punjabi
                'ta-IN': 'tam_Taml',  # Tamil
                'te-IN': 'tel_Telu',  # Telugu
                'ur-IN': 'urd_Arab',  # Urdu
            }
            
            source_lang = indic_lang_map.get(source_language_code)
            target_lang = 'eng_Latn'  # English
            
            if not source_lang:
                print(f"Language {source_language_code} not supported by IndicTrans2. Using original text.")
                return text
            

            # Use IndicProcessor for proper preprocessing
            batch = self.indic_processor.preprocess_batch(
                [text],
                src_lang=source_lang,
                tgt_lang=target_lang,
            )
            
            # Tokenize the preprocessed sentences
            device = "cuda" if torch.cuda.is_available() and self.input_translator.device.type == "cuda" else "cpu"
            inputs = self.input_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.input_translator.generate(
                    **inputs,
                    use_cache=False,
                    length_penalty=1.5,
                    repetition_penalty=2.0,
                    max_new_tokens=2048,
                    num_beams=10,
                    num_return_sequences=1,
                )
            
            # Decode translation
            generated_tokens = self.input_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Postprocess the translations
            if generated_tokens and len(generated_tokens) > 0:
                translations = self.indic_processor.postprocess_batch(generated_tokens, lang=target_lang)
                translated_text = translations[0].strip() if translations else generated_tokens[0].strip()
                print(f"🔄 Input translation ({source_language_code} → English): '{text}' → '{translated_text}'")
                return translated_text
            else:
                print("Input translation returned empty result. Using original text.")
                return text
                
        except Exception as e:
            print(f"❌ Input translation failed: {e}")
            print("🔄 Using original text as fallback.")
            return text


    def translate_from_english(self, text, target_language_code=None):
        """
        Translate English text back to user's language using AI4Bharat IndicTrans2.
        Optimized for large texts using batch processing.
        """
        if not target_language_code:
            target_language_code = self.user_language
        
        if not self.output_translator or not self.output_tokenizer or self.indic_processor is None:
            print("AI4Bharat output translation model not available. Using original text.")
            return text
        
        if target_language_code in ['en-IN', 'en']:
            return text
    
        indic_lang_map = {
            'hi-IN': 'hin_Deva', 'bn-IN': 'ben_Beng', 'gu-IN': 'guj_Gujr',
            'kn-IN': 'kan_Knda', 'ml-IN': 'mal_Mlym', 'mr-IN': 'mar_Deva',
            'or-IN': 'ory_Orya', 'pa-IN': 'pan_Guru', 'ta-IN': 'tam_Taml',
            'te-IN': 'tel_Telu', 'ur-IN': 'urd_Arab',
        }
            
        source_lang = 'eng_Latn'
        target_lang = indic_lang_map.get(target_language_code)
        
        if not target_lang:
            print(f"Target language {target_language_code} not supported. Using original text.")
            return text
            
        # --- BATCHING LOGIC STARTS HERE ---
        
        # 1. Split the entire text into sentences using a robust tokenizer
        sentences = sent_tokenize(text)
        
        # 2. Group sentences into chunks for batch processing
        chunk_size = 16  # Key parameter to tune
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        
        all_translated_sentences = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # 3. Process each chunk in a batch
        for chunk in chunks:
            if not chunk:
                continue
                
            # The processor and tokenizer handle the whole batch of sentences at once
            batch = self.indic_processor.preprocess_batch(
                chunk,
                src_lang=source_lang,
                tgt_lang=target_lang,
            )
            
            inputs = self.output_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            
            with torch.no_grad():
                generated_tokens = self.output_translator.generate(
                    **inputs,
                    use_cache=False,
                    length_penalty=1.5,
                    repetition_penalty=2.0,
                    max_new_tokens=2048, # Max length per sentence
                    num_beams=10,
                    num_return_sequences=1,
                )
                
            decoded_tokens = self.output_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Post-process the entire batch of translations
            translations = self.indic_processor.postprocess_batch(decoded_tokens, lang=target_lang)
            
            # Add the translated sentences from this chunk to our final list
            all_translated_sentences.extend(translations)
    
        # 4. Join all translated sentences back into a single text
        return " ".join(all_translated_sentences)
    
    # def translate_from_english(self, text, target_language_code=None):
    #     """
    #     Translate English text back to user's language using AI4Bharat IndicTrans2 (en-indic-1B)
    #     Main method for RAG integration - takes English response and returns in user's language
    #     """
    #     if not target_language_code:
    #         target_language_code = self.user_language
        
    #     if not self.output_translator or not self.output_tokenizer or not self.indic_processor:
    #         print("AI4Bharat output translation model not available. Using original text.")
    #         return text
        
    #     # If target is English, no need to translate
    #     if target_language_code in ['en-IN', 'en']:
    #         return text

    #         # Map language codes for IndicTrans2
    #     indic_lang_map = {
    #             'hi-IN': 'hin_Deva',  # Hindi
    #             'bn-IN': 'ben_Beng',  # Bengali
    #             'gu-IN': 'guj_Gujr',  # Gujarati
    #             'kn-IN': 'kan_Knda',  # Kannada
    #             'ml-IN': 'mal_Mlym',  # Malayalam
    #             'mr-IN': 'mar_Deva',  # Marathi
    #             'or-IN': 'ory_Orya',  # Odia
    #             'pa-IN': 'pan_Guru',  # Punjabi
    #             'ta-IN': 'tam_Taml',  # Tamil
    #             'te-IN': 'tel_Telu',  # Telugu
    #             'ur-IN': 'urd_Arab',  # Urdu
    #         }
            
    #     source_lang = 'eng_Latn'  # English
    #     target_lang = indic_lang_map.get(target_language_code)
        
    #     if not target_lang:
    #         print(f"Target language {target_language_code} not supported by IndicTrans2. Using original text.")
    #         return text
            
    #     # --- CHUNKING LOGIC STARTS HERE ---
    #     import re
    #     # Split text into sentences (or paragraphs if you prefer)
    #     sentences = re.split(r'(?<=[.!?])\s+', text)
    #     translated_chunks = []

    #     for sentence in sentences:
    #         if not sentence.strip():
    #             continue
    #         batch = self.indic_processor.preprocess_batch(
    #             [sentence],
    #             src_lang=source_lang,
    #             tgt_lang=target_lang,
    #         )
    #         device = "cuda" if torch.cuda.is_available() and self.output_translator.device.type == "cuda" else "cpu"
    #         inputs = self.output_tokenizer(
    #             batch,
    #             truncation=True,
    #             padding="longest",
    #             return_tensors="pt",
    #             return_attention_mask=True,
    #         ).to(device)
    #         with torch.no_grad():
    #             generated_tokens = self.output_translator.generate(
    #                 **inputs,
    #                 use_cache=True,
    #                 min_length=0,
    #                 max_length=2048,
    #                 num_beams=5,
    #                 num_return_sequences=1,
    #             )
    #         generated_tokens = self.output_tokenizer.batch_decode(
    #             generated_tokens,
    #             skip_special_tokens=True,
    #             clean_up_tokenization_spaces=True,
    #         )
    #         if generated_tokens and len(generated_tokens) > 0:
    #             translations = self.indic_processor.postprocess_batch(generated_tokens, lang=target_lang)
    #             translated_text = translations[0].strip() if translations else generated_tokens[0].strip()
    #             translated_chunks.append(translated_text)
    #         else:
    #             translated_chunks.append(sentence)  # Fallback to original

    #     # Join translated chunks
    #     return " ".join(translated_chunks)
    # except Exception as e:
    #     print(f"❌ Output translation failed: {e}")
    #     print("🔄 Using original text as fallback.")
    #     return text
            
        #     # Use IndicProcessor for proper preprocessing
        #     batch = self.indic_processor.preprocess_batch(
        #         [text],
        #         src_lang=source_lang,
        #         tgt_lang=target_lang,
        #     )
            
        #     # Tokenize the preprocessed sentences
        #     device = "cuda" if torch.cuda.is_available() and self.output_translator.device.type == "cuda" else "cpu"
        #     inputs = self.output_tokenizer(
        #         batch,
        #         truncation=True,
        #         padding="longest",
        #         return_tensors="pt",
        #         return_attention_mask=True,
        #     ).to(device)
            
        #     # Generate translation
        #     with torch.no_grad():
        #         generated_tokens = self.output_translator.generate(
        #             **inputs,
        #             use_cache=True,
        #             min_length=0,
        #             max_length=256,
        #             num_beams=5,
        #             num_return_sequences=1,
        #         )
            
        #     # Decode translation
        #     generated_tokens = self.output_tokenizer.batch_decode(
        #         generated_tokens,
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=True,
        #     )
            
        #     # Postprocess the translations
        #     if generated_tokens and len(generated_tokens) > 0:
        #         translations = self.indic_processor.postprocess_batch(generated_tokens, lang=target_lang)
        #         translated_text = translations[0].strip() if translations else generated_tokens[0].strip()
        #         print(f"🔄 Output translation (English → {target_language_code}): '{text}' → '{translated_text}'")
        #         return translated_text
        #     else:
        #         print("Output translation returned empty result. Using original text.")
        #         return text
                
        # except Exception as e:
        #     print(f"❌ Output translation failed: {e}")
        #     print("🔄 Using original text as fallback.")
        #     return text
    
    def process_query_for_rag(self, user_query):
        """
        Process user query for RAG system integration
        Returns: (english_query, user_language_code)
        """
        # Detect language and store for output translation
        detected_lang = self.detect_language(user_query)
        logger.info(f"🔄 \n\n\n\Detected language: {detected_lang}\n\n\n\n")
        user_lang_code = self.get_ai4bharat_language_code(detected_lang)
        self.user_language = user_lang_code
        
        # Translate to English for RAG processing
        english_query = self.translate_to_english(user_query, user_lang_code)
        
        return english_query, user_lang_code
    
    def process_response_from_rag(self, english_response, target_language_code=None):
        """
        Process RAG system response back to user's language
        """
        return self.translate_from_english(english_response, target_language_code)

class ChatBot:
    def __init__(self):
        self.tts_engine = None
        self.recognizer = sr.Recognizer()
        
        # External LLM model (placeholder for any LLM)
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_model_name = None
        
        # Legacy Speech-to-Text model (AI4Bharat) - kept for backward compatibility
        self.stt_model = None
        self.stt_processor = None
        self.stt_model_name = None
        
        # NEW: SeamlessM4Tv2 Speech-to-Text model
        self.seamless_stt_model = None
        self.seamless_stt_processor = None
        self.seamless_stt_model_name = "facebook/seamless-m4t-v2-large"
        
        # NEW: SpeechT5 TTS models
        self.tts_model = None
        self.tts_processor = None
        self.tts_vocoder = None
        self.tts_model_name = "microsoft/speecht5_tts"

        self.soil_model = None
        self.rice_disease_model = None
        self.soil_model_name = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Capital_One/capital_one_agent_ai/backend/model/soil_classification_model.pth"  # IMPORTANT: Update this path
        self.rice_disease_model_name = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Capital_One/capital_one_agent_ai/backend/model/paddy_disease_best_model.pth" # IMPORTANT: Update this path

        
        # Translation pipeline for RAG integration
        self.translation_pipeline = TranslationPipeline()
        
        # Model loading configuration
        self.auto_load_models = False  # Changed to False for on-demand loading
        self.model_cache_timeout = 300  # 5 minutes timeout for unused models
        self.last_model_use = {}  # Track when models were last used
        
        # Only initialize TTS engine (lightweight)
        self.load_tts_engine()
    
    def free_gpu_memory(self):
        """Free up GPU memory by clearing cached tensors"""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            if torch.cuda.is_available():
                # Get memory info before cleanup
                allocated_before = torch.cuda.memory_allocated(0) / 1024**3
                
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Get memory info after cleanup
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                freed = allocated_before - allocated_after
                
                logger.info(f"🗑️ GPU memory cleaned: {freed:.2f} GB freed (was {allocated_before:.2f} GB, now {allocated_after:.2f} GB)")
            else:
                logger.info("🗑️ CPU memory garbage collection completed")
                
        except Exception as e:
            logger.warning(f"Could not clear GPU cache: {e}")
    
    def unload_model(self, model_type):
        """Unload specific model to free memory"""
        try:
            if model_type == "llm":
                if self.llm_model:
                    del self.llm_model
                    del self.llm_tokenizer
                    self.llm_model = None
                    self.llm_tokenizer = None
                    logger.info("🗑️ LLM model unloaded")
            
            elif model_type == "seamless_stt":
                if self.seamless_stt_model:
                    del self.seamless_stt_model
                    del self.seamless_stt_processor
                    self.seamless_stt_model = None
                    self.seamless_stt_processor = None
                    logger.info("🗑️ SeamlessM4Tv2 STT model unloaded")
            
            elif model_type == "tts":
                if self.tts_model:
                    del self.tts_model
                    del self.tts_processor
                    del self.tts_vocoder
                    self.tts_model = None
                    self.tts_processor = None
                    self.tts_vocoder = None
                    logger.info("🗑️ SpeechT5 TTS model unloaded")
            
            elif model_type == "legacy_stt":
                if self.stt_model:
                    del self.stt_model
                    del self.stt_processor
                    self.stt_model = None
                    self.stt_processor = None
                    logger.info("🗑️ Legacy STT model unloaded")
            
            # Clear GPU cache after unloading
            self.free_gpu_memory()
            
        except Exception as e:
            logger.error(f"Error unloading {model_type} model: {e}")
    
    def cleanup_unused_models(self):
        """Clean up models that haven't been used recently"""
        current_time = time.time()
        
        for model_type, last_use in self.last_model_use.items():
            if current_time - last_use > self.model_cache_timeout:
                logger.info(f"⏰ Model {model_type} unused for {self.model_cache_timeout}s, unloading...")
                self.unload_model(model_type)
                del self.last_model_use[model_type]
    
    def force_cleanup_all_models(self):
        """Force cleanup of all models to free maximum memory"""
        logger.info("🧹 Force cleaning all models for memory optimization")
        
        # Unload all models regardless of usage time
        models_to_unload = list(self.last_model_use.keys())
        
        for model_type in models_to_unload:
            self.unload_model(model_type)
        
        # Clear the usage tracking
        self.last_model_use.clear()
        
        # Aggressive memory cleanup
        self.free_gpu_memory()
        
        # Additional memory check
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            logger.info(f"💾 Available GPU memory after cleanup: {available_memory / 1024**3:.2f} GB")
    
    def load_llm_model_on_demand(self, llm_model_path=None):
        """Load LLM model only when needed"""
        if self.llm_model is not None:
            self.last_model_use["llm"] = time.time()
            return True
        
        if not llm_model_path:
            llm_model_path = "Qwen/Qwen3-8B"  # Default model
        
        try:
            logger.info(f"📥 Loading LLM model on demand: {llm_model_path}")
            
            # Force cleanup ALL models to make maximum space for the LLM
            self.force_cleanup_all_models()
            
            # Additional memory check
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                required_memory = 8 * 1024**3  # Estimate 8GB needed for Qwen3-8B
                
                if available_memory < required_memory:
                    logger.warning(f"⚠️ Low GPU memory: {available_memory / 1024**3:.2f} GB available, {required_memory / 1024**3:.2f} GB required")
                else:
                    logger.info(f"✅ Sufficient GPU memory: {available_memory / 1024**3:.2f} GB available")
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path, 
                trust_remote_code=True
            )
            
            # Try different model architectures with optimized memory usage
            # Try to load with FP8 quantization first
            try:
                # Try Seq2Seq first
                try:
                    from transformers import BitsAndBytesConfig
                    
                    # Configure 4-bit quantization (closest to FP8)
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,  # Use 4-bit for FP8-like precision
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                        llm_model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("📝 Loaded as Seq2Seq model with 4-bit quantization (FP8-like)")
                except:
                    # Fallback to CausalLM with quantization
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("📝 Loaded as CausalLM model with 4-bit quantization (FP8-like)")
                    
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, using float16 (FP8-like) fallback")
                # Fallback without quantization but with FP8-like precision
                try:
                    self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                        llm_model_path,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,  # FP8-like
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("📝 Loaded as Seq2Seq model with float16 (FP8-like)")
                except:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,  # FP8-like
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("📝 Loaded as CausalLM model with float16 (FP8-like)")
            
            self.llm_model_name = llm_model_path
            self.last_model_use["llm"] = time.time()
            
            # Log final memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"✅ LLM model loaded successfully! GPU memory used: {allocated:.2f} GB")
            else:
                logger.info(f"✅ LLM model loaded successfully on CPU!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading LLM model on demand: {e}")
            return False
    
    def load_seamless_stt_on_demand(self):
        """Load SeamlessM4Tv2 STT model only when needed"""
        if self.seamless_stt_model is not None:
            self.last_model_use["seamless_stt"] = time.time()
            return True
        
        try:
            logger.info("📥 Loading SeamlessM4Tv2 STT model on demand...")
            
            # Clean up other models to make space
            self.cleanup_unused_models()
            
            self.seamless_stt_processor = AutoProcessor.from_pretrained(self.seamless_stt_model_name)
            self.seamless_stt_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                self.seamless_stt_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            if torch.cuda.is_available():
                self.seamless_stt_model = self.seamless_stt_model.to("cuda")
            
            self.last_model_use["seamless_stt"] = time.time()
            logger.info("✅ SeamlessM4Tv2 STT model loaded successfully on demand!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load SeamlessM4Tv2 model on demand: {e}")
            return False
    
    def load_tts_model_on_demand(self):
        """Load SpeechT5 TTS model only when needed"""
        if self.tts_model is not None:
            self.last_model_use["tts"] = time.time()
            return True
        
        try:
            logger.info("📥 Loading SpeechT5 TTS model on demand...")
            
            # Clean up other models to make space
            self.cleanup_unused_models()
            
            fallback_tts_model = "microsoft/speecht5_tts"
            self.tts_processor = SpeechT5Processor.from_pretrained(fallback_tts_model)
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(fallback_tts_model)
            self.tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            if torch.cuda.is_available():
                self.tts_model = self.tts_model.to("cuda")
                self.tts_vocoder = self.tts_vocoder.to("cuda")
            
            self.last_model_use["tts"] = time.time()
            logger.info(f"✅ SpeechT5 TTS model loaded successfully on demand!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load TTS model on demand: {e}")
            return False
    
    def ensure_translation_models_loaded(self):
        """Ensure translation models are loaded when needed"""
        if (not self.translation_pipeline.input_translator or 
            not self.translation_pipeline.output_translator):
            logger.info("🔄 Reloading translation models for text processing...")
            self.translation_pipeline.initialize_translation_models()
    
    def load_legacy_stt_on_demand(self, stt_model_path=None):
        """Load legacy STT model only when needed"""
        if self.stt_model is not None:
            self.last_model_use["legacy_stt"] = time.time()
            return True
        
        if not stt_model_path:
            stt_model_path = "openai/whisper-medium"  # Default model
        
        try:
            logger.info(f"📥 Loading legacy STT model on demand: {stt_model_path}")
            
            # Clean up other models to make space
            self.cleanup_unused_models()
            
            self.stt_processor = WhisperProcessor.from_pretrained(stt_model_path, local_files_only=False)
            self.stt_model = WhisperForConditionalGeneration.from_pretrained(
                stt_model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=False
            )
            
            self.stt_model_name = stt_model_path
            self.last_model_use["legacy_stt"] = time.time()
            logger.info(f"✅ Legacy STT model loaded successfully on demand!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading legacy STT model on demand: {e}")
            return False
    
    def load_tts_engine(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"TTS engine initialization failed: {e}")
            else:
                logger.error(f"TTS engine initialization failed: {e}")
    
    # Legacy methods for backward compatibility - delegate to translation pipeline
    @property
    def user_language(self):
        return self.translation_pipeline.user_language
    
    @user_language.setter
    def user_language(self, value):
        self.translation_pipeline.user_language = value
    
    def initialize_translation_models(self):
        """Legacy method - reinitialize translation pipeline"""
        self.translation_pipeline.initialize_translation_models()
    
    def detect_language(self, text):
        """Legacy method - delegate to translation pipeline"""
        return self.translation_pipeline.detect_language(text)
    
    def get_ai4bharat_language_code(self, detected_lang):
        """Legacy method - delegate to translation pipeline"""
        return self.translation_pipeline.get_ai4bharat_language_code(detected_lang)
    
    def translate_input_to_english(self, text, source_language_code):
        """Legacy method - delegate to translation pipeline"""
        return self.translation_pipeline.translate_to_english(text, source_language_code)
    
    def translate_output_to_user_language(self, text, target_language_code):
        """Legacy method - delegate to translation pipeline"""
        return self.translation_pipeline.translate_from_english(text, target_language_code)
    
    def process_input_with_translation(self, user_input):
        """Legacy method - process user input with language detection and translation to English"""
        # Ensure translation models are loaded
        self.ensure_translation_models_loaded()
        
        english_query, user_lang_code = self.translation_pipeline.process_query_for_rag(user_input)
        
        # For backward compatibility, also return detected language
        detected_lang = user_lang_code.split('-')[0] if '-' in user_lang_code else user_lang_code
        
        if hasattr(st, 'info'):
            st.info(f"Detected language: {detected_lang} (AI4Bharat code: {user_lang_code})")
        
        return english_query, detected_lang, user_lang_code
    
    def load_models(self, llm_model_path=None, stt_model_path=None):
        """Configure model paths for on-demand loading (no longer loads immediately)"""
        success = True
        
        # Configure LLM model path for on-demand loading
        if llm_model_path:
            self.llm_model_name = llm_model_path
            # Clear any existing model to force reload
            self.unload_model("llm")
            if hasattr(st, 'info'):
                st.info(f"� LLM model configured for on-demand loading: {llm_model_path}")
        
        # Configure STT model path for on-demand loading
        if stt_model_path:
            self.stt_model_name = stt_model_path
            # Clear any existing model to force reload
            self.unload_model("legacy_stt")
            if hasattr(st, 'info'):
                st.info(f"📋 STT model configured for on-demand loading: {stt_model_path}")
        
        if hasattr(st, 'success'):
            st.success("✅ Models configured for on-demand loading! They will load when first used.")
        
        return success
        
        return success
    
    # RAG Integration Methods
    def prepare_query_for_rag(self, user_query):
        """
        Main method for RAG integration - prepares user query for RAG system
        Returns: (english_query, user_language_code)
        Usage: english_query, user_lang = chatbot.prepare_query_for_rag("आप कैसे हैं?")
        """
        return self.translation_pipeline.process_query_for_rag(user_query)
    
    def process_rag_response(self, english_response, target_language_code=None):
        """
        Main method for RAG integration - processes RAG response back to user language
        Usage: final_response = chatbot.process_rag_response(english_response, user_lang)
        """
        return self.translation_pipeline.process_response_from_rag(english_response, target_language_code)
    
    def get_translation_pipeline(self):
        """
        Get the standalone translation pipeline for external use
        Usage: translator = chatbot.get_translation_pipeline()
        """
        return self.translation_pipeline
    
    def generate_llm_response(self, english_input, max_length=200, use_agent=True):
        """
        Generate response using FarmerAgent with Langgraph for complex tasks
        Falls back to direct LLM for simple responses only if agent fails
        """
        
        # First try to use the farmer agent (recommended approach)
        if use_agent:
            try:
                logger.info("🤖 Using Farmer Agent with LangGraph...")
                
                # Create a basic farmer profile (you can enhance this with user data)
                # Create farmer profile from session state
                profile_data = st.session_state.get('farmer_profile', {})
                profile = FarmerProfile(
                    id=profile_data.get('username', ''),
                    name=profile_data.get('name', 'User'),
                    state=profile_data.get('state', ''),
                    taluka=profile_data.get('taluka', ''),
                    district=profile_data.get('district', ''),
                    village=profile_data.get('village', ''),
                    crops=profile_data.get('crops', []),
                    land_size_acres=profile_data.get('land_size'),
                    preferred_language=profile_data.get('preferred_language', 'en'),
                )
                
                # Create the agent (this handles all model loading internally)
                agent = create_agent()
                
                # Process the query using the full agent pipeline
                response = process_query(agent, english_input, profile)
                
                logger.info("✅ Farmer Agent response generated successfully")
                return response
                
            except ImportError as e:
                logger.error(f"Farmer Agent not available: {e}")
                logger.info("🔄 Falling back to direct LLM approach...")
            except Exception as e:
                logger.error(f"Farmer Agent error: {e}")
                logger.info("🔄 Falling back to direct LLM approach...")
        
        # Fallback: Direct LLM approach (only if agent fails)
        logger.info("🔄 Using direct LLM fallback...")
        
        # Load LLM model on demand for fallback
        if not self.load_llm_model_on_demand():
            return f"I received your message: '{english_input}'. Unable to load any models. Please try again."
        
        try:
            # Basic LLM response generation (simplified version)
            if hasattr(self.llm_tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant for farmers in India. Provide clear, concise, and practical advice."},
                    {"role": "user", "content": english_input}
                ]
                prompt = self.llm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt = f"<|im_start|>user\n{english_input}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize and generate
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available() and hasattr(self.llm_model, 'device'):
                inputs = inputs.to(self.llm_model.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.1,
                    pad_token_id=self.llm_tokenizer.pad_token_id if self.llm_tokenizer.pad_token_id else self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
            else:
                response = response[len(self.llm_tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            return response.strip() if response.strip() else "I'm here to help with your farming questions."
            
        except Exception as e:
            logger.error(f"Direct LLM fallback failed: {e}")
            return f"Error occurred while generating response: {str(e)}"
    
    def get_complete_response(self, user_input, max_length=200, input_type="text"):
        """Complete pipeline: translate input → LLM → translate output"""
        # Unload speech models if using text input to free memory
        if input_type == "text":
            self.unload_model("seamless_stt")
            self.unload_model("legacy_stt")
            self.unload_model("tts")
            logger.info("📝 Text input detected - speech models unloaded to free memory")
        
        # Step 1: Process input and translate to English
        english_input, detected_lang, user_lang_code = self.process_input_with_translation(user_input)
        logger.info(f"🔄 Input processed: '{user_input}' → '{english_input}' (detected language: {detected_lang}, user language code: {user_lang_code})")
        
        # Step 2: Generate response using LLM
        english_response = self.generate_llm_response(english_input, max_length)
        logger.info(f"💬 LLM response generated: {english_response}")
        
        # Step 3: Translate response back to user's language
        final_response = self.translate_output_to_user_language(english_response, user_lang_code)
        indic_langs = {'hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te', 'ur'}
        if detected_lang in indic_langs:
            factory = IndicNormalizerFactory()
            normalizer = factory.get_normalizer(detected_lang)
            normalized_response = normalizer.normalize(final_response)
        else:
            normalized_response = final_response  # No normalization for English or unsupported languages

        logger.info(f"🔄 Final response translated to {user_lang_code}: {final_response}")

        return normalized_response, english_input, english_response
        
        # return normalized_response, english_input, english_response
    
    def rice_disease_classification(self, image: Image.Image):
        """
        Classifies the uploaded image of a rice plant to identify diseases.
        
        Args:
            image (PIL.Image.Image): The uploaded image of the rice plant."""
        from torchvision import models
        from torch import nn
        from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize,
    RandomHorizontalFlip, RandomRotation, RandomErasing,
    RandomAdjustSharpness, ColorJitter
)
        from PIL import Image
        LABEL_MAP = {
    'normal': 0,
    'bacterial_leaf_blight': 1,
    'bacterial_leaf_streak': 2,
    'bacterial_panicle_blight': 3,
    'blast': 4,
    'brown_spot': 5,
    'dead_heart': 6,
    'downy_mildew': 7,
    'hispa': 8,
    'tungro': 9
}

        if image.mode != "RGB":
            image = image.convert("RGB")

        transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        NUM_CLASSES = len(LABEL_MAP)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Load pretrained model in model

        model_instance = models.resnet34(pretrained=False)
        # Freeze all layers
        for param in model_instance.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (layer3 and layer4) for fine-tuning
        for param in model_instance.layer3.parameters():
            param.requires_grad = True
        for param in model_instance.layer4.parameters():
            param.requires_grad = True

        # Replace the final classifier to match the number of classes
        num_ftrs = model_instance.fc.in_features
        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

        state_dict = torch.load(self.rice_disease_model_name, map_location=device)

        model_instance.load_state_dict(state_dict)
        model_instance.to(device)
        model_instance.eval()  # Set the model to evaluation mode
        
        # Transform the image
        image_tensor = transforms(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
        image_tensor = image_tensor.to(device)


        #infer
        with torch.no_grad():
            output = model_instance(image_tensor)
        
        softmax_output = nn.Softmax(dim=1)(output)

        # Get the top 3 predicted classes
        _, predicted_indices = torch.topk(softmax_output, k=3, dim=1)
        predicted_classes = [list(LABEL_MAP.keys())[i] for i in predicted_indices[0].cpu().numpy()]
        logger.info(f"Predicted classes: {predicted_classes}")
        
        # Return the most probable class
        return predicted_classes

    def soil_classification(self, image: Image.Image):
        """
        Classifies the uploaded image of soil to identify its type.
        
        Args:
            image (PIL.Image.Image): The uploaded image of the soil sample.
        
        Returns:
            str: The classification label (e.g., "Red Soil") or None if failed.
        """
        from torchvision import models
        from torch import nn
        from torchvision.transforms import (
            Compose, ToTensor, Normalize, Resize
        )
        
        LABEL_MAP = {
    'alluvial_soil': 0,
    'black_soil': 1,
    'clay_soil': 2,
    'red_soil': 3,
}
        if image.mode != "RGB":
            image = image.convert("RGB")

        transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        NUM_CLASSES = len(LABEL_MAP)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained model
        model_instance = models.resnet34(pretrained=False)
        
        # Freeze all layers
        for param in model_instance.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (layer3 and layer4) for fine-tuning
        for param in model_instance.layer3.parameters():
            param.requires_grad = True
        for param in model_instance.layer4.parameters():
            param.requires_grad = True

        # Replace the final classifier to match the number of classes
        num_ftrs = model_instance.fc.in_features
        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES)
        )

        # Load the model state
        state_dict = torch.load(self.soil_model_name, map_location=device)
        model_instance.load_state_dict(state_dict)
        model_instance.to(device)
        model_instance.eval()  # Set the model to evaluation mode
        
        # Transform the image
        image_tensor = transforms(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            output = model_instance(image_tensor)
        print(f"Model output: {output}")
        softmax_output = nn.Softmax(dim=1)(output)

        # Get the top predicted class
        _, predicted_index = torch.max(softmax_output, dim=1)
        if _ < 0.5:
            logger.warning("Prediction confidence is low, returning None")
            predicted_class = 'unknown'
            return predicted_class
        predicted_class = list(LABEL_MAP.keys())[predicted_index.item()]
        
        logger.info(f"Predicted soil class: {predicted_class}")
        
        return predicted_class
    
    def classify_image(self, image: Image.Image, task: str):
        """
        Performs image classification and returns the resulting label.
        This is a helper function for the RAG pipeline.
        
        Returns:
            str: The classification label (e.g., "Red Soil") or None if failed.
        """
        logger.info(f"Performing image classification for task: {task}")
        
        # --- THIS IS A PLACEHOLDER for your actual model inference ---
        # You would implement your on-demand model loading and inference here.
        
        try:
            if task == "soil_classification":
                # Replace with your actual model prediction
                predicted_class = self.soil_classification(image)
                logger.info(f"Predicted soil class: {predicted_class}")
                return predicted_class
            elif task == "rice_disease_classification":
                # Replace with your actual model prediction
                # For example, using a pre-trained model to classify rice diseases
                
                predicted_class = self.rice_disease_classification(image)
                logger.info(f"Predicted rice disease class: {predicted_class}")
                return predicted_class
            else:
                logger.warning(f"Unknown image analysis task: {task}")
                return None
        except Exception as e:
            logger.error(f"Error during image classification: {e}")
            return None

    # [++ ADDED ++] New orchestrator method for the image RAG pipeline.
    def get_image_analysis_recommendations(self, image: Image.Image, task: str, max_length: int = 500):
        """
        Orchestrates the full pipeline: image classification -> RAG query -> final response.
        
        Args:
            image (PIL.Image.Image): The uploaded image.
            task (str): The classification task.
            max_length (int): Max length for the final LLM response.

        Returns:
            tuple: (final_recommendation, classification_label)
        """
        # Step 1: Get the classification label from the image.
        classification_label = self.classify_image(image, task)
        
        if not classification_label:
            error_message = "I'm sorry, I was unable to analyze the image. Please try again with a clearer picture."
            return error_message, None

        # Step 2: Construct a detailed prompt for the RAG system based on the classification.
        rag_query = ""
        if task == "soil_classification":
            rag_query = (
                f"An image of a soil sample has been identified as '{classification_label}'. "
                f"Based on this, provide a detailed set of recommendations for a farmer in India. Include:"
                f"\n1. Suitable crops for this soil type."
                f"\n2. Recommended soil management and amendment practices."
                f"\n3. Typical fertilization requirements."
            )
        elif task == "rice_disease_classification":
            rag_query = (
                f"An image of a rice plant has been diagnosed with one of the following '{classification_label}' diseases. "
                f"Provide a comprehensive action plan for a farmer in India for all these diseases. Include:"
                f"\n1. A brief description of the disease."
                f"\n2. Immediate control measures (both organic and chemical options)."
                f"\n3. Long-term prevention strategies for future crops."
            )
        
        logger.info(f"Constructed RAG query: {rag_query}")

        # Step 3: Feed the constructed query into the existing RAG pipeline.
        # The `get_complete_response` method will handle translation to English,
        # querying the LLM/RAG, and translating the response back to the user's language.
        final_recommendations, _, _ = self.get_complete_response(
            user_input=rag_query,
            max_length=max_length,
            input_type="text"  # We treat this as a text input to the RAG system
        )
        
        return final_recommendations, classification_label    

    def speech_to_text_seamless(self, audio_file):
        """Convert uploaded audio file to text using SeamlessM4Tv2 speech recognition model"""
        try:
            # Unload translation models when using speech input to free memory
            if hasattr(self.translation_pipeline, 'input_translator') and self.translation_pipeline.input_translator:
                logger.info("🎤 Speech input detected - temporarily unloading translation models to free memory")
                del self.translation_pipeline.input_translator
                del self.translation_pipeline.input_tokenizer
                del self.translation_pipeline.output_translator
                del self.translation_pipeline.output_tokenizer
                self.translation_pipeline.input_translator = None
                self.translation_pipeline.input_tokenizer = None
                self.translation_pipeline.output_translator = None
                self.translation_pipeline.output_tokenizer = None
                self.free_gpu_memory()
            
            # Load SeamlessM4Tv2 model on demand
            if not self.load_seamless_stt_on_demand():
                logger.warning("SeamlessM4Tv2 model could not be loaded, falling back to legacy ASR")
                return self.speech_to_text_ai4bharat(audio_file)
            
            # Load and process audio
            audio_data, sample_rate = sf.read(audio_file)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Process with SeamlessM4Tv2
            logger.info("Processing audio with SeamlessM4Tv2...")
            
            # Prepare inputs for the model
            inputs = self.seamless_stt_processor(
                audios=audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() and self.seamless_stt_model.device.type == "cuda" else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.seamless_stt_model.generate(**inputs, max_length=256)
            
            # Decode the transcription
            transcription = self.seamless_stt_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            logger.info(f"🎤 SeamlessM4Tv2 ASR: {transcription}")
            
            # For speech input, return raw transcription - translation will be handled later when needed
            # Don't immediately reload translation models to save memory
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error with SeamlessM4Tv2 speech recognition: {e}")
            # Fallback to legacy ASR
            return self.speech_to_text_ai4bharat(audio_file)
    
    def speech_to_text_ai4bharat(self, audio_file):
        """Convert uploaded audio file to text using legacy speech recognition model"""
        try:
            # Load legacy STT model on demand
            if not self.load_legacy_stt_on_demand():
                return self.speech_to_text_fallback(audio_file)
            
            # Load and resample audio
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            
            # Use Whisper model (legacy fallback)
            if not self.stt_processor:
                return "Speech-to-text processor not available."
            
            inputs = self.stt_processor(audio_data, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                predicted_ids = self.stt_model.generate(inputs["input_features"])
            transcription = self.stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            st.info(f"🎤 Legacy ASR transcription: {transcription}")
            
            # For speech input, return raw transcription - translation will be handled later when needed
            return transcription.strip()
            
        except Exception as e:
            st.error(f"Error with legacy speech recognition: {e}")
            return self.speech_to_text_fallback(audio_file)
    
    def speech_to_text_fallback(self, audio_file):
        """Fallback speech recognition using Google API"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                processed_text, _, _ = self.process_input_with_translation(text)
                return processed_text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"
        except Exception as e:
            return f"Error processing audio: {e}"
    
    def text_to_speech_indic_parler(self, text, language_code=None):
        """Convert text to speech using SpeechT5 TTS model with proper text length handling"""
        try:
            # Unload other models before loading TTS to save memory
            self.unload_model("seamless_stt")
            self.unload_model("legacy_stt")
            logger.info("🔊 TTS requested - speech recognition models unloaded to free memory")
            
            # Load TTS model on demand
            if not self.load_tts_model_on_demand():
                logger.warning("SpeechT5 model could not be loaded, falling back to legacy TTS")
                return self.text_to_speech(text)
            
            logger.info(f"Generating speech with SpeechT5")
            
            # Handle long text by chunking it
            max_length = 500  # Safe limit for SpeechT5 (well below 600 token limit)
            
            # Split text into sentences first
            import re
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if adding this sentence would exceed the limit
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                # Quick token estimate: roughly 4 characters per token
                if len(potential_chunk) * 0.25 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Single sentence is too long, truncate it
                        chunks.append(sentence[:max_length * 4].strip())
                        current_chunk = ""
                else:
                    current_chunk = potential_chunk
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If no chunks created, truncate the original text
            if not chunks:
                chunks = [text[:max_length * 4].strip()]
            
            logger.info(f"Split text into {len(chunks)} chunks for TTS processing")
            
            # Process each chunk
            all_audio_data = []
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                try:
                    logger.info(f"Processing TTS chunk {i+1}/{len(chunks)}: {len(chunk)} characters")
                    
                    # Process text with SpeechT5
                    inputs = self.tts_processor(text=chunk, return_tensors="pt")
                    
                    # Check input length
                    if inputs["input_ids"].shape[1] > 600:
                        logger.warning(f"Chunk {i+1} still too long ({inputs['input_ids'].shape[1]} tokens), truncating...")
                        # Truncate the input_ids to fit
                        inputs["input_ids"] = inputs["input_ids"][:, :600]
                        if "attention_mask" in inputs:
                            inputs["attention_mask"] = inputs["attention_mask"][:, :600]
                    
                    # Create default speaker embeddings (512-dimensional)
                    speaker_embeddings = torch.randn((1, 512))
                    
                    # Move to appropriate device
                    device = "cuda" if torch.cuda.is_available() and self.tts_model.device.type == "cuda" else "cpu"
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    speaker_embeddings = speaker_embeddings.to(device)
                    
                    # Generate speech
                    with torch.no_grad():
                        speech = self.tts_model.generate_speech(
                            inputs["input_ids"], 
                            speaker_embeddings, 
                            vocoder=self.tts_vocoder
                        )
                    
                    # Convert to numpy and store
                    audio_data = speech.cpu().numpy()
                    all_audio_data.append(audio_data)
                    
                    # Add small pause between chunks
                    if i < len(chunks) - 1:  # Don't add pause after last chunk
                        pause_duration = 0.3  # 300ms pause
                        pause_samples = int(16000 * pause_duration)  # 16kHz sample rate
                        pause = np.zeros(pause_samples)
                        all_audio_data.append(pause)
                        
                except Exception as chunk_error:
                    logger.error(f"Error processing TTS chunk {i+1}: {chunk_error}")
                    continue
            
            if not all_audio_data:
                logger.error("No audio data generated from any chunks")
                return self.text_to_speech(text)
            
            # Concatenate all audio chunks
            final_audio = np.concatenate(all_audio_data)
            
            # Save and play audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # SpeechT5 outputs at 16kHz
                sf.write(tmp_file.name, final_audio, 16000)
                
                # Play audio using streamlit
                with open(tmp_file.name, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                
                # Clean up
                os.unlink(tmp_file.name)
            
            logger.info("✅ Speech generated successfully with SpeechT5 (chunked)")
            
        except Exception as e:
            logger.error(f"Error with SpeechT5 TTS: {e}")
            # Fallback to legacy TTS
            self.text_to_speech(text)

    def text_to_speech(self, text):
        """Convert text to speech (legacy method)"""
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                st.error(f"Text-to-speech error: {e}")

def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    pwd = "test1234"
    profile_store = StoreFarmerProfile(uri=uri, user=user, password=pwd)
    st.set_page_config(
        page_title="AI Chatbot - Text, Speech & Vision",
        page_icon="🤖",
        layout="wide"
    )
    # Login screen
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("Login to SAHAYAK KRISHI")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.session_state.farmer_id = username
                # Fetch profile from DB after login
                fetched_profile = profile_store.get_profile(username)
                if fetched_profile:
                    # If your get_profile returns a Neo4j Record, convert to dict
                    if hasattr(fetched_profile, "data"):
                        fetched_profile = fetched_profile.data()
                    st.session_state.farmer_profile = fetched_profile
                else:
                    # If no profile found, initialize with defaults
                    st.session_state.farmer_profile = {
                        'username': username,
                        'name': '',
                        'state': '',
                        'district': '',
                        'taluka': '',
                        'village': '',
                        'country': 'India',
                        'crops': [],
                        'land_size': 0.0,
                        'preferred_language': 'en'
                    }
                    
                    st.success("Login successful!")
                    st.rerun()
            else:
                st.error("Invalid credentials, please try again.")
                st.stop()
    
    if st.session_state.logged_in:    
        st.title("🤖 SAHAYAK KRISHI")
        st.subheader("Hi, How may I assist you today?")

        
        # # Initialize chatbot in session state
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = ChatBot()
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Configure default model paths (for on-demand loading)
        if 'models_configured' not in st.session_state:
            st.session_state.models_configured = False
        
        # Default model paths
        # default_llm_model = "Qwen/Qwen3-8B"
        # default_stt_model = "openai/whisper-medium"
        
        # Configure model paths for on-demand loading (first time only)
        if not st.session_state.models_configured:
            # st.session_state.chatbot.load_models(
            #     llm_model_path=default_llm_model,
            #     stt_model_path=default_stt_model
            # )
            st.session_state.models_configured = True
            st.info("🎯 Models configured for on-demand loading - they will load when first used to save GPU memory!")
        
        # Sidebar for model selection and settings
        with st.sidebar:
            st.header("⚙️ Model Settings")
            
            st.info("💡 **On-Demand Loading**: Models load only when needed to save GPU memory!")
            
            st.subheader("🤖 External LLM Model")
            llm_model_options = ["Qwen/Qwen3-8B"]
            
            selected_llm_model = st.selectbox(
                "Select External LLM Model:",
                llm_model_options,
                index=0,
                help="Choose external LLM model for response generation (loads on first use)"
            )
            
            st.subheader("🎤 Speech-to-Text Model")
            stt_model_options = [
                "openai/whisper-medium",
                "openai/whisper-large-v3",
                "openai/whisper-small"
            ]
            
            selected_stt_model = st.selectbox(
                "Select ASR Model:",
                stt_model_options,
                index=0,
                help="Whisper model for speech recognition with fallback support"
            )
            
            if st.button("Configure Models", type="secondary"):
                success = st.session_state.chatbot.load_models(
                    llm_model_path=selected_llm_model,
                    stt_model_path=selected_stt_model
                )
                if success:
                    st.success("Model paths configured for on-demand loading!")
            
            # Memory Management Controls
            st.subheader("🗑️ Memory Management")
            col_mem1, col_mem2 = st.columns(2)
            
            with col_mem1:
                if st.button("🔄 Clear GPU Cache"):
                    st.session_state.chatbot.free_gpu_memory()
                    st.success("GPU cache cleared!")
            
            with col_mem2:
                if st.button("🗑️ Unload All Models"):
                    # Unload all relevant models, including the new image models
                    for model_type in ["llm", "legacy_stt", "soil", "rice_disease"]:
                        st.session_state.chatbot.unload_model(model_type)
                    st.success("All models unloaded!")
            
            # Display GPU memory info if available
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    st.metric("GPU Memory", f"{memory_allocated:.1f}GB / {memory_reserved:.1f}GB")
                except Exception as e:
                    st.warning(f"Could not retrieve GPU info: {e}")
            
            st.divider()
            
            # Image Models Status
            st.subheader("🖼️ Image Models (On-Demand)")
            soil_status = "✅ Loaded" if st.session_state.chatbot.soil_model else "⏳ Ready to Load"
            rice_status = "✅ Loaded" if st.session_state.chatbot.rice_disease_model else "⏳ Ready to Load"
            st.write(f"Soil Classification: {soil_status}")
            st.write(f"Rice Disease ID: {rice_status}")
            st.divider()

            # WebRTC Status Check
            st.subheader("🎤 Voice Recording Status")
            try:
                import streamlit_webrtc
                webrtc_status = "✅ Available"
            except ImportError:
                webrtc_status = "❌ Not Available"
            st.write(f"WebRTC Status: {webrtc_status}")
            if webrtc_status == "❌ Not Available":
                st.info("Install `streamlit-webrtc` to enable voice recording.")
            st.divider()

            # Show currently loaded models
            st.subheader("🖥️ GPU Memory Management")
            loaded_models = []
            if st.session_state.chatbot.llm_model:
                loaded_models.append("🤖 LLM")
            if st.session_state.chatbot.stt_model:
                loaded_models.append("🎤 Legacy STT")
            if st.session_state.chatbot.soil_model:
                loaded_models.append("🏞️ Soil Model")
            if st.session_state.chatbot.rice_disease_model:
                loaded_models.append("🌾 Rice Disease Model")
            
            if loaded_models:
                st.write("**Currently Loaded:**")
                for model in loaded_models:
                    st.write(f"• {model}")
            else:
                st.write("**Currently Loaded:** None (Memory Optimized)")
            
            st.divider()
            
            # Settings
            max_length = st.slider("Response Length", 50, 2000, 1500)
            
            # Farmer Profile Section
            st.subheader("👨‍🌾 Farmer Profile")
            if 'farmer_profile' not in st.session_state:
                st.session_state.farmer_profile = {
                    'username':'',
                    'name': '',
                    'state': '',
                    'district': '',
                    'taluka': '',
                    'village': '',
                    'country': '',
                    'crops': [],
                    'land_size': 0.0,
                    'preferred_language': 'en'
                }

            with st.expander("🔧 Setup Profile", expanded=False):
                # Use a form to prevent page rerun on every input change
                with st.form("farmer_profile_form"):
                    username = st.text_input("Username:", value=st.session_state.farmer_profile.get('username', ''))
                    name = st.text_input("Name:", value=st.session_state.farmer_profile.get('name', ''))
                    state = st.text_input("State:", value=st.session_state.farmer_profile.get('state', ''))
                    district = st.text_input("District:", value=st.session_state.farmer_profile.get('district', ''))
                    village = st.text_input("Village:", value=st.session_state.farmer_profile.get('village', ''))
                    taluka = st.text_input("Taluka:", value=st.session_state.farmer_profile.get('taluka', ''))
                    country = st.text_input("Country:", value=st.session_state.farmer_profile.get('country', 'India'))
                    preferred_language = st.selectbox(
                        "preferred Language:",
                        options=['en', 'hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te', 'ur'],
                        index=0 if st.session_state.farmer_profile.get('preferred_language') == 'en' else 1
                    )

                    crops_str = st.text_input("Crops (comma-separated):", 
                                            value=','.join(st.session_state.farmer_profile.get('crops', [])))
                    land_size = st.number_input("Land Size (acres):", 
                                            value=st.session_state.farmer_profile.get('land_size', 0.0), 
                                            min_value=0.0)
                    
                    # Form submit button - this prevents reruns
                    submitted = st.form_submit_button("Save Profile")
                    
                    if submitted:
                        st.session_state.farmer_profile = {
                            'username': username.strip(),
                            'name': name.strip(),
                            'state': state.strip(),
                            'district': district.strip(),
                            'crops': [c.strip() for c in crops_str.split(',') if c.strip()],
                            'land_size': land_size if land_size > 0 else 0.0,
                            'preferred_language': preferred_language,   
                            'taluka': taluka.strip(),
                            'village': village.strip(),
                            'country': country.strip()
                        }
                        # Store in Neo4j
                        try:
                            
                            #generate a unique ID for the profile
                            # import uuid
                            # # unique_id = str(uuid.uuid4())
                            profile_store.store_profile(username , st.session_state.farmer_profile)
                            st.success("Profile saved and stored in database!")
                        except Exception as e:
                            st.error(f"Profile saved locally, but failed to store in database: {e}")

            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Main chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header("💬 Chat Interface")
            
            # Input method selection
            input_options = ["Text Input", "Voice Recording", "Audio Upload", "Image Analysis"]
            input_method = st.radio(
                "Choose input method:",
                options=input_options,
                horizontal=True
            )
            
            user_input = ""
            uploaded_image = None
            analysis_task = None
            send_button_text = "Send Message"

            if input_method == "Text Input":
                user_input = st.text_area(
                    "Type your message:",
                    height=100,
                    placeholder="Enter your message here..."
                )
                
            elif input_method == "Voice Recording":
                st.warning("Voice recording feature requires `streamlit-webrtc`. This is a placeholder.")
                # Your WebRTC implementation would go here
                
            elif input_method == "Audio Upload":
                uploaded_file = st.file_uploader(
                    "Choose audio file",
                    type=['wav', 'mp3', 'm4a', 'flac']
                )
                if uploaded_file is not None:
                    with st.spinner("Processing audio..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name
                        user_input = st.session_state.chatbot.speech_to_text_seamless(tmp_file_path)
                        st.success(f"Transcription: {user_input}")
                        os.unlink(tmp_file_path)

            elif input_method == "Image Analysis":
                st.info("🖼️ Upload an image for soil or rice disease classification.")
                analysis_task = st.selectbox(
                    "Select Analysis Task:",
                    ("soil_classification", "rice_disease_classification"),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                uploaded_image = st.file_uploader(
                    "Choose an image file",type=['jpg', 'jpeg', 'png']
                )
                if uploaded_image:
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                send_button_text = "Analyze Image"
            disable_send = not user_input.strip()

            # Set disable_send appropriately
            if input_method == "Image Analysis":
                disable_send = uploaded_image is None
            else:
                disable_send = not user_input.strip()
            
            if st.button(send_button_text, type="primary", disabled=disable_send):
                # Handle Text/Audio Input
                if input_method in ["Text Input", "Voice Recording", "Audio Upload"] and user_input.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    with st.spinner("Processing..."):
                        final_response, _, _ = st.session_state.chatbot.get_complete_response(
                            user_input, max_length=max_length, input_type="text"
                        )
                    
                    # After final_response is generated
                        triples = extract_triples(final_response)
                        if triples:
                            cypher_statements = triples_to_cypher(triples, st.session_state.farmer_id)
                            update_neo4j_with_triples(profile_store, cypher_statements)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.rerun()

                # Handle Image Analysis
                elif input_method == "Image Analysis" and uploaded_image:
                    image = Image.open(uploaded_image)
                    
                    # Add a user message to the chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"Image submitted for {analysis_task.replace('_', ' ').title()}.",
                        "image": image,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    with st.spinner("Analyzing image and generating recommendations..."):
                        # [** MODIFIED **] Call the new orchestrator method
                        logger.info(f"Starting image analysis for task: {analysis_task}")
                        recommendations, label = st.session_state.chatbot.get_image_analysis_recommendations(
                            image=image, 
                            task=analysis_task,
                            max_length=3000 # You can adjust the response length
                        )

                    # Add the assistant's detailed response to history
                    # We can prepend the label to the recommendations for clarity
                    final_response_content = f"**Analysis Result:** {label}\n\n---\n\n{recommendations}"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_response_content,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.rerun()

        
        with col2:
            st.header("📊 Chat Stats")
            st.metric("Total Messages", len(st.session_state.chat_history))
            st.metric("User Messages", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
            st.metric("Bot Responses", len([m for m in st.session_state.chat_history if m["role"] == "assistant"]))
        
        # Chat history display
        st.header("💭 Chat History")
        
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role = message.get("role", "user")
                with st.chat_message(role):
                    st.write(f"**{'You' if role == 'user' else 'Assistant'}** ({message['timestamp']})")
                    st.write(message["content"])
                    # Display the image if it exists in the user's message
                    if "image" in message:
                        st.image(message["image"], width=250)
        else:
            st.info("No messages yet. Start a conversation!")
        
        # Footer
        st.divider()
        st.info("**SAHAYAK KRISHI** - Multilingual Agricultural Assistant")

if __name__ == "__main__":
    main()