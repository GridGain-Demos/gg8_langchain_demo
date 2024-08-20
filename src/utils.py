from langchain_gridgain.document_loaders import GridGainDocumentLoader
from langchain_gridgain.storage import GridGainStore
from langchain_gridgain.chat_message_histories import GridGainChatMessageHistory
from langchain_gridgain.llm_cache import GridGainCache
from langchain_gridgain.vectorstores import GridGainVectorStore

from langchain_gridgain.llm_cache import GridGainSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from pygridgain import Client


import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_embeddings_model(api_key)-> OpenAIEmbeddings:
    try:
        # Initialize embeddings model
        os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize Embedding Model: {e}")
        raise

def initialize_opneai_llm(api_key)-> OpenAI:
    try:
        # Initialize Gemini
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI()
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize GridGainDocumentLoader: {e}")
        raise

def initialize_doc_loader(client)-> GridGainDocumentLoader:
    try:
        doc_loader = GridGainDocumentLoader(
            cache_name="review_cache",
            client=client,
            create_cache_if_not_exists=True
        )
        logger.info("GridGainDocumentLoader initialized successfully.")
        return doc_loader
    except Exception as e:
        logger.error(f"Failed to initialize GridGainDocumentLoader: {e}")
        raise

def initialize_keyvalue_store(client)-> GridGainStore:
    try:
        key_value_store = GridGainStore(
            cache_name="laptop_specs",
            client=client
        )
        logger.info("GridGainStore initialized successfully.")
        return key_value_store
    except Exception as e:
        logger.error(f"Failed to initialize GridGainStore: {e}")
        raise

def initialize_chathistory_store(client)-> GridGainChatMessageHistory:
    try:
        chat_history = GridGainChatMessageHistory(
            session_id="user_session",
            cache_name="chat_history",
            client=client
        )
        logger.info("GridGainChatMessageHistory initialized successfully.")
        return chat_history
    except Exception as e:
        logger.error(f"Failed to initialize GridGainChatMessageHistory: {e}")
        raise

def initialize_llm_cache(client)-> GridGainCache:
    try:
        llm_cache = GridGainCache(
            cache_name="llm_cache",
            client=client
        )
        logger.info("GridGainCache initialized successfully.")
        return llm_cache
    except Exception as e:
        logger.error(f"Failed to initialize GridGainCache: {e}")
        raise

def initialize_semantic_llm_cache(client, embedding)-> GridGainSemanticCache:
    try:
        llm_cache = GridGainCache(
            cache_name="llm_cache",
            client=client
        )
        semantic_cache = GridGainSemanticCache(
            llm_cache=llm_cache,
            cache_name="semantic_llm_cache",
            client=client,
            embedding=embedding,
            similarity_threshold=0.85
        )
        logger.info("GridGainSemanticCache initialized successfully.")
        return semantic_cache
    except Exception as e:
        logger.error(f"Failed to initialize GridGainSemanticCache: {e}")
        raise

def initialize_vector_store(client, embedding_model)-> GridGainVectorStore:
    try:
        vector_store = GridGainVectorStore(
            cache_name="vector_cache",
            embedding=embedding_model,
            client=client
        )
        logger.info("GridGainVectorStore initialized successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize GridGainVectorStore: {e}")
        raise

def connect_to_gridgain(host: str, port: int)-> Client:
    try:
        client = Client()
        client.connect(host, port)
        logger.info("Connected to ignite successfully.")
        return client
    except Exception as e:
        logger.exception(f"Failed to connect to Ignite: {e}")
        raise

def populate_caches_rs(reviews, specs, doc_loader,key_value_store):
    """
    Populate caches with sample data for demonstration purposes.
    """
    try:
        # Populate review cache
        doc_loader.populate_cache(reviews)
        
        # Populate specs cache
        key_value_store.mset([(k, v) for k, v in specs.items()])

        # Verify cache contents
        logger.info("Verifying cache contents:")
        for key in reviews.keys():
            value = doc_loader.get(key)
            logger.info(f"Review cache entry for {key}: {value}")
        for key in specs.keys():
            value = key_value_store.mget([key])[0]
            logger.info(f"Specs cache entry for {key}: {value}")
        
        logger.info("Caches populated and verified with sample data.")
    except Exception as e:
        logger.error(f"Error populating caches: {e}", exc_info=True)
        raise  # Re-raise the exception to ensure it's not silently ignored

def populate_caches(doc_loader,key_value_store):
    """
    reviews = {
            "laptop1": "Great performance for coding and video editing. The 16GB RAM and dedicated GPU make multitasking a breeze."
        }
    specs = {
            "laptop1": "16GB RAM, NVIDIA RTX 3060, Intel i7 11th Gen"
        }
    """
    reviews = {
            "laptop1": "Great performance for coding and video editing. The 16GB RAM and dedicated GPU make multitasking a breeze.",
            "laptop2": "Excellent battery life, perfect for students. Lightweight and portable, but the processor is a bit slow for heavy tasks.",
            "laptop3": "High-resolution display, ideal for graphic design. Comes with a stylus, but the price is on the higher side.",
            "laptop4": "Budget-friendly option with decent specs. Good for everyday tasks, but struggles with gaming.",
        }
    specs = {
            "laptop1": "16GB RAM, NVIDIA RTX 3060, Intel i7 11th Gen",
            "laptop2": "8GB RAM, Intel Iris Xe Graphics, Intel i5 11th Gen",
            "laptop3": "32GB RAM, NVIDIA RTX 3080, AMD Ryzen 9",
            "laptop4": "8GB RAM, Intel UHD Graphics, Intel i3 10th Gen",
        }
    
    #reviews = load_dict_from_file("data/reviews.txt")
    #specs = load_dict_from_file("data/specs.txt")
    
    populate_caches_rs(reviews,specs,doc_loader,key_value_store)


def load_dict_from_file(filename):
    """Load a dictionary from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')