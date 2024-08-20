import logging
from pygridgain import Client
from utils import (
    initialize_keyvalue_store,
    initialize_doc_loader,
    initialize_vector_store,
    initialize_embeddings_model,
)
from custom_retriever import CustomRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def instantiate_retriever(client:Client, api_key: str) -> CustomRetriever:
    try:
        # Initialize stores and models
        key_value_store = initialize_keyvalue_store(client)
        doc_loader = initialize_doc_loader(client)
        embeddings = initialize_embeddings_model(api_key)
        
        vector_store = initialize_vector_store(client, embeddings)
        custom_retriever = CustomRetriever(
            vector_store=vector_store,
            key_value_store=key_value_store,
        )
        logger.info(f"custom_retriever is {custom_retriever}")
        return custom_retriever

    except Exception as e:
        logger.error(f"An error occurred while instantiating retriever: {e}", exc_info=True)
