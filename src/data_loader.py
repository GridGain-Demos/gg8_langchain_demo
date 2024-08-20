import json
import logging
import argparse
from utils import (
    initialize_keyvalue_store,
    initialize_doc_loader,
    initialize_vector_store,
    initialize_embeddings_model,
)
from custom_retriever import CustomRetriever
from langchain.schema import Document
from pygridgain import Client


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def populate_key_value_store(store, data):
    store.mset([(k, json.dumps(v)) for k, v in data.items()])
    logger.info(f"Populated key-value store with {len(data)} entries.")

def populate_doc_loader(loader, data):
    loader.populate_cache(data)
    logger.info(f"Populated document loader with {len(data)} entries.")

def populate(client:Client, api_key: str):
    try:
        # Initialize stores and models
        specs_store = initialize_keyvalue_store(client)
        doc_loader = initialize_doc_loader(client)
        embeddings = initialize_embeddings_model(api_key)
        vector_store = initialize_vector_store(client, embeddings)

        # Load data from JSON files
        specs_data = load_json_data("data/laptop_specs.json")
        reviews_data = load_json_data("data/laptop_reviews.json")

        # Populate stores
        populate_key_value_store(specs_store, specs_data)
        populate_doc_loader(doc_loader, reviews_data)

        # Prepare documents for vector store population
        documents = [
            Document(page_content=review, metadata={"id": laptop_id})
            for laptop_id, review in reviews_data.items()
        ]

        # Populate vector store
        logger.info("GridGain vector store selected")

        # Initialize custom retriever
        custom_retriever = CustomRetriever(
            vector_store=vector_store,
            key_value_store=specs_store,
        )
        custom_retriever.populate_vector_store(documents)
        logger.info("Vector store populated successfully.")
        return custom_retriever

    except Exception as e:
        logger.error(f"An error occurred while loading and indexing data: {e}", exc_info=True)

def main(api_key: str):
    populate(api_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant Data Loader")
    parser.add_argument("--use_api_key", help="The API key to be used")
    args = parser.parse_args()
    api_key = args.use_api_key or input("\nPlease provide your OpenAI API key: ")
    main(api_key)