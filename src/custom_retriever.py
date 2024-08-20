import logging
from typing import Dict, List, Any
from pydantic import Field
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomRetriever(BaseRetriever):
    """Custom retriever class that combines GridGain vector store search with key-value store lookup."""

    vector_store: Any = Field(default=None)
    key_value_store: Any = Field(default=None)
    embeddings: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def generate_timestamp_id(self):
        """Generate a unique ID based on the current timestamp in milliseconds."""
        return time.time_ns()  # Current time in milliseconds
    
    def populate_vector_store(self, reviews: List[Document]):
        """
        Populate GridGain vector store with combined data from reviews and specs.

        Args:
            reviews (List[Document]): List of review documents.
        """
        try:
            combined_texts = []
            metadatas = []

            for doc in reviews:
                #print(f"\doc is {doc}")
                print(f"\doc.metadata is {doc.metadata}")
                print(f"\doc.page_content is {doc.page_content}")


                laptop_id = doc.metadata.get("id")
                print(f"\ laptop_id is {laptop_id}")

                review_text = doc.page_content
                spec_text_str = self.key_value_store.mget([laptop_id])[0]  # Get the first (and only) item

                #logger.info(f"\spec_text_str is {spec_text_str}")

                # Convert spec_text from string to JSON
                specs = json.loads(spec_text_str)

                #logger.info(f"\spec_text is {specs}")

                # Combine the review and spec text exactly as in the CSV generator
                combined_text = f"Name: {specs['name']}, RAM: {specs['ram']}, GPU: {specs['gpu']}, CPU: {specs['cpu']}, Storage: {specs['storage']}. " \
                  f"Review: {review_text}"
                
                #logger.info(f"\ncombined_text is {combined_text}")
                
                combined_texts.append(combined_text)
                metadatas.append({"id": laptop_id})

            # Use the add_texts method to populate the GridGainVectorStore
            added_titles = self.vector_store.add_texts(combined_texts, metadatas=metadatas)

            logger.info(f"GridGain vector store populated with combined review and spec data for {added_titles}")
        except Exception as e:
            logger.error(f"Error populating data: {e}", exc_info=True)
            raise  

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): The user's query.

        Returns:
            List[Document]: A list of relevant documents.
        """
        try:
            if self.vector_store is None:
                return [Document(page_content="No laptop reviews available yet.", metadata={"source": "empty"})]
            
            # Get relevant documents from VectorStore
            relevant_docs = self.vector_store.similarity_search(query, k=12, score_threshold=0.6)
            validated_docs = []
            for item in relevant_docs:
                if isinstance(item, Document):
                    if item.page_content and isinstance(item.metadata, dict) and 'id' in item.metadata:
                        validated_docs.append(item)
                    else:
                        logger.warning(f"Invalid Document structure: {item}")
                elif isinstance(item, dict):
                    # Create a new Document if the item is a dictionary
                    content = item.get('content', '')
                    id = item.get('id', 'unknown')
                    if content:
                        new_doc = Document(page_content=content, metadata={'id': id})
                        validated_docs.append(new_doc)
                    else:
                        logger.warning(f"Invalid item structure, missing content: {item}")
                else:
                    logger.warning(f"Unexpected item type in relevant_docs: {type(item)}")

            return validated_docs

        except Exception as e:
            logger.exception(f"Error in get_relevant_documents: {e}")
            return [Document(page_content="Error retrieving relevant documents.", metadata={"source": "error"})]

    def get_laptop_specs(self, doc_ids: List[str]) -> Dict[str, str]:
        """
        Retrieve laptop specifications from the key-value store for specific document IDs.

        Args:
            doc_ids (List[str]): List of document IDs to fetch specs for.

        Returns:
            Dict[str, str]: A dictionary mapping laptop IDs to their specifications.
        """
        try:
            specs = self.key_value_store.mget(doc_ids)
            return {k: v for k, v in zip(doc_ids, specs) if v is not None}
        except Exception as e:
            logger.exception(f"Error in get_laptop_specs: {e}")
            return {}