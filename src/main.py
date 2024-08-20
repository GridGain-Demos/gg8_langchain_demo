"""
Comprehensive example of a laptop recommendation system using LangChain components and GridGain/Ignite for data storage.
"""
import logging
import utils as utils
from typing import Dict, List, Any
from pygridgain import Client

from langchain.memory import ConversationBufferMemory
from custom_retriever import CustomRetriever

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import argparse
import data_loader
import retriever_instantiator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize(use_history: bool, use_semantic_llm_cache: bool, api_key: str, load_data: bool):
    """
    Initialize the laptop recommendation system with specified configuration.

    This function sets up global variables and initializes various components
    of the system, including data stores, document loader, LLM cache, vector store,
    memory, and the conversation chain.

    Args:
        use_history (bool): If True, enables chat history functionality.
                            If False, initializes and uses LLM cache instead.
        use_semantic_llm_cache (bool): If True, uses semantic LLM cache.
        api_key (str): The API key for OpenAI.
        load_data (bool): If True, loads data using data_loader. If False, instantiates retriever without loading data.

    Global Variables:
        chat_history: Store for chat history
        chain: Conversation chain for processing queries
        memory: ConversationBufferMemory for maintaining conversation context

    Returns:
        None

    Note:
        This function modifies global variables and should be called once at the start
        of the application to set up the environment.
    """
    global chat_history, chain, memory
    
    # Initialize llm
    llm=utils.initialize_opneai_llm(api_key)

    # Initialize embeddings model
    embeddings = utils.initialize_embeddings_model(api_key)

    # Connect to Ignite
    client= utils.connect_to_gridgain("localhost", 10800)
    
    # Configure LLM cache if chat history is not used
    if not use_history:
        if use_semantic_llm_cache:
            llm_cache = utils.initialize_semantic_llm_cache(client,embeddings)
        else:
            llm_cache = utils.initialize_llm_cache(client)
        llm.cache = llm_cache
    else:
        chat_history = utils.initialize_chathistory_store(client)
         # Set up conversation memory
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            output_key="output"
        )
    
    # Initialize custom_retriever based on load_data flag
    if load_data:
        logger.info("load_data is true, will be loading the data in the databases")
        custom_retriever: CustomRetriever | None = data_loader.populate(client, api_key)
    else:
        logger.info("load_data is false, will not be loading the data in the databases")
        custom_retriever:CustomRetriever = retriever_instantiator.instantiate_retriever(client, api_key)


    # Define conversation template
    conversation_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("human", "You are a helpful AI assistant specializing in laptop recommendations. Use the provided laptop information to assist the user."),
        MessagesPlaceholder(variable_name="history"),
        ("human", """
    Relevant Laptop Information:
    {context}

    User Query: {input}
    """),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, conversation_template)
    chain = create_retrieval_chain(custom_retriever, question_answer_chain)


def process_user_input(user_input: str, use_history: bool) -> str:
    """
    Process user input and generate a response using the retrieval chain.

    Args:
        user_input (str): The user's input query.
        use_history (bool): whether to use history in the conversation

    Returns:
        str: The generated response.
    """
    try:
        logger.info(f"Processing user input: {user_input}")

        if use_history:
            response = chain.invoke({
                "input": user_input,
                "history": chat_history.messages
            })
        else:
            response = chain.invoke({
                "input": user_input,
                "history": []  # Empty history when not using chat history
            })
        
        # Extract the answer from the response
        if isinstance(response, dict) and 'answer' in response:
            answer = response['answer']
        else:
            answer = str(response)  # Fallback to string representation if not in expected format
        
        if use_history:
            # Update memory with the new interaction
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(answer)
        
        return answer
    except Exception as e:
        logger.exception(f"Error in process_user_input: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again."


def main():
    """
    Main function to run the Laptop Recommendation System.

    This function sets up the argument parser, initializes the system,
    populates data stores, and runs the main conversation loop.
    """

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Laptop Recommendation System")
    parser.add_argument("--use_history", default="false", help="Use chat history")
    parser.add_argument("--use_semantic_llm_cache", default="false", help="Use Semantic LLM Cache")
    parser.add_argument("--load_data", default="false", help="Load data or use pre-loaded data")
    parser.add_argument("--use_api_key", help="The API key to be used")
    args = parser.parse_args()

    # Convert string arguments to boolean values
    use_history = str2bool(args.use_history)
    use_semantic_llm_cache = str2bool(args.use_semantic_llm_cache)
    load_data = str2bool(args.load_data)
    api_key = args.use_api_key

    if api_key is None:
        api_key = input("\nPlease provide your Open AI api key: ")

    if use_semantic_llm_cache:
        use_history = False

    print("Welcome to the Laptop Recommendation System!")
    print("Populating vector store with data...")

    try:
        # Initialize the system with specified configuration
        initialize(use_history, use_semantic_llm_cache, api_key, load_data)

        print("You can ask questions about laptops or request recommendations.")
        print("Type 'exit' to end the conversation.")

        # Main conversation loop
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Thank you for using the Laptop Recommendation System. Goodbye!")
                break
            # Process user input and generate response
            print("\n")
            response = process_user_input(user_input, use_history)
            print(f"\nBot: {response}\n")

    except Exception as e:
        # Log any errors that occur during execution
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        print("An error occurred. Please check the logs for more information.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    main()