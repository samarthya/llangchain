"""
Basic LangChain Example with Ollama

This script demonstrates the fundamental usage of LangChain with Ollama's Gemma model.
It shows how to:
1. Initialize an LLM (Language Model)
2. Send a basic prompt
3. Handle streaming output

The example uses the Gemma 3B model through Ollama and implements real-time
streaming of the model's output using callback handlers.
"""

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    """
    Main function that demonstrates basic LLM interaction.
    
    This function:
    1. Initializes the Ollama LLM with the Gemma3 model
    2. Sets up streaming output
    3. Sends a test prompt
    4. Displays the response
    """
    # Initialize Ollama with the Gemma 3B model
    llm = Ollama(
        model="gemma3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )
    
    # Simple prompt to test the model
    prompt = "Explain what is LangChain in 3 sentences."
    
    print("\nAsking:", prompt)
    print("\nResponse:")
    response = llm.invoke(prompt)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
