"""
Advanced LangChain Example with Chains and Memory

This script demonstrates more advanced LangChain concepts including:
1. Chains - Combining multiple components into a single workflow
2. Memory - Maintaining conversation history
3. PromptTemplates - Structured prompt generation
4. Stateful Conversations - Context-aware interactions

The example uses the Gemma 3B model through Ollama and shows how to create
a conversational agent that remembers previous interactions.
"""

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def create_llm():
    """
    Creates and configures the Language Model instance.
    
    Returns:
        Ollama: Configured Ollama LLM instance with Gemma3 model and streaming output
    """
    return Ollama(
        model="gemma3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )

def create_chat_chain():
    """
    Creates a conversational chain with memory.
    
    This function sets up:
    1. A PromptTemplate for consistent message formatting
    2. ConversationBufferMemory to store chat history
    3. LLMChain to combine the LLM, prompt, and memory
    
    Returns:
        LLMChain: A configured chain ready for conversation
    """
    # Create a prompt template
    template = """
    You are a helpful AI assistant. Your responses should be friendly and concise.
    
    Previous conversation:
    {chat_history}
    
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the chain
    chain = LLMChain(
        llm=create_llm(),
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    return chain

def main():
    """
    Main function that demonstrates a stateful conversation using LangChain.
    
    This function:
    1. Creates a conversation chain
    2. Runs a series of related questions
    3. Demonstrates how the chain maintains context between questions
    4. Shows how the model can reference previous answers in the conversation
    """
    chain = create_chat_chain()
    
    # Let's have a simple conversation
    questions = [
        "What is the capital of France?",
        "What's the population of this city?",
        "Tell me an interesting historical fact about it."
    ]
    
    for question in questions:
        print(f"\nHuman: {question}")
        response = chain.run(human_input=question)
        print(f"\nAssistant: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
