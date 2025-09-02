"""
Advanced LangChain Example with Chains and Memory

This script demonstrates more advanced LangChain concepts including:
1. RunnableSequence - Modern way to combine multiple components
2. Memory - Maintaining conversation history
3. PromptTemplates - Structured prompt generation
4. Stateful Conversations - Context-aware interactions

The example uses the Gemma 3B model through Ollama and shows how to create
a conversational agent that remembers previous interactions.
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def create_llm():
    """
    Creates and configures the Language Model instance.
    
    Returns:
        OllamaLLM: Configured Ollama LLM instance with Gemma3 model and streaming output
    """
    return OllamaLLM(
        model="gemma3",
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=True,
    )

def create_chat_chain():
    """
    Creates a conversational chain with memory.
    
    This function sets up:
    1. A ChatPromptTemplate for consistent message formatting
    2. ChatMessageHistory for storing conversation
    3. RunnableSequence to combine components
    
    Returns:
        tuple: (RunnableSequence, ChatMessageHistory) - The chain and history objects
    """
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant. Your responses should be friendly and concise."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create the LLM
    llm = create_llm()
    
    # Create the chain using the new RunnableSequence pattern
    chain = prompt | llm
    
    return chain

def main():
    """
    Main function that demonstrates stateful conversation using LangChain.
    
    This function:
    1. Creates a conversation chain with memory
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
    
    chat_history = []
    for question in questions:
        print(f"\nHuman: {question}")
        
        # Invoke the chain with the current context
        response = chain.invoke({
            "chat_history": chat_history,
            "input": question
        })
        
        # Update the chat history
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=str(response))
        ])
        
        print(f"\nAssistant: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
