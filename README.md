# LangChain with Ollama (Gemma) Tutorial

This project demonstrates the incremental learning of LangChain concepts using Ollama's Gemma model. Each example builds upon the previous ones to show different aspects of LangChain's capabilities.

## Project Structure

```
.
├── README.md
├── src/
    ├── 01_basic_chat.py    # Basic LLM interaction
    └── 02_chain_memory.py  # Chains with conversation memory
```

## Examples Explained

### 1. Basic Chat (`01_basic_chat.py`)

This example demonstrates the fundamental interaction with an LLM using LangChain. Key concepts:
- Initializing an LLM (Ollama with Gemma3)
- Basic prompt handling
- Streaming output with callbacks

To run:
```bash
python src/01_basic_chat.py
```

### 2. Chains with Memory (`02_chain_memory.py`)

This example introduces more advanced LangChain concepts:
- LLMChain for structured conversation flow
- PromptTemplates for consistent message formatting
- ConversationBufferMemory for maintaining chat history
- Stateful conversation with context retention

To run:
```bash
python src/02_chain_memory.py
```

## Requirements

- Python 3.x
- Ollama with Gemma3 model installed
- Required packages:
  - langchain
  - langchain-community
  - langchain-core
  - python-dotenv

## Key Concepts

1. **LLMs (Large Language Models)**
   - Direct interface to language models
   - Streaming capability for real-time responses

2. **Chains**
   - Combine multiple components into a single workflow
   - Sequential processing of prompts and responses

3. **Memory**
   - Store and retrieve conversation history
   - Maintain context across multiple interactions

4. **PromptTemplates**
   - Standardize input format
   - Dynamic prompt generation with variables

## Next Steps

Future examples will cover:
1. Different types of memory implementations
2. Tools and agents for task automation
3. Complex chain architectures
4. Structured output formatting
5. RAG (Retrieval Augmented Generation)
