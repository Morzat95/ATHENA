# ATHENA

This script implements a question answering system over documents. It utilizes the OpenAI GPT-3.5 Turbo model for language understanding and retrieval-based question answering.

## Installation

Install [Langchain](https://github.com/hwchase17/langchain) and other required packages.

```
pip install langchain openai chromadb tiktoken unstructured python-dotenv
```

Set up your environment by creating a .env file in the project root directory to use your own [OpenAI API key](https://platform.openai.com/account/api-keys).

```
OPENAI_API_KEY = "..."
```

Prepare your data by placing the documents you want to perform question answering on in the `data/` directory.

## Example usage

Test reading `data/data.txt` file.

```
> python athena.py "what is my dog's name"
Your dog's name is Sunny.
```

Test reading `data/cat.pdf` file.

```
> python athena.py "what is my cat's name"
Your cat's name is Muffy.
```

### Based on https://github.com/techleadhd/chatgpt-retrieval
