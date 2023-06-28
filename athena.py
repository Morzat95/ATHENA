import os
import sys

import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False
dataDirectory = "data/"
persistDirectory = "persist"
returnSourceDocuments = False

query = sys.argv[1]

if PERSIST and os.path.exists(persistDirectory):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory=persistDirectory, embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = DirectoryLoader(dataDirectory)
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":persistDirectory}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = RetrievalQA.from_chain_type(
  llm = ChatOpenAI(model="gpt-3.5-turbo"),
  retriever = index.vectorstore.as_retriever(search_kwargs={"k": 1}),
  return_source_documents=returnSourceDocuments  # Set this flag to True to include sources in the response
)

response = chain._call({'query': query})

print(response['result'])

if (returnSourceDocuments):
  print()
  print(response['source_documents'])
