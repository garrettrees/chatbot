import os
import sys
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import SlackDirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader


from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI


input = sys.argv[1]

#loader = TextLoader('data.txt')
loader = DirectoryLoader("./data/", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(input))