import os
import constants
import gradio as gr

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

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Define the function to connect with Gradio UI
def query_langchain(input_text):
    loader = DirectoryLoader("./data", glob="*")
    index = VectorstoreIndexCreator().from_loaders([loader])
    result = index.query(input_text)
    return result

def clear_input_output():
    return "", ""

# Define the Gradio Blocks
with gr.Blocks() as app:
    with gr.Row():
        text_input = gr.Textbox(label="Question", elem_id="input-textbox", submit=True)
    with gr.Row():    
        submit_button = gr.Button("Submit")
    with gr.Row():
        text_output = gr.Textbox(label="Answer")
    with gr.Row():
        clear_button = gr.Button("Clear")

    submit_button.click(fn=query_langchain, inputs=text_input, outputs=text_output)
    clear_button.click(fn=clear_input_output, inputs=None, outputs=[text_input, text_output])

# Launch the Gradio app
app.launch()
