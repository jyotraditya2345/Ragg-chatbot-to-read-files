from langchain_community.document_loaders import PyPDFLoader

import gradio as gr

import os
import glob
import tiktoken
import pandas as pd
from datetime import datetime
# import openai # Remove OpenAI import
# from openai import AzureOpenAI # Remove OpenAI import
from utils import *
from chromadb_utils import *

## Chromadb configuration
import chromadb
chroma_client = chromadb.Client()
collection_name = 'story_collection'



# Add Gemini AI configuration
gemini_api_key = ""

already_stored_cdb = False
collection = None

def init_vector_db_condfig():    
    global already_stored_cdb, collection
    #if not already_stored_cdb:
    # PDF data location/path
    root_directory = 'data'
    chunk_size = 1024
    overlap = 80
    doc_content_df = pd.DataFrame()
    
    # Listing all the pdf files listed in root_directory
    files = get_pdf_files(root_directory)
    
    # Extract the files content and persisting in DF
    for file in files:
        doc_content = extract_pdf_content(file)    
        entry  = {'doc_name': file, 'doc_content': doc_content}
        doc_content_df = pd.concat([doc_content_df, pd.DataFrame([entry])], ignore_index=True)
    
    # creating the chunks of files content of configured size
    encoding = tiktoken.get_encoding("cl100k_base")
    doc_content_df = doc_content_df.apply(lambda row: create_chunks(row, encoding, chunk_size, overlap), axis=1)
    
    # Creating the chromadb collection. if it already created, will return collection instance with exception - collection already present.
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
    except Exception as ex:
        print(f'Collection creation failed with - {ex}')
    
    # Upsert the chunks in chroma db with respective ID's
    doc_content_df = doc_content_df.apply(lambda row: upsert_to_chromadb(row, collection), axis=1)
    
    timestamp = datetime.now().timestamp()
    
    doc_content_df.to_csv(f'data/{collection_name}-{timestamp}.csv')

system_prompt = "You are helpful AI assistant."

import requests
import json
 
def generate_ai_response(context, sys_msg, criteria):
 
    headers = {
        'Content-Type': 'application/json',
    }
 
    params = {
        'key': f'{gemini_api_key}',
    }
 
    json_data = {
        'contents': [
            {
                'parts': [
                    {
                        'text': f'Context : {context}, System role: {sys_msg}  - Question {criteria}',
                    },
                ],
            },
        ],
    }
 
    response = requests.post(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent',
        params=params,
        headers=headers,
        json=json_data,
    )
 
 
 
    byte_data = response.content
    json_str = byte_data.decode('utf-8')
    json_data = json.loads(json_str)
    response_text = json_data['candidates'][0].get('content').get('parts')[0].get('text')
    print(response_text)
    return response_text
def get_matching_chunk(query):
    
    collection = chroma_client.get_collection(name=collection_name)
    
    results = collection.query(query_texts=[f"{query}"], n_results=1)
    ref_document_name = results.get('ids')[0][0].split('/')[-1]
    chunk_text = results.get('documents')[0][0]
    
    return ref_document_name, chunk_text
    
with gr.Blocks() as demo:
    gr.Markdown("Knowledge Base Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        ref_document_name, chunk_text = get_matching_chunk(message)
        ai_response = generate_ai_response(chunk_text, system_prompt, message)
        chat_history.append((message, ai_response + f' [Reference Document - {ref_document_name}]'))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    init_vector_db_condfig()
    demo.launch(server_name="0.0.0.0", server_port=4065, debug=False, auth=('Quantum123', 'pass123'), share=True)
