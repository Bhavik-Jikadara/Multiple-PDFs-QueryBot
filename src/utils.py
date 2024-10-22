import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import json
import docx
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

def get_docx_text(docx_files):
    text = ""
    try:
        for docx_file in docx_files:
            doc = docx.Document(docx_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
    return text

def get_json_text(json_files):
    text = ""
    try:
        for json_file in json_files:
            data = json.load(json_file)
            text += json.dumps(data, indent=4)
    except Exception as e:
        st.error(f"Error reading JSON: {str(e)}")
    return text

def get_text_from_files(files):
    if not files:
        st.error("No files uploaded. Please upload at least one file.")
        return ""
        
    text = ""
    pdf_files = []
    docx_files = []
    json_files = []

    for file in files:
        if file.name.endswith('.pdf'):
            pdf_files.append(file)
        elif file.name.endswith('.docx'):
            docx_files.append(file)
        elif file.name.endswith('.json'):
            json_files.append(file)

    if pdf_files:
        text += get_pdf_text(pdf_files)
    if docx_files:
        text += get_docx_text(docx_files)
    if json_files:
        text += get_json_text(json_files)
    
    if not text.strip():
        st.error("No text could be extracted from the uploaded files.")
        return ""
        
    return text

def get_text_chunks(raw_text):
    if not raw_text:
        return []
        
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Increased chunk size
        chunk_overlap=100,  # Increased overlap
        length_function=len
    )   
    chunks = text_splitter.split_text(raw_text)
    
    if not chunks:
        st.error("Could not create text chunks. The text might be too short or empty.")
        return []
        
    return chunks

def get_vectorstore(text_chunks, api_key):
    if not text_chunks:
        st.error("No text chunks to process. Please ensure your documents contain text.")
        return None
        
    if not api_key:
        st.error("OpenAI API key is required.")
        return None
        
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, api_key):
    if not vectorstore:
        st.error("Vector store is not initialized.")
        return None
        
    if not api_key:
        st.error("OpenAI API key is required.")
        return None
        
    try:
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

# Custom template remains the same
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)