#Import Python modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
import streamlit as st

def fn_process_config():
    #Load the models
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="your_google_api_key")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key="your_google_api_key")

    #Load the PDF and create chunks
    loader = PyPDFLoader("Introduction to Artificial Intelligence.pdf")
    txt_splitter = CharacterTextSplitter(separator=".",chunk_size=200,chunk_overlap=20,length_function=len,is_separator_regex=False,)
    splitted_pages = loader.load_and_split(txt_splitter)

    #chunks into embeddings and store it in Chroma-db
    vectordb=Chroma.from_documents(splitted_pages,embedding_model)

    #Configure Chroma as a retriever with top_k=N
    retriever = vectordb.as_retriever(search_type='similarity',search_kwargs={"k": 3})

    #Create the retrieval chain
    template = """
    You are an expert AI.
    you will Answer based on the context provided. 
    context: {context}
    input: {input}
    answer:
    """
    prompt = PromptTemplate.from_template(template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

def fn_process_model(pass_prompt,retrieval_chain):
    #Invoke the retrieval chain
    
    response=retrieval_chain.invoke({"input":pass_prompt})

    #Print the answer to the question
    print(response["answer"])
    return response["answer"]