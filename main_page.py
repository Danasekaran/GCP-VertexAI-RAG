import streamlit as st
import process_model as process_model

st.set_page_config(
    page_title="RAG Demo",
    page_icon="📝",
)

@st.cache_resource
def initialization_function():
    retrieval_chain=process_model.fn_process_config()
    return retrieval_chain

st.write("# Google Generative-AI - Smart Bot")

st.sidebar.write("## Choose your source here!")
# st.sidebar.checkbox("Introduction to Artificial Intelligence",True)

selected_source=st.sidebar.radio("",["Introduction to Artificial Intelligence","Learn Vertex AI"])
# st.write(selected_source)


prompt =st.text_input("Enter your prompt")
# st.write(prompt)

retrieval_chain=initialization_function()

if prompt:
    resp=process_model.fn_process_model(prompt,retrieval_chain)
    st.markdown(resp)
