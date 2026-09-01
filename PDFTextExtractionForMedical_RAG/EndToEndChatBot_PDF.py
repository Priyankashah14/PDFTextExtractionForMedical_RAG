import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

### Loading Environment Variables and Setting Up Embeddings ##########
from dotenv import load_dotenv
load_dotenv()

os.environ["Groq_API_KEY"] = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

### Set up streamlit app

st.title("Conversational Chatbot with PDF")
st.write("Upload PDF files and chat with their content.")

#Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

#Check if Groq API key is provided
if api_key:
    llm= ChatGroq(groq_api_key=api_key, model_name="groq/compound-mini")

    ### Chat interface
    session_id = st.text_input("Session ID", value="default_session") 

    ### Manage chat history statefully
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    ### Process uploaded PDFs
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        context_system_prompt =(
            
            "You are a helpful assistant that answers questions based on the content of the uploaded PDF documents. "
            "Use the retrieved information to provide accurate and concise responses. "
            "If the answer is not found in the documents, respond with 'I don't know.'"
        )

        context_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    ### Answer questions based on the uploaded PDFs

        system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
        qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id]=ChatMessageHistory()
                    return st.session_state.store[session_id]
                
        conversational_rag_chain=RunnableWithMessageHistory(
                    rag_chain,get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
        )
        
        user_input = st.text_input("Your question:")
        if user_input:
                    session_history=get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={
                            "configurable": {"session_id":session_id}
                        },  # constructs a key "abc123" in `store`.
                    )
                    st.write(st.session_state.store)
                    st.write("Assistant:", response['answer'])
                    st.write("Chat History:", session_history.messages)
        else:
            st.warning("Please enter the GRoq API Key")