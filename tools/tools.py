import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

CRITERIA_PAPERS_DIRECTORY = "path"

def load_and_index_papers():
    """
    Load all PDF papers from the specified directory, split them into chunks,
    embed these chunks, and index them using FAISS for efficient retrieval.
    """
    embeddings = OpenAIEmbeddings()
    all_docs = []

    for filename in os.listdir(CRITERIA_PAPERS_DIRECTORY):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(CRITERIA_PAPERS_DIRECTORY, filename)
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
            docs = text_splitter.split_documents(documents=documents)
            all_docs.extend(docs)

    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local("faiss_index_criteria")

    print("Indexed and saved criteria papers.")

def get_relevant_section(question: str):
    """
    Retrieve the most relevant section from the indexed criteria papers
    based on a given question.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index_criteria", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA(llm=OpenAI(), retriever=retriever)

    res = qa.run(question)
    return res