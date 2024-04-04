import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

CRITERIA_PAPERS_DIRECTORY = "/Users/crystalalice/Desktop/ICSHP_Research/Criteria_papers"

def save_and_index_papers():
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

if __name__ == "__main__":
    save_and_index_papers()