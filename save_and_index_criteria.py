import os
import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

CRITERIA_PAPERS_DIRECTORY = (
    "/Users/crystalalice/Desktop/ICSHP_Research/criteria_papers_for_database"
)

def save_and_index_papers():
    embeddings = OpenAIEmbeddings()
    all_docs = []

    for filename in os.listdir(CRITERIA_PAPERS_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(CRITERIA_PAPERS_DIRECTORY, filename)
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load()
            text_splitter = SemanticChunker(
                OpenAIEmbeddings(show_progress_bar=True)
            )
            docs = text_splitter.split_documents(documents=documents)
            for doc in docs:
                all_docs.append(doc)
                print("added doc")
                time.sleep(2)

    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local("faiss_index_criteria")

    print("Indexed and saved criteria papers.")


if __name__ == "__main__":
    save_and_index_papers()
