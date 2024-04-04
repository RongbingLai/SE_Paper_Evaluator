from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.tools import tool

@tool
def get_relevant_section(question: str):
    """
    Retrieve the most relevant section from the indexed 
    papers that evaluates research paper based on a given question.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index_criteria", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA(llm=OpenAI(), retriever=retriever)

    res = qa.run(question)
    return res