from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
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
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

    res = qa.invoke(question)
    return res