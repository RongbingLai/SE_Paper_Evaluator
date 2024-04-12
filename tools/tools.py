import re
import os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.tools import tool


@tool
def fetch_all_section_titles(path: str) -> list:
    """
    Fetches all section titles in a specified research paper.
    
    This function searches for the given path, extracts pages in the pdf file
    parses its content, and extracts the titles of all sections. This can include 
    standard sections such as 'Introduction', 'Literature Review', 'Methodology', etc.
    
    Parameters:
        path (str): The path of the research paper from which to fetch the section titles.
        
    Returns:
        list: A list of section titles found in the specified research paper.
    
    Example:
        If the research paper has sections 'Introduction', 'Literature Review', 'Methodology',
        the function will return:
        ['Introduction', 'Literature Review', 'Methodology']
    """
    # This is only suitable for the current used paper
    title_pattern = re.compile(r'^[IVXLCDM]+\.\s+[A-Z\s]+$', re.MULTILINE)
    section_titles = []

    for page_layout in extract_pages(path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_text = text_line.get_text().strip()
                    if title_pattern.match(line_text):
                        section_titles.append(line_text)

    return section_titles

@tool
def get_relevant_section(question: str):
    # make it longer, explain a tool to someone, give examples
    """
    Retrieve the most relevant section from the indexed
    papers that evaluates research paper based on a given question.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_index_criteria", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever
    )

    res = qa.invoke(question)
    return res


# @tool
# def get_relevant_section():
#     """
#     Retrieve the most relevant section from the indexed
#     papers that evaluates research paper based on a given question.
#     """
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.load_local("faiss_index_criteria", embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever()

#     return create_retriever_tool(retriever,
#                                  "get_relevant_section",
#                                  "Retrieve the most relevant section from the indexed \
#                                   papers that evaluates research paper based on a given question.")
