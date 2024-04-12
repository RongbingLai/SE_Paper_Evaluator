import re
import os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.tools import tool


@tool
def fetch_all_section_titles(path: str) -> list:
    """    
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
def get_path_by_title(title: str) -> str:
    """
    Given the title, this function returns the local path of the research paper.

    Parameters: 
        title (str): The title of the research paper.
    
    Returns:
        str: The path of the research paper location on the local computer.
    """
    return "/Users/crystalalice/Desktop/ICSHP_Research/SE_paper/" + title + ".pdf"

@tool
def fetch_section_content_by_titles(section_title: str, path: str) -> str:
    """
    The function fetechs the content of a research paper section for the given section
    title from the given local path. This include standard section titles such as 
    Introduction, Methodology and etc.

    Parameters: 
        section_title (str): The title of the research paper section.
        path (str): The path of the research paper in the local computer.
    
    Returns: 
        str: The content of the research paper section.

    Examples:
        If the given section title is I.Introduction, then the function will return the content
        of the Introduction section from the given path:
        "Good old documentation, the ideal companion of any software system, is intended to provide 
        stakeholders with useful knowledge about the system and related processes...address them."
    """
    title_pattern = re.compile(r'^' + re.escape(section_title) + r'$', re.IGNORECASE)
    next_section_pattern = re.compile(r'^[IVXLCDM]+\.\s|\bREFERENCES\b|\ACKNOWLEDGMENT\b', re.IGNORECASE)
    
    content_started = False
    content = ""

    for page_layout in extract_pages(path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_text = text_line.get_text().strip()
                    # records the start of a section
                    if title_pattern.match(line_text):
                        content_started = True
                        continue
                    # returns the content if reaches the next section title
                    if content_started and next_section_pattern.match(line_text):
                        return content
                    
                    if content_started:
                        content += line_text    
    return content

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