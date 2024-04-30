from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from papermage.recipes import CoreRecipe
import json

section_title_list = []

@tool
def fetch_all_section_titles(path: str) -> list:
    """
    This function searches for the given path and extracts the titles of all sections. 

    Parameters:
        path (str): The path of the research paper from which to fetch the section titles.

    Returns:
        list: A list of section titles found in the specified research paper.

    Example:
        If the research paper has sections 'Introduction', 'Literature Review', 'Methodology',
        the function will return:
        ['Introduction', 'Literature Review', 'Methodology']
    """
    recipe = CoreRecipe()
    doc = recipe.run(path)
    for section in doc.sections:
        section_title_list.append(section.text)

    return section_title_list

@tool
def fetch_section_content_by_titles(params) -> str:
    """
    The function fetechs the content of a research paper section for the given section
    title from the given local path. 

    Parameters:
        params (str): a string dictionary that contains the actual parameters. The first key is "section_title" with 
        the title of the research paper section as the value. The second key is "path" with the path of the 
        research paper in the local computer as the value

    Returns:
        str: The content of the research paper section.

    Examples:
        If the given section title is I.Introduction, then the function will return the content
        of the Introduction section from the given path:
        "Good old documentation, the ideal companion of any software system, is intended to provide
        stakeholders with useful knowledge about the system and related processes...address them."
    """
    content_started = False
    content = ""
    next_index = 0
    section_title = json.loads(params)['section_title']
    path = json.loads(params)['path']

    for page_layout in extract_pages(path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_text = text_line.get_text().strip()
                    # records the start of a section
                    if line_text.replace(" ", "") == section_title.replace(" ", ""):
                        content_started = True
                        next_index = section_title_list.index(section_title) + 1
                        continue

                    # returns the content if reaches the next section title
                    if content_started and (line_text.replace(" ", "") == section_title_list[next_index].replace(" ","")):
                        return content

                    if content_started:
                        content += line_text + "\n"
    return content

@tool
def generate_review(section_content: str, question: str) -> str:
    """
    The function returns a response that answers the question regarding the section
    content by searching in the local faiss database.

    Parameters:
        question (str): The question given in the prompt context.
        section_content (str): The section content from the manuscript.

    Returns:
        str: The response answering the given question that evaluates the section_content.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "/Users/crystalalice/Desktop/ICSHP_Research/SE_Paper_Evaluator/faiss_index_criteria", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()
    llm = OpenAI()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    try:
        definition = qa.invoke("What kind of result is considered to be specific and concrete? What kind of results are not precise?")
        print(definition)
    except Exception as e:
        print(f"An error occurred: {e}")

    temp = '''
    Given the section below, answer the question in detail with reference sentences from the section content.

    Section Content: 
    {section_content}
    END OF SECTION

    Assessment Criteria:
    {criterion_definition}

    Question:
    {question}

    In your final answer, you need to include two parts using this format:
        ```
        Manuscript Text: [the sentence(s) that you want to review in the section content]
        Review: [Your evaluation]
        ... (you should have enough Manuscript Text/Review to cover the review for the whole section content)
        ```
    '''

    # question = (
    #     f"Given the section below, answer the question.\n"
    #     "Section Content:\n{section_content}\nEND OF SECTION\n\n"
    #     "Assessment Criteria:\n{criterion_definition}"
    #     "Question: {question}"
    # )

    prompt = PromptTemplate(template=temp, input_variables=["section_content","criterion_definition","question"])
    chain = prompt | llm
    res = chain.invoke({
        "section_content": section_content,
        "criterion_definition": definition,
        "question": question,
    })
    return res

# @tool
# def get_review_from_similar_paper():
#     pass