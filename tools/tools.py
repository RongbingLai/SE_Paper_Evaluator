from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts.chat import ChatPromptTemplate
from papermage.recipes import CoreRecipe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

section_title_list = []


@tool
def fetch_all_section_titles(path: str) -> list:
    """
    This function searches for the given path and extracts the titles of all sections.

    Parameters:
        path (str): The path of the manuscript from which to fetch the section titles.

    Returns:
        list: A list of section titles found in the specified manuscript.

    Example:
        If the manuscript has sections 'Introduction', 'Literature Review', 'Methodology',
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
    The function fetechs the content of a manuscript section for the given section
    title from the given local path.

    Parameters:
        params (str): a string dictionary that contains the actual parameters. The first key is "section_title" with
        the title of the manuscript section as the value. The second key is "path" with the path of the
        manuscript in the local computer as the value

    Returns:
        str: The content of the manuscript section.

    Examples:
        If the given section title is I.Introduction, then the function will return the content
        of the Introduction section from the given path:
        "Good old documentation, the ideal companion of any software system, is intended to provide
        stakeholders with useful knowledge about the system and related processes...address them."
    """
    content_started = False
    content = ""
    next_index = 0
    section_title = json.loads(params)["section_title"]
    path = json.loads(params)["path"]

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
                    if content_started and (
                        line_text.replace(" ", "")
                        == section_title_list[next_index].replace(" ", "")
                    ):
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
        "/Users/crystalalice/Desktop/ICSHP_Research/SE_Paper_Evaluator/faiss_index_criteria",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    llm = OpenAI(temperature=0.0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"verbose": True},
    )

    try:
        definition = qa.invoke(
            "What kind of result is considered to be specific and concrete? What kind of results are not precise?"
        )["result"]
        print(definition)
    except Exception as e:
        print(f"An error occurred: {e}")
    chat = ChatOpenAI(temperature=0)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a review committee member at a established software engineering conference. You will be given a section of a manuscruipt along with a review question.\n"
                    "While reviewing the section, you need to consider the assessment criteria. You should review whole the section based on the review question and the assessment criteria.\n"
                    "Here is the assessment criteria:\n```\n{criterion_definition}\n```\n"
                    "Your review must be formatted as below for each part of the section on which you want to leave a comment:\n"
                    "```Manuscript Text: [the sentence(s) from the section that you want to leave comment on]\nReview: [Your comment]```\n"
                    "Please remember to leave comments on all parts of the section that are relevant to the review question."
                ),
            ),
            (
                "user",
                "Section:\n```\n{section_content}\n```\n\nReview Question: {question}\n\nYour comments:",
            ),
        ]
    )
    chain = chat_prompt | chat

    res = chain.invoke(
        {
            "section_content": section_content,
            "criterion_definition": definition,
            "question": question,
        }
    )
    return res

@tool
def get_openreview_reviews(path: str):
    """
    The function finds the most similar research paper to the 
    manuscript from OpenReview, which is a platform that contains the actual reviews
    for research papers, and gets the reviews for the paper as reference. 
    
    Parameters:
        path (str): The path of the research paper from which to fetch the section titles.

    Returns:
        str: The reviews for a similar research paper
    """
    paper_abstract = _get_paper_abstract()
    index = _find_similiar_paper(paper_abstract)

    with open('correct_file.json', 'r') as file:
        data = json.load(file)
        for paper in data['orb_submissions'][index]['article_versions'].values():
            print("title is " + paper['title'] + "\n")
            return paper['reviews'] if len(paper['review']) > 0 else "No Similar Review"


def _find_similiar_paper(paper_abstract: list) -> int:
    db_paper_abstracts = []

    with open('correct_file.json', 'r') as file:
        data = json.load(file)
        for submission in data['orb_submissions']:
            if '0' in submission['article_versions']:
                paper = submission['article_versions']['0'] # only need one version
                if paper['title'] and paper['abstract']:
                    db_paper_abstracts.append(paper['title'] + ": " + paper['abstract'])
                    print(paper['title'] + ": " + paper['abstract'] + "\n\n")
                else:
                    db_paper_abstracts.append('')
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_db = tfidf_vectorizer.fit_transform(db_paper_abstracts)
    tfidf_matrix = tfidf_vectorizer.transform(paper_abstract)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix_db).flatten()

    print("Cosine similarity scores:", cosine_sim)

    index = cosine_sim.argmax()
    print("index is " + str(index) + "\n")
    print(db_paper_abstracts[index])

    return index

def _get_paper_abstract(path) -> list:
    recipe = CoreRecipe()
    doc = recipe.run(path)
    return [doc.titles[0].text + ": " + doc.abstracts[0].text]