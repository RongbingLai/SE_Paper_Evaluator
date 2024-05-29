from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts.chat import ChatPromptTemplate
from papermage.recipes import CoreRecipe
from langchain_community.retrievers import BM25Retriever
import json

section_title_list = []

def load_path():
    global path
    with open('current_path.txt', 'r') as f:
        path = f.read()

load_path()

@tool
def fetch_all_section_titles() -> list:
    """
    This function extracts the titles of all sections.

    Parameters:
        none

    Returns:
        list: A list of section titles found in the specified manuscript.

    Example:
        If the manuscript has sections 'Introduction', 'Literature Review', 'Methodology',
        the function will return:
        ['Introduction', 'Literature Review', 'Methodology']
    """
    recipe = CoreRecipe()
    path = path.strip("'").strip('"')
    doc = recipe.run(path)
    for section in doc.sections:
        section_title_list.append(section.text)

    return section_title_list


def _fetch_section_content_by_titles(section_title: str) -> str:
    """
    The function fetechs the content of a manuscript section for the given section
    title.

    Parameters:
        section_title: the title of the manuscript section

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
def generate_review(section_title: str) -> str: 
    """
    The function returns a response that answers the question regarding the section
    content by searching in the local faiss database.

    Parameters:
        section_title (str): the title of the manuscript section

    Returns:
        str: The response answering the given question that evaluates the section content.
    """
    assert section_title in section_title_list, "It seems like you have not provided a correct section titlte. Please use one of the section titles that was provided to you."
    section_content = _fetch_section_content_by_titles(section_title, path)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_index_full_criteria",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"verbose": True},
    )

    question_dict = _get_criteria_questions(section_content)
    eval_questions = []
    definitions = {}

    for value in question_dict.values():
        for q in value:
            eval_questions.append(q['Question'])
            for subquestion in q['Subquestions']:
                result = qa.invoke(subquestion)['result']
                definitions[subquestion] = _summarize_definition(result)

    definitions_str = "\n".join(f"{key}: {value}" for key, value in definitions.items())
    questions_str = "\n".join(q for q in eval_questions)

    chat = ChatOpenAI(temperature=0)

    template = """
You are a review committee member at an established software engineering conference. You will be given a section 
of a manuscruipt along with a list of review questions. While reviewing the section, you need to consider the 
assessment criteria. You should review whole the section based on the review question and the assessment criteria.

Here is the assessment criteria:
{definitions_str}

Your final result must be a JSON blob structured as below:
```
{{
"excerpt": $excerpt,
"comment": $comment
}}
```

- $excerpt should be the Manuscript Text that you want to leave comment on. Manuscript Text must be from the section content of the manuscript
- $comment should be a constructive and practical review for the manuscript's author 


Please remember to leave comments on all parts of the section that are relevant to the review question.
    """
    
    # """ You should make it formatted as below for each part of the section on which you want to leave a comment
    #     ```
    #     Manuscript Text: [sentence(s) from the section content that you want to leave comment on]
    #     Review: [Your comment]
    #     ```
    # """

    human_template = """
        Section:
        {section_content}
        
        Review Questions: 
        {questions_str}
        
        Your comments:
    """

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",template),
            ("user",human_template),
        ]
    )

    chat_prompt.format(section_content=section_content, definitions_str=definitions_str, questions_str=questions_str)

    chain = chat_prompt | chat

    res = chain.invoke(
        {
            "section_content": section_content,
            "definitions_str": definitions_str,
            "questions_str": questions_str,
        }
    )
    return res.content


def _summarize_definition(definition: str) -> str:
    llm = OpenAI(temperature=0)
    prompt = f"Summarize the following definition:\n\n{definition}"
    response = llm.invoke(prompt)
    return response

def _get_criteria_questions(section_content: str) -> dict:
    llm = ChatOpenAI(temperature=0)

    with open("tools/quality_checklist.json", "r") as file:
        checklist = json.load(file)

    template = """
        You are a review committee member at an established software engineering conference. You will be given 
        a section of a manuscruipt and a quality checklist that contains questions that review committee member
        might ask. 
        
        Here is the quality checklist: {checklist}. The quality checklist is in Python dictionary format. 
        The keys are research aspects regarding the manuscript. Each aspect has a list of review questions. 
        Each element of the list is a dictionary. For each element, key "Question"'s value is the question itself 
        and key "Subquestions"'s value are subquestions that needed to be answered in order to answer the main question.

        For the given section of manuscript, select main questions that are helpful for evaluating the section.
        Only select the ones that are highly related to the section. Select top 4 questions. If there are less than 4
        related questions, select them all. Do not select all questions.
        If there is no question related to this section, then return an empty dictionary. 
        Return a dictionary that contains all questions in json format. Ignore new line symbols.
    """

    # """
    # You are reviewing from these aspects:

    #     1. Research Question: the question that the manuscript tries to answer or solve.

    #     2. Research Results: describes what the authors found when they 
    #     analyzed their data. Its primary purpose is to use the data collected to answer the research 
    #     question(s) posed in the introduction, even if the findings challenge the hypothesis.

    #     3. Research Methodology: the strategies, processes or techniques utilized in the collection of 
    #     data or evidence for analysis in order to uncover new information or create better understanding 
    #     of a topic.

    #     4. Research Validation: refers to the process of providing clear and convincing evidence that 
    #     research results are sound. Validation seeks to demonstrate that the findings are robust and reliable.

    #     5. Research Strategy: how well the manuscript uses an approriate combination of research
    #     question, results, methodology and validation. """

    human_template = "What questions should be asked to review this section of the manuscript? {section_content}"

    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("user", human_template)]
    )

    chat_prompt.format_messages(checklist=checklist, section_content=section_content)

    chain = chat_prompt | llm

    res = chain.invoke(
        {
            "section_content": section_content,
            "checklist": checklist,
        }
    )

    selected_questions = json.loads(res.content)

    return selected_questions

@tool
def get_openreview_reviews() -> list:
    """
    The function finds the most similar research paper to the
    manuscript from OpenReview, which is a platform that contains the actual reviews
    for research papers, and gets the reviews for the paper as reference.

    Parameters:
        None

    Returns:
        list: A list of reviews for similar research papers
    """
    path = path.strip('"\'')
    paper_abstract = _get_paper_abstract(path)
    indexes = _find_similiar_paper(paper_abstract)
    reviews = []

    with open("correct_file.json", "r") as file:
        data = json.load(file)
        for index in indexes:
            for paper in data["orb_submissions"][index]["article_versions"].values():
                reviews.append((paper["title"], paper["reviews"]))
    return reviews


def _find_similiar_paper(paper_abstract: list) -> list:
    db_paper_abstracts = []
    indexes = []

    with open("correct_file.json", "r") as file:
        data = json.load(file)
        for submission in data["orb_submissions"]:
            if "0" in submission["article_versions"]:
                paper = submission["article_versions"]["0"]  # only need one version
                if paper["title"] and paper["abstract"] and (paper["reviews"] != []):
                    db_paper_abstracts.append(paper["title"] + ": " + paper["abstract"])
                    print(paper["title"] + ": " + paper["abstract"] + "\n\n")
                else:
                    db_paper_abstracts.append("")
    
    retriever = BM25Retriever.from_texts(db_paper_abstracts) # k: number of documents

    similiar_paper = retriever.invoke(paper_abstract)

    for paper in similiar_paper:
        indexes.append(db_paper_abstracts.index(paper.to_json()["kwargs"]["page_content"]))

    return indexes


def _get_paper_abstract() -> str:
    recipe = CoreRecipe()
    doc = recipe.run(path)
    return doc.titles[0].text + ": " + doc.abstracts[0].text
