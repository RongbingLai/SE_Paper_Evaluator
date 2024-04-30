from dotenv import load_dotenv
load_dotenv()
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from papermage.recipes import CoreRecipe
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def fetch_all_section_titles(path: str) -> list:
    recipe = CoreRecipe()
    doc = recipe.run(path)
    titles = []
    for section in doc.sections:
        titles.append(section.text)

    return titles

section_title_list = fetch_all_section_titles("/Users/crystalalice/Desktop/ICSHP_Research/SE_paper/Software_Documentation_Issues_Unveiled.pdf")

def fetch_section_content_by_titles(section_title: list, path: str) -> str:
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
                    if content_started and (line_text.replace(" ", "") == section_title_list[next_index].replace(" ","")):
                        return content

                    if content_started:
                        content += line_text + "\n"
    return content

section_content = fetch_section_content_by_titles('IV. R ESULTS D ISCUSSION', "/Users/crystalalice/Desktop/ICSHP_Research/SE_paper/Software_Documentation_Issues_Unveiled.pdf")

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
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
        'k': 10
    })
    llm = OpenAI(temperature=0.0)
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"verbose":True}
    )

    try:
        definition = qa.invoke("What kind of result is considered to be specific and concrete? What kind of results are not precise?")['result']
        print(definition)
    except Exception as e:
        print(f"An error occurred: {e}")
    chat = ChatOpenAI(temperature=0)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system",(
            "You are a review committee member at a established software engineering conference. You will be given a section of a manuscruipt along with a review question.\n"
        "While reviewing the section, you need to consider the assessment criteria. You should review whole the section based on the review question and the assessment criteria.\n"
        "Here is the assessment criteria:\n```\n{criterion_definition}\n```\n"
        "Your review must be formatted as below for each part of the section on which you want to leave a comment:\n"
        "```Manuscript Text: [the sentence(s) from the section that you want to leave comment on]\nReview: [Your comment]```\n"
        "Please remember to leave comments on all parts of the section that are relevant to the review question."
        )
            ),
            ("user", "Section:\n```\n{section_content}\n```\n\nReview Question: {question}\n\nYour comments:")
    ])
    chain = chat_prompt | chat
    # temp = '''
    # Given the section below, answer the question in detail with reference sentences from the section content.

    # Section Content: 
    # {section_content}
    # END OF SECTION

    # Assessment Criteria:
    # {criterion_definition}

    # Question:
    # {question}

    # In your final answer, you need to include two parts using this format:
    #     ```
    #     Manuscript Text: [the sentence(s) that you want to review in the section content]
    #     Review: [Your evaluation]
    #     ... (you should have enough Manuscript Text/Review to cover the review for the whole section content)
    #     ```
    # '''

    # question = (
    #     f"Given the section below, answer the question.\n"
    #     "Section Content:\n{section_content}\nEND OF SECTION\n\n"
    #     "Assessment Criteria:\n{criterion_definition}"
    #     "Question: {question}"
    # )

    # prompt = PromptTemplate(template=temp, input_variables=["section_content","criterion_definition","question"])
    # chain = prompt | llm
    res = chain.invoke({
        "section_content": section_content,
        "criterion_definition": definition,
        "question": question,
    })
    return res

print(generate_review(section_content, "research result: Is the result concrete and specific?"))