from agents.paper_evaluate_agent import evaluate_paper
from langchain_community.document_loaders import PyPDFLoader

QUESTIONS = {
    "research question": "In the research question, what, precisely, does the author claim to contribute?",
    "research question": "What larger question does this address?",
    "research result": "What, specifically, is the research result?",
    "research result": "How can readers apply this result?",
    "research result": "Is the result concrete and specific?",
    "research method": "What research method is used?",
    "research validation": "What evidence is presented to support the claim?",
    "research validation": "What kind of evidence is offered?",
    "research validation": "Does it meet the usual standard of the subdiscipline?",
    "research validation": "Is the evaluation described clearly and accurately?",
    "research validation": "Is the evidence and validation related to the claim?",
    "research strategy": "Does the author use a good combination of research question, research result and research validation types?"
}

PAPER_PATH = '/Users/crystalalice/Desktop/ICSHP_Research/SE_paper/Software_Documentation_Issues_Unveiled.pdf'

def get_paper_content() -> str:
    '''Returns a string of research paper content''' 
    content = ''

    loader = PyPDFLoader(PAPER_PATH)
    pages = loader.load_and_split()
    for p in pages:
        content += p.page_content

    return content

def get_paper_with_page(page_num: int) -> str:
    loader = PyPDFLoader(PAPER_PATH)
    pages = loader.load_and_split()

    return pages[page_num].page_content

if __name__ == "__main__":
    content = get_paper_with_page(8)
    evaluation_results = evaluate_paper(content, QUESTIONS)
    print(evaluation_results)