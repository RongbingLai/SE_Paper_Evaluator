from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from agents.search_agent import search_paper

if __name__ == "__main__":
    evaluation_aspects = [
        "reseach question",
        "research method",
        "research result",
        "research validation",
        "research strategy"
    ]

    questions = {
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

    paper_title = "paper for evaluation"
    evaluation_results = search_paper(paper_title, evaluation_aspects, questions)

    print(evaluation_results)