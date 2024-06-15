from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from agents.custom_react_agent import create_custom_react_agent
from criteria_aspect import CriteriaAspect
from langchain.agents import create_react_agent
from tools.tools import (
    generate_review,
    load_path,
    fetch_all_section_titles,
    create_markdown_string,
    get_openreview_reviews,
)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def evaluate_paper():
    ResearchQuestion = CriteriaAspect(1, "Research Question", "the questions that the manuscript tries to answer or solve.")
    ResearchQuestion.add_question("Are the research question(s) clearly stated by the authors?") # MaryShaw 2003, Thesien 2017, Wohlin 2015
    ResearchQuestion.add_question("Is the research question related to software engineering?") # Thesien 2017
    ResearchQuestion.add_question("How novel is the research question?") # MaryShaw 2003, Thesien 2017
    ResearchQuestion.add_question("How significant is the research question?") # MaryShaw 2003, Thesien 2017

    ResearchResult = CriteriaAspect(2, "Research Result", "what the authors found in the research.")
    ResearchResult.add_question("What is new in the research result compared to the previous research results?") # MaryShaw 2003
    ResearchResult.add_question("Are the results presented in this manuscript concrete and specific?") # MaryShaw 2003
    ResearchResult.add_question("Are there any interesting, novel, exciting results that significantly enhance researchers' ability to develop and maintain software, to know the quality of the software we develop, to recognize general principles about software, or to analyze the properties of software?") # MaryShaw 2003

    ResearchValidation = CriteriaAspect(3, "Research Validation", "refers to the process of providing clear and convincing evidence that research results are sound. Validation seeks to demonstrate that the findings are robust and reliable.")
    ResearchValidation.add_question("Is the evidence presented in the manuscript convincing?") # MaryShaw 2003
    ResearchValidation.add_question("Are there any research results that do not have concrete evidence supporting the claims made in the manuscript?") # MaryShaw 2003

    ResearchStrategy = CriteriaAspect(4, "Research Strategy", "how well the manuscript uses an appropriate combination of research question, results and validation.")
    ResearchStrategy.add_question("Does the manuscript presents a good combination of research question, result, and validation?") # MaryShaw 2003
    
    template = (
        "You are a Software Engineering Research Paper committee reviewer from a top conference. "
        "Your task is to leave constructive feedback for manusrctipt stated in the user's input. This feedback is directed to the manuscript's author to make modifications and resubmit their manuscript.\n"
        "You will be given the manuscript that needs to be reviewed. Here are some rules when you are reviewing the manuscript:\n"
        "- You should review the manuscript section by section. \n"
        "- You need to complete review all sections before you stop. \n"
        "- You need to consider the content of previous sections when reviewing a section.\n"
        "- You must review each paper by considering all the reviewing criteria.\n"
        "- You need to remember all section titles so that you are able to review the sections.\n"
        "Each section's review must be structured as a JSON blob (called $FEEDBACK) as below:\n"
        '```\n{{"Feedback":[A list of $REVIEW JSON blobs]}}\n'
        "A $REVIEW JSON blob must be structured as below:\n"
        "```{{\n"
        '"Section Title": $section_title,\n'
        '"Criterion": $criterion,\n'
        '"Review": $review\n}}\n```'
        "- $section_title will be the title of the section to be reviewed. Section titles in the provided manuscript are denoted by a markdown heading.\n"
        "- $criterion should be one of the criteria that the $comment is based on.\n"
        "- $review should be a constructive and practical feedback for the manuscript's author. "
        "The author will use this feedback to make necessary changes to their manuscript. "
        "You must avoid compliments and unnecessary feedback that don't help the author in improving their manuscript.\n\n"
        "In each section, for every sentence that you think needs to improve, you will need to take notes of that sentence."
        "Follow the exact step-by-step answering process in reviewing the manuscript:\n"
        "Section Title: the title of the section being reviewed currently starting from the first section.\n"
        "Criterion: the criterion through which you are assessing the section.\n"
        "Review: a $FEEDBACK JSON blob that contains a list of $REVIEW JSON blobs.\n"
        "... (this Section Title/Criterion/Review loop must be repeated as many times as needed to review all the sections and cover all evalaution criteria)\n"
        "Finally, you must review all the section reviews that you have left and "
        "Begin!\n\n Can you please review the following manuscript?\n{manuscript}\n"
        "Section Title: {section_title}\n"
        "Criterion: {criterion}\n"
        "Review: {agent_scratchpad}\n"
    )

    print(template)

    path = load_path()
    section_title_list = fetch_all_section_titles()
    markdown_string = create_markdown_string()

    llm = OpenAI(temperature=0, model="gpt-4o")  # gpt-4o

    prompt = PromptTemplate(
        input_variables=["manuscript", "agent_scratchpad"], template=template
    )

    agent = create_custom_react_agent(
        llm,
        section_title_list,
        criteria_list=[ResearchQuestion, ResearchResult, ResearchValidation, ResearchStrategy],
        prompt=prompt,
    )

    agent_chain = AgentExecutor(
        agent=agent,
        verbose=True,
        handle_parsing_errors=True,
    )

    agent_chain.invoke(
        {
            "manuscript": markdown_string,
        }
    )
