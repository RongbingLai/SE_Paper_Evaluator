from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from tools.tools import (
    generate_review,
    load_path,
    fetch_all_section_titles,
    get_openreview_reviews,
)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def evaluate_paper():
    template = (
        "You are a Software Engineering Research Paper committee reviewer from a top conference. "
        # "Your task is to: 1) leave constructive feedback for manusrctipt stated in the user's input 2) To predict its acceptance chance at {conference_name}.\n"
        "Your task is to leave constructive feedback for manusrctipt stated in the user's input. This feedback is directed to the manuscript's author to make modifications and resubmit their manuscript.\n"
        "The reviewing is done from 4 criteria as below:\n"
        "1. Research Questions: the questions that the manuscript tries to answer or solve. DEFINITION\n" #Criterion
        "\t1.1: What is the contribution of this manuscript by answering its research questions?" #Question(s) to answer in order to evaluate the manuscript from that criterion  # Cite Marysha 2002
        "..."
        ""
        "2. Research Results: describes what the authors found when they\n"
        "\t2.1: \n"
        "5. Results: Results should be specific.\n"
        "\t 5.1: Are the results presented in this paper considered concrete?"
        "You will be given a path of the manuscript that needs to be reviewed. Here are some rules when you are reviewing the manuscript:\n"
        "- You should review the manuscript section by section. \n"
        "- You need to complete review all sections before you stop. \n"
        "- You need to consider the content of previous sections when reviewing a section.\n"
        "- You must review each paper by considering all the reviewing criteria.\n"
        # "You should use a tool to generate the review of a section in the manuscript.\n"
        "You need to remember all section titles so that you are able to review the sections.\n"
        # "Your final answer must be a JSON blob structured as below:\n"
        "Each section's review must be structured as a JSON blob (called $FEEDBACK) as below:\n"
        '```\n{{"Feedback":[A list of $REVIEW JSON blobs]}}\n'
        "A $REVIEW JSON blob must be structured as below:\n"
        "```{{\n"
        '"Manuscript Text": $manuscript_text,\n'
        '"Comment": $comment,\n'
        '"Criterion": $criterion\n}}\n```'
        "- $manuscript_text should be a sentence or sentences that you want to leave comment on. Manuscript Text must be a direct quote from the manuscript\n"
        "- $comment should be a constructive and practical feedback for the manuscript's author. "
        "The author will use this feedback to make necessary changes to their manuscript. "
        "You must avoid compliments and unnecessary feedback that don't help the author in improving their manuscript.\n"
        "- $criterion should be one of the criteria that the $comment is based on.\n\n"
        "In each section, for every sentence that you think needs to improve, you will need to take notes of that sentence."
        # "You have access to the following tools:\n{tools}\n\n"
        "Follow the exact step-by-step answering process in reviewing the manuscript:\n"
        "Section Title: the title of the section being review currently starting from the first section.\n"
        "Criterion: the criterion through which you are assessing the section.\n"
        "Review: a $FEEDBACK JSON blob that contains a list of $REVIEW JSON blobs.\n"
        "... (this Section Title/Criterion/Review loop must be repeated as many times as needed to review all the sections and cover all evalaution criteria)\n"
        "Finally, you must review all the section reviews that you have left and"
        # "```\nThought: You should always think what you want to do\n"
        # "Action: the action to take, should be exactly one of [{tool_names}]\n"
        # "Action Input: the input to the action\n"
        # "Observation: the result of the action that will be returned by the tool (DO NOT GENERATE THIS PART)\n"
        # "... (this Thought/Action/Action Input/Observation can repeat N times)\n```\n\n"
        # "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n"
        # "```\nThought: Do I need to use a tool? No\nFinal Answer: $REVIEWS\n```\n\n"
        # "Begin!\nQuestion: {input}\n"
        "Begin!\n\n{input}\n"
        "{agent_scratchpad}"
    )

    #     template = """
    # You are a Software Engineering Research Paper committee reviewer from a top
    # conference. Your task is to evaluate the quality of a submission stated in the user
    # input. You only need to review the section.

    # You will be given by a path of the manuscript that needs to be reviewed, and a title of the section
    # that you needs to review.

    # You should use a tool to generate the review of a section in the manuscript.

    # You need to review the section sentence by sentence. For every sentence that you would like to leave
    # comment on, you need to take notes of that sentence. It will be used in your final answer.

    # You have access to the following tools:
    # {tools}

    # You must always use one of the tools as your action.

    # Follow the exact step-by-step answering process in reviewing the manuscript:

    # ```
    # Thought: You should always think what you want to do
    # Action: the action to take, should be exactly one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action that will be returned by the tool
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # ```

    # When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    # ```
    # Thought: Do I need to use a tool? No
    # Final Answer: [your response here]
    # ```

    # Your final review result must be a JSON blob structured as below:
    # ```
    # {{
    # "Manuscript Text": $manuscript text,
    # "Comment": $comment
    # }}
    # ```

    # - $Manuscript Text should be a sentence or sentences that you want to leave comment on. Manuscript Text must be from the section content of the manuscript
    # - $Comment should be a constructive and practical review for the manuscript's author

    # Begin!

    # Question: {input}
    # {agent_scratchpad}
    #     """

    # list of tools
    tools = [
        # Tool(
        #     name="Fetch All Section Titles",
        #     func=fetch_all_section_titles,
        #     description="Use this tool when you want to fetch all section titles from the manuscript. Input should be a string of path to the manuscript.",
        # ),
        Tool(
            name="Generate Review",
            func=generate_review,
            description="Use this tool when you want to generate a review for a section by answering the evaluation question. The input of this tool is a section_title that contains a string of section title",
        ),
        # Tool(
        #     name="Get Openreview Reviews",
        #     func=get_openreview_reviews,
        #     description="Use this tool when you want to get an actual review from a research paper that is similar to the manuscript. \
        #                  Input should be a string of path to the manuscript.",
        # )
    ]

    path = load_path()
    section_title_list = fetch_all_section_titles()
    section_title_str = ("\n").join(section_title_list)

    initial_input = f"Can you please review the the manuscript located at: '{path}'?\n\nHere are the section titles of the manuscript:\n{section_title_str}"

    llm = OpenAI(temperature=0, model="gpt-4o")  # gpt-4o

    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"], template=template
    )

    agent = create_react_agent(
        llm,
        tools,
        prompt,
    )

    agent_chain = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    # for section_title in section_title_list:
    #     input_prompt = f"Can you please review the the manuscript located at: '{path}'?\nPlease review the section titled: '{section_title}'"
    #     agent_chain.invoke(
    #         {
    #             "input": input_prompt,
    #         }
    #     )

    agent_chain.invoke(
        {
            "input": initial_input,
        }
    )
