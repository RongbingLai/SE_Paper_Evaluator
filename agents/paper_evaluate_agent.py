from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import StructuredTool
from tools.tools import (
    generate_review,
    fetch_all_section_titles,
    fetch_section_content_by_titles,
)
from langchain.memory import ConversationBufferMemory

PAPER_PATH = "/Users/crystalalice/Desktop/ICSHP_Research/SE_paper/Software_Documentation_Issues_Unveiled.pdf"

EVAL_QUESTIONS = [
    (
        "research question",
        "In the research question, what, precisely, does the author claim to contribute?",
    ),
    ("research question", "What larger question does this address?"),
    ("research result", "What, specifically, is the research result?"),
    ("research result", "How can readers apply this result?"),
    ("research result", "Is the result concrete and specific?"),
    ("research method", "What research method is used?"),
    ("research validation", "What evidence is presented to support the claim?"),
    ("research validation", "What kind of evidence is offered?"),
    ("research validation", "Does it meet the usual standard of the subdiscipline?"),
    ("research validation", "Is the evaluation described clearly and accurately?"),
    ("research validation", "Is the evidence and validation related to the claim?"),
    (
        "research strategy",
        "Does the author use a good combination of research question, research result and research validation types?",
    ),
]


# https://www.jmu.edu/uwc/_files/link-library/empirical/findings-results_section_overview.pdf
def evaluate_paper():
    # manuscript
    # Reviewing a manuscript
    template = """
        You are a Software Engineering Research Paper committee reviewer from a top 
        conference. Your task is to evaluate the quality of a submission stated in the user 
        input from the following aspects:

        1. Research Question: the question that the manuscript tries to answer or solve.

        2. Research Results: describes what the authors found when they 
        analyzed their data. Its primary purpose is to use the data collected to answer the research 
        question(s) posed in the introduction, even if the findings challenge the hypothesis.

        3. Research Methodology: the strategies, processes or techniques utilized in the collection of 
        data or evidence for analysis in order to uncover new information or create better understanding 
        of a topic.

        4. Research Validation: refers to the process of providing clear and convincing evidence that 
        research results are sound. Validation seeks to demonstrate that the findings are robust and reliable.

        5. Research Strategy: how well the manuscript uses an approriate combination of research
        question, results, methodology and validation. 

        You have a list of questions that top Software Engineering committee reviewers will ask when they
        review manuscripts. Each question is in the format "Aspect: Question" where Aspect is one of the 
        research aspects listed above. Here is the list of evaluation questions:

        {eval_questions_str}

        You will be given by a path of the manuscript that needs to be reviewed.
        You are supposed to review the manuscript section by section. 
        
        You should use a tool to fetch all the section titles from the manuscript.
        You should use a tool to fetch the content of a given section in the manuscript.
        You should use a tool to get the review of a section in the manuscript. 

        For every sentence that you think needs to improve, you will need to 
        take notes of that sentence. It will be used in your final answer.
        
        You have access to the following tools:
        {tools}

        Follow the exact step-by-step answering process in reviewing the manuscript:

        ```
        Thought: You should always think what you want to do
        Action: the action to take, should be exactly one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        In your final answer, you need to include two parts using this format:
        ```
        Manuscript Text: [the text that you want to review in the manuscript]
        Review: [Your evaluation]
        ... (you should have enough Manuscript Text/Review to cover the review for the whole manuscript)
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
    """

    # list of tools
    tools = [
        Tool(
            name="Fetch All Section Titles",
            func=fetch_all_section_titles,
            description="Use this tool when you want to fetch all section titles from the manuscript. Input should be a string of path to the manuscript.",
        ),
        Tool(
            name="Fetch Section Content by Title",
            func=fetch_section_content_by_titles,
            description="Use this tool when you want to fetch the content of a section from the manuscript. Use this tool with arguments like `{``section_title``: str, ``path``: str}` when you need to retrieve the section content. Use double quotes for key and value strings in the format.",
        ),
        Tool(
            name="Generate Review",
            func=generate_review,
            description="Use this tool when you want to get a review for a section to the evaluation question. Input should be a string of one of the questions in the evaluation question list and a string of section content that you want to review.",
        ),
        # A tool to get the human reviews of a/a few similar paper(s) (Openreview DB / Use the openrview API) -> The reviews human reviewers left for a manuscript
    ]

    eval_questions_str = "\n".join(f"{key}: {value}" for key, value in EVAL_QUESTIONS)

    initial_input = (
        f"Evaluate the manuscript. The path of the manuscript is {PAPER_PATH}"
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    memory = ConversationBufferMemory(memory_key="chat_history")

    prompt = PromptTemplate.from_template(template=template).partial(
        input=initial_input,
        eval_questions_str=eval_questions_str,
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm_with_stop = llm.bind(stop=["\nObservation"])

    output_parser = ReActSingleInputOutputParser()

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )

    agent_chain = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    agent_chain.invoke(
        {
            "input": initial_input,
        }
    )
