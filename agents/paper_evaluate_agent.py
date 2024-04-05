from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from tools.tools import get_relevant_section

def evaluate_paper(content: str, questions: dict) -> dict:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    evaluation_questions = "\n".join(questions)

    evaluation_template = """
        Evaluate the research paper {content} based on the following aspects and questions:
        {evaluation_questions}. In your final answer, for each aspect, write a paragraph to
        evaluate the quality from this aspect. 
    """

    prompt_template = PromptTemplate(
        template=evaluation_template,
        input_variables=["content", "evaluation_questions"]
    )

    tools_for_agent = [
        Tool(
            name="Retrieve Criteria Paper",
            func=get_relevant_section, 
            description="Retrieves relevant sections from criteria papers for evaluation stored locally.",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={
            "input": prompt_template.format_prompt(content=content, evaluation_questions=evaluation_questions)
        }
    )

    return result["output"]