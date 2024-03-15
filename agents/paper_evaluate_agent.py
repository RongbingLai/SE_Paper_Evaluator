from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from tools import get_paper 

def evaluate_paper(title: str, aspects: list, questions: dict) -> dict:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    evaluation_template = """
    Evaluate the research paper titled "{title}" based on the following aspects and questions:
    {evaluation_questions}
    """

    evaluation_questions = "\n".join([f"{aspect}: {questions[aspect]}" for aspect in aspects])

    prompt_template = PromptTemplate(
        template=evaluation_template,
        input_variables=["title", "evaluation_questions"]
    )

    tools_for_agent = [
        Tool(
            name="Retrieve Criteria Paper",
            func=get_paper, 
            description="Retrieves relevant sections from criteria papers stored locally.",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={
            "title": title,
            "evaluation_questions": evaluation_questions
        }
    )

    return result["output"]
