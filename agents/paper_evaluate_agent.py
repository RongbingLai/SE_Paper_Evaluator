from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from papermage.recipes import CoreRecipe
from tools.tools import (
    generate_review,
    get_openreview_reviews
)

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_path():
    global path
    with open('tools/current_path.txt', 'r') as f:
        path = f.read()

def fetch_all_section_titles() -> list:
    titles = []
    recipe = CoreRecipe()
    path = path.strip("'").strip('"')
    doc = recipe.run(path)
    for section in doc.sections:
        titles.append(section.text)

    return titles

def evaluate_paper():
    template = """
You are a Software Engineering Research Paper committee reviewer from a top 
conference. Your task is to evaluate the quality of a submission stated in the user 
input.

You will be given by a path of the manuscript that needs to be reviewed.
You are supposed to review the manuscript section by section. You need to complete review all sections
before you finish. 

You should use a tool to fetch all the section titles from the manuscript.
You should use a tool to generate the review of a section in the manuscript. 
You should use a tool to get the actual reviews for a research paper that is similar to the manuscript.

Your action order should be:
1. Get manuscript section titles
2. [Optional] Check if there are reviews for similiar research paper and get reviews as reference
3. For each section, generate review for the section by asking evaluation questions from the list
4. Repeat step 2-3 until you reviewed all of the sections listed

You need to remember all the section titles so that you are able to review the sections.
In each section, for every sentence that you think needs to improve, you will need to
take notes of that sentence. It will be used in your final answer.

You have access to the following tools:
{tools}

Follow the exact step-by-step answering process in reviewing the manuscript:

```
Thought: You should always think what you want to do
Action: the action to take, should be exactly one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action that will be returned by the tool (DO NOT GENERATE THIS PART)
... (this Thought/Action/Action Input/Observation can repeat N times)
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Question: {input}
{agent_scratchpad}
    """

    # list of tools
    tools = [
        # Tool(
        #     name="Fetch All Section Titles",
        #     func=fetch_all_section_titles,
        #     description="Use this tool when you want to fetch all section titles from the manuscript. \
        #                  Input should be a string of path to the manuscript.",
        # ),
        # Tool(
        #     name="Fetch Section Content by Title",
        #     func=fetch_section_content_by_titles,
        #     description="Use this tool when you want to fetch the content of a section from the manuscript. \
        #                  Use this tool with arguments like `{``section_title``: str, ``path``: str}` when you need to \
        #                  retrieve the section content. Use double quotes for key and value strings in the format.",
        # ),
        Tool(
            name="Generate Review",
            func=generate_review,
            description="Use this tool when you want to generate a review for a section by answering the evaluation question. \
                         The input of this tool is a section_title that contains a string of section title",
        ),
        # Tool(
        #     name="Get Openreview Reviews",
        #     func=get_openreview_reviews,
        #     description="Use this tool when you want to get an actual review from a research paper that is similar to the manuscript. \
        #                  Input should be a string of path to the manuscript.",
        # )
    ]

    section_title_list = fetch_all_section_titles()
    section_title_str = (", ").join(section_title_list())
    load_path()

    initial_input = (
        f"Can you please review the the manuscript located at: '{path}'?\n\nHere are the section titles of the manuscript:\n{section_title_str}"
    )
    print(initial_input)

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    
    prompt = PromptTemplate(input_variables=["input", "agent_scratchpad"], template=template)

    agent = create_react_agent(
        llm, tools, prompt, 
)

    agent_chain = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    agent_chain.invoke(
        {
            "input": initial_input,
        }
    )
