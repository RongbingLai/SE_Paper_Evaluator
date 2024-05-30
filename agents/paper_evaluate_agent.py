from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from tools.tools import (
    generate_review,
    load_path,
    fetch_all_section_titles,
    get_openreview_reviews
)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def evaluate_paper():
#     old_template = """
# You are a Software Engineering Research Paper committee reviewer from a top 
# conference. Your task is to evaluate the quality of a submission stated in the user 
# input.

# You will be given by a path of the manuscript that needs to be reviewed.
# You are supposed to review the manuscript section by section. You need to complete review all sections
# before you stop. 

# You should use a tool to generate the review of a section in the manuscript. 

# You need to remember all section titles so that you are able to review the sections.
# In each section, for every sentence that you think needs to improve, you will need to
# take notes of that sentence. It will be used in your final answer.

# You have access to the following tools:
# {tools}

# Follow the exact step-by-step answering process in reviewing the manuscript:

# ```
# Thought: You should always think what you want to do
# Action: the action to take, should be exactly one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action that will be returned by the tool (DO NOT GENERATE THIS PART)
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# ```

# When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

# ```
# Thought: Do I need to use a tool? No
# Final Answer: [your response here]
# ```

# Begin!

# Question: {input}
# {agent_scratchpad}
#     """

    template = """
You are a Software Engineering Research Paper committee reviewer from a top 
conference. Your task is to evaluate the quality of a submission stated in the user 
input. You only need to review the section. 

You will be given by a path of the manuscript that needs to be reviewed, and a title of the section
that you needs to review.

You should use a tool to generate the review of a section in the manuscript. 

You need to review the section sentence by sentence. For every sentence that you would like to leave
comment on, you need to take notes of that sentence. It will be used in your final answer.

You have access to the following tools:
{tools}

You must always use one of the tools as your action.

Follow the exact step-by-step answering process in reviewing the manuscript:

```
Thought: You should always think what you want to do
Action: the action to take, should be exactly one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action that will be returned by the tool
... (this Thought/Action/Action Input/Observation can repeat N times)
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Your final review result must be a JSON blob structured as below:
```
{{
"Manuscript Text": $manuscript text,
"Comment": $comment
}}
```

- $Manuscript Text should be a sentence or sentences that you want to leave comment on. Manuscript Text must be from the section content of the manuscript
- $Comment should be a constructive and practical review for the manuscript's author

Begin!

Question: {input}
{agent_scratchpad}
    """

    # list of tools
    tools = [
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

    # initial_input = (
    #     f"Can you please review the the manuscript located at: '{path}'?\n\nHere are the section titles of the manuscript:\n{section_title_str}"
    # )

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

    for section_title in section_title_list:
        input_prompt = f"Can you please review the the manuscript located at: '{path}'?\nPlease review the section titled: '{section_title}'"
        agent_chain.invoke(
            {
                "input": input_prompt,
            }
        )

    # agent_chain.invoke(
    #     {
    #         "input": initial_input,
    #     }
    # )
