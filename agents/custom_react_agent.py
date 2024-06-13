from __future__ import annotations

from typing import List, Optional, Sequence, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents import AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import ToolsRenderer, render_text_description

from criteria_aspect import CriteriaAspect

import csv


def create_custom_react_agent(
    llm: BaseLanguageModel,
    section_title_list: List[str],
    criteria_list: List[CriteriaAspect],
    prompt: BasePromptTemplate,
    output_parser: Optional[AgentOutputParser] = None,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    # Check for missing variables in the prompt
    missing_vars = {"manuscript", "agent_scratchpad", "section_title", "criterion", "criteria_descriptions"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")
    
    # Define stop sequence, bind to llm
    if stop_sequence:
        stop = ["\nSection Title: "] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm

    # Set up output parser
    output_parser = output_parser or ReActSingleInputOutputParser()

    # Define agent logic
    agent = RunnablePassthrough.assign(
        agent_scratchpad=""
    )

    # Initialize comments.csv to contain results
    with open('comments.csv', mode='w+', newline='') as output_file:
        writer=csv.writer(output_file)
        writer.writerow(['Section Title', 'Criterion', 'Response'])

    def write_to_csv(data):
        with open('comments.csv', mode='a', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([data['section_title'], data['criterion'], data['response']])

    # Iterate through sections and criteria
    for section_title in section_title_list:
        for criteria in criteria_list:
            for criterion in criteria.get_all_questions():
                agent |= RunnablePassthrough.assign(
                    section_title=section_title,
                    criterion=criterion.aspect
                ) | prompt.partial(
                    section_title=lambda x: x["section_title"],
                    criterion=lambda x: x["criterion"],
                    agent_scratchpad=lambda x: x["agent_scratchpad"]
                ) | llm_with_stop | output_parser | RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: x["agent_scratchpad"] + x.get("response", "")
                )

                write_to_csv(agent)

    return agent
