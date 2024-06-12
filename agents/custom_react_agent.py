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

def create_custom_react_agent(
    llm: BaseLanguageModel,
    section_title_list: List[str],
    criteria_list: List[CriteriaAspect],
    intiial_prompt: BasePromptTemplate,
    subsequent_prompt: BasePromptTemplate,
    output_parser: Optional[AgentOutputParser] = None,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    # Check for missing variables in the prompt
    missing_vars = {"manuscript", "agent_scratchpad"}.difference(
        intiial_prompt.input_variables + list(intiial_prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")
    
    # Define stop sequence
    stop = ["\nSection Title: "] if stop_sequence is True else stop_sequence

    # Set up output parser
    output_parser = output_parser or ReActSingleInputOutputParser()

    # Define agent logic
    agent = RunnablePassthrough.assign(
        section_title=lambda x: section_title_list.pop(0),  # Get section title dynamically
    ) | intiial_prompt.partial(section_title=lambda x: x["section_title"]) | llm.bind(stop=stop)

    for section_title in section_title_list:
        section_reviews = ""
        for criterion in criteria_list:
            for question in criterion.questions:
                agent |= RunnablePassthrough.assign(
                    question=lambda x: question  # Get current question dynamically
                ) | subsequent_prompt.partial(
                    section_title=lambda x: section_title,
                    criterion=lambda x: criterion.aspect,
                    question=lambda x: question,
                ) | llm.bind(stop=stop) | output_parser | RunnablePassthrough.assign(
                    section_reviews=lambda x: x["agent_scratchpad"]
                )
        agent |= RunnablePassthrough.assign(
            agent_scratchpad=lambda x: section_reviews
        )

    return agent
