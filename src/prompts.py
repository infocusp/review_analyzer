"""This file contains prompt templates."""

from typing import List, Tuple, Union

from langchain import output_parsers
from langchain import prompts

from src import few_shot_examples
from utils import data_models


def format_assistant_examples(
    example_reviews: List[List[Tuple[str, Union[prompts.PromptTemplate, str]]]]
) -> List[Tuple[str, str]]:
    """Arranges all assistant examples in a chat format.
    
    Args:
        example_reviews (List[List[Tuple[str, Union[PromptTemplate, str]]]): 
            A list of few-shot examples where each example is a list of (role, message) tuples.
    Returns:
        assistant_prompt (List[Tuple[str, str]]):
            A flattened list of formatted (role, message) pairs, suitable for sending to the model.
    """

    assistant_examples = []
    for user_message, assistant_message in example_reviews:
        user_role, user_content = user_message
        assistant_role, assistant_content = assistant_message

        if isinstance(user_content, prompts.PromptTemplate):
            user_content = user_content.format()  # Ensure it's a string

        # Append user and assistant messages one by one
        assistant_examples.append((user_role, user_content))
        assistant_examples.append((assistant_role, assistant_content))

    return assistant_examples


def get_user_prompt(existing_entities: List[str],
                    formatted_reviews: str) -> str:
    """Generates a structured user prompt using PromptTemplate.

    Args:
        formatted_reviews(str): The reviews formatted as a string.
        task_description(str): Command to the model.
    
    Returns:
      user_prompt (str): A formatted user prompt.
    """
    user_prompt_template = prompts.PromptTemplate(
        input_variables=["task_description", "formatted_reviews"],
        template=
        """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        {existing_entities} 

        You are tasked with extracting entities/themes/topics and their corresponding sentiment from the new set of reviews:
        {formatted_reviews}
        """)

    return user_prompt_template.format(existing_entities=existing_entities,
                                       formatted_reviews=formatted_reviews)


def get_system_propmt(existing_entities: List[str] = []) -> str:
    """Generates a structured system prompt using PromptTemplate.

    Args:
        existing_entities(List[str]): List of extracted entities.
    
    Returns:
        system_prompt (str): A formatted system prompt.
    """
    parser = output_parsers.PydanticOutputParser(
        pydantic_object=data_models.AggregatedResults)
    system_prompt = prompts.PromptTemplate(partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
                                           template="""
            You are an AI assistant specializing in **extracting structured insights from Spotify user reviews**.
            Your goal is to **identify key entities, classify sentiment, and avoid redundant entity creation**.

            ### **Key Responsibilities**
            - **Extract entities**: Identify relevant aspects in reviews.
            - **Standardize names**: Group similar entities to avoid duplication.
            - **Assign sentiment**: Classify as `Positive` or `Negative`, ignoring neutral statements.
            - **Track occurrences**: Store review IDs under respective sentiment categories.

            ### **Important Instructions**
            - Focus on **meaning and implication** of the review sentence, not just keywords.
        """)

    return system_prompt.format(existing_entities=existing_entities)


chat_prompt_template = prompts.ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),  # System message
    *format_assistant_examples(few_shot_examples.generalized_examples
                              ),  # Example input/output from assistant
    ("user", "{user_prompt}")  # User's actual input
])
