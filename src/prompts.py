from typing import List, Tuple, Union

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from src import few_shot_examples


def format_assistant_examples(
    example_reviews: List[List[Tuple[str, Union[PromptTemplate, str]]]]
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

        if isinstance(user_content, PromptTemplate):
            user_content = user_content.format()  # Ensure it's a string

        # Append user and assistant messages one by one
        assistant_examples.append((user_role, user_content))
        assistant_examples.append((assistant_role, assistant_content))

    return assistant_examples


def get_user_prompt(
    formatted_reviews: str,
    task_description: str = "Extract entities and sentiment from these reviews:"
) -> str:
    """Generates a structured user prompt using PromptTemplate.

    Args:
        formatted_reviews(str): The reviews formatted as a string.
        task_description(str): Command to the model.
    
    Returns:
      user prompt (str): A formatted user prompt.
    """
    user_prompt_template = PromptTemplate(
        input_variables=["task_description", "formatted_reviews"],
        template="""
        {task_description}
        {formatted_reviews}
        """)

    return user_prompt_template.format(task_description=task_description,
                                       formatted_reviews=formatted_reviews)


def get_system_propmt(existing_entities: List[str] = []) -> str:
    """Generates a structured system prompt using PromptTemplate.

    Args:
        existing_entities(List[str]): List of extracted entities.
    
    Returns:
      user prompt (str): A formatted system prompt.
    """

    system_prompt = PromptTemplate(input_variables=["existing_entities"],
                                   template="""
            You are an AI assistant specializing in **extracting structured insights from Spotify user reviews**.
            Your goal is to **identify key entities, classify sentiment, and avoid redundant entity creation**.

            ### **Key Responsibilities**
            - **Extract entities**: Identify relevant aspects in reviews.
            - **Standardize names**: Group similar entities to avoid duplication.
            - **Assign sentiment**: Classify as `Positive` or `Negative`, ignoring neutral statements.
            - **Track occurrences**: Store review IDs under respective sentiment categories.

            ### **Existing Entities Context**
            Below are the entities identified so far across previous reviews.Reuse these entities whenever possible and do not create redundant entries.

            {existing_entities}

            ### **Response Format**
            Return structured **JSON only**, without explanations or markdown.
            If only "positive_reviews" are found for a particular entity, give empty list for "negative_reviews", vice versa.
        """)

    return system_prompt.format(existing_entities=existing_entities)


chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),  # System message
    *format_assistant_examples(few_shot_examples.spotify_examples
                              ),  # Example input/output from assistant
    ("user", "{user_prompt}")  # User's actual input
])