# -*- coding: utf-8 -*-
"""
Prompt engineering and management for Stackademy.
Handles function calling and response parsing.
"""

import json
from typing import Optional, Union

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from app import settings
from app.const import MISSING, ToolChoice
from app.logging_config import get_logger, setup_logging
from app.settings import LLM_ASSISTANT_NAME, LLM_TOOL_CHOICE
from app.stackademy import stackademy_app
from app.synastra import synastra_app
from app.utils import color_text, dump_json_colored

setup_logging()
logger = get_logger(__name__)

MessagesType = list[
    Union[
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam,
        ChatCompletionToolMessageParam,
    ]
]

SYSTEM_PROMPT_CAPSTONE = """
You are a pleasant but honest astrologer with many years of experience who provides relationship advice using 
structured astrology compatibility data derived from Natal Charts. A tool will be provided to you to use once 
the Birth Month, Birth Year, Birth Country, and Birth City of the User AND the Birth Month, Birth Year, 
Birth Country, and Birth City of the user’s partner. If you are unable to provide the necessary astrological 
natal chart calculator parameters, then ask the user for clarification. Once you have this information, you may 
call the tool that is provided to you with those details to get the formatted compatibility report with many 
relationship domain areas and compatibility scores. You will use this synastry compatibility report to respond 
to follow up questions from the user. Use only the provided areas, scores, descriptions, and key factors. 
Do not invent additional placements, houses, aspects, or chart facts.
"""

SYSTEM_PROMPT = """
You are Synastra, a playful but practical natal/synastry relationship chatbot.

You treat astrology as symbolic and personality-style context, not as scientific fact,
certainty, fate, diagnosis, or proof.

You have access to one tool: get_synastry_report.

The tool requires exactly two subjects. Each subject must include:
- optional name
- birth_data.year
- birth_data.month
- birth_data.city
- birth_data.country_code

Before calling get_synastry_report, determine whether the user has provided complete
required birth data for exactly two people.

If either person is missing required details, ask a concise clarification question.
Do not call the tool with incomplete data.

Call get_synastry_report only after complete data for exactly two subjects is available.

If a valid synastry report is already present in the conversation, answer using that
report instead of calling the tool again, unless the user changes one of the birth details.

Do not copy, inherit, or reuse one subject’s birth city or country for the other subject unless 
the user explicitly says both people were born there.

Location details belong only to the subject they are grammatically attached to.

If one subject has month/year but no city or country, ask for that subject’s missing birth city 
and country before calling the tool.

Never guess missing birth city or country from another person’s details.

After get_synastry_report returns a valid report, answer relationship questions using
only the returned compatibility data:
- life_area_compatibility.area
- life_area_compatibility.compatibility_score
- life_area_compatibility.description
- life_area_compatibility.key_factors

Do not invent chart factors, houses, placements, signs, aspects, scores,
compatibility areas, or other astrology details.

If the report does not contain information relevant to the user’s question, say so.
You may still give practical relationship advice, but clearly separate it from what
the report supports.

For safety, health, legal, financial, or major life decisions, do not rely on astrology.
Give practical caution.

Keep responses concise, warm, and useful.
"""

messages: MessagesType = [
    ChatCompletionSystemMessageParam(
        role="system",
        content=SYSTEM_PROMPT,
        name=LLM_ASSISTANT_NAME,
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="How can I assist you with Stackademy today?",
        name=LLM_ASSISTANT_NAME,
    ),
]


def handle_function_call(function_name: str, arguments: dict) -> str:
    """Handle function calls from the OpenAI API."""
    if function_name == "get_courses":
        # Extract parameters with defaults
        description = arguments.get("description")
        max_cost = arguments.get("max_cost")

        # Call the actual function
        courses = stackademy_app.get_courses(description=description, max_cost=max_cost)

        # Return as JSON string
        return json.dumps(courses, default=str, indent=2)

    if function_name == "register_course":
        course_code = arguments.get("course_code", MISSING)
        email = arguments.get("email", MISSING)
        full_name = arguments.get("full_name", MISSING)

        # Call the actual function
        success = stackademy_app.register_course(course_code=course_code, email=email, full_name=full_name)

        # Return result as JSON string
        return json.dumps({"success": success})

    if function_name == "get_synastry_report":
        # Return as JSON string
        return synastra_app.get_synastry_report(**arguments)

    return json.dumps({"error": f"Unknown function: {function_name}"})


def process_tool_calls(message: ChatCompletionMessage) -> list[str]:
    """Process tool calls in the messages list."""
    functions_called = []
    if not isinstance(message, ChatCompletionMessage) or not message.tool_calls:
        return functions_called
    for tool_call in message.tool_calls:

        if tool_call.type == "function":
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            functions_called.append(function_name)
            tool_calls_param = [
                ChatCompletionMessageFunctionToolCallParam(
                    id=tool_call.id,
                    type="function",
                    function={
                        "name": function_name,
                        "arguments": tool_call.function.arguments,
                    },
                )
            ]
            assistant_content = message.content if message.content else "Accessing tool..."
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=assistant_content, tool_calls=tool_calls_param, name=LLM_ASSISTANT_NAME
                )
            )
            msg = f"Calling function: {function_name} with args {json.dumps(function_args)}"
            logger.info(color_text(msg, "green"))

            function_result = handle_function_call(function_name, function_args)

            tool_message = ChatCompletionToolMessageParam(
                role="tool", content=function_result, tool_call_id=tool_call.id
            )
            messages.append(tool_message)

        logger.debug(
            "Updated messages: %s",
            [dump_json_colored(msg.model_dump(), "blue") if not isinstance(msg, dict) else msg for msg in messages],
        )
    return functions_called


def completion(prompt: str) -> tuple[Optional[ChatCompletion], list[str]]:
    """LLM text completion"""

    def handle_completion(tools: list | None = None, tool_choice=None) -> ChatCompletion:
        """Handle the OpenAI chat completion call."""
        openai.api_key = settings.OPENAI_API_KEY
        model = settings.OPENAI_API_MODEL

        try:
            logger.debug(
                "Sending messages to OpenAI: %s %s",
                dump_json_colored(messages, "blue"),
                dump_json_colored(tools, "blue"),
            )

            request_args = {
                "model": model,
                "messages": messages,
                "temperature": settings.OPENAI_API_TEMPERATURE,
                "max_tokens": settings.OPENAI_API_MAX_TOKENS,
            }

            if tools:
                request_args["tools"] = tools

                if tool_choice is not None:
                    request_args["tool_choice"] = tool_choice

            response = openai.chat.completions.create(**request_args)

            logger.debug("OpenAI response: %s", dump_json_colored(response.model_dump(), "green"))
            return response
        except openai.RateLimitError as e:
            logger.error("OpenAI rate limit exceeded: %s", e)
            raise
        except openai.APIConnectionError as e:
            logger.error("OpenAI API connection error: %s", e)
            raise
        except openai.AuthenticationError as e:
            logger.error("OpenAI authentication error. Did you set OPENAI_API_KEY in your .env file? %s", e)
            raise
        except openai.BadRequestError as e:
            logger.error("OpenAI bad request error: %s", e)
            raise
        except openai.APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Unexpected error during OpenAI completion: %s", e)
            raise

    if not prompt.strip():
        logger.warning("Received empty prompt.")
        return None, []

    messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))
    functions_called = []

    available_tools = synastra_app.available_tools(messages)

    response = handle_completion(
        # tool_choice={"type": "function", "function": {"name": "get_courses"}},
        tool_choice=LLM_TOOL_CHOICE,
        tools=available_tools
    )
    logger.debug("Initial response: %s", dump_json_colored(response.model_dump(), "green"))

    message = response.choices[0].message
    while message.tool_calls:
        if message.content and "Goodbye!" in message.content:
            break
        functions_called = process_tool_calls(message)

        available_tools = synastra_app.available_tools(messages)

        logger.debug("Available tools count: %s", len(available_tools))

        response = handle_completion(
            tools=available_tools,
            tool_choice=ToolChoice.AUTO,
        )

        message = response.choices[0].message
        logger.debug("Updated response: %s", dump_json_colored(response.model_dump(), "green"))

    return response, functions_called
