# -*- coding: utf-8 -*-
"""
User registration and management for Synastra.
"""

from typing import Optional, Tuple

from .logging_config import get_logger, setup_logging
from .prompt import completion

setup_logging()
logger = get_logger(__name__)


def main(prompts: Optional[Tuple[str, ...]] = None) -> None:
    """Main function to demonstrate user registration."""
    print("=" * 50)
    print("Synastra Terminal Chat Prototype")
    print("=" * 50)

    i = 0
    user_prompt = prompts[i] if prompts else input("Welcome! I am Synastra the Natal Chart Chatbot, and I use the latest in astrological technology to help you with your relationship by analysing the sky! To start talking with me, I will need to create and prepare your natal star charts. Please provide the birth month, birth year, and the city and country each of you were born in. Once this is complete we can begin the session.\n")

    response, functions_called = completion(prompt=user_prompt)
    while response and response.choices[0].message.content != "Goodbye!":
        i += 1
        message = response.choices[0].message
        response_message = message.content or ""
        logger.info("ChatGPT: %s", response_message.strip())

        # Check if there's a follow-up question in the response
        if "QUESTION:" in response_message:
            question_line = [
                line.strip() for line in response_message.split("\n") if line.strip().startswith("QUESTION:")
            ][0]
            followup_question = question_line.replace("QUESTION:", "").strip() + " "
        else:
            followup_question = None

        if "get_courses" in functions_called:
            default_prompt = "Would you like to register for a course? "
        elif "register_course" in functions_called:
            default_prompt = "Can I help you with anything else? "
        else:
            default_prompt = "Please let me know: "

        user_prompt = prompts[i] if prompts and len(prompts) > i else input(followup_question or default_prompt)

        if user_prompt and user_prompt.lower().strip() in [
            "no",
            "no thanks",
            "nothing",
            "exit",
            "quit",
            "bye",
            "goodbye",
            "that's all",
            "nothing else",
        ]:
            print("Thank you for using Synastra! Goodbye!")
            break

        response, functions_called = completion(prompt=user_prompt)


if __name__ == "__main__":
    main()
