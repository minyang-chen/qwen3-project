"""A Qwen3-Agent demo implementation"""

import os
import asyncio
from typing import Optional
from qwen_agent.agents import Assistant
from agent_ui import AgentWebUI
import json

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")


def init_agent_service(
    model_name: str = "qwen3:1.7b",
    model_server: str = "http://localhost:11434/v1",
    model_api_key: str = "EMPTY",
    thinking: bool = True,
):
    ## best practices
    ## -------------------------------------------------------------------------
    # For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0
    # For non-thinking mode, use Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
    ## -------------------------------------------------------------------------
    llm_cfg = {}
    if thinking:
        print(f"***ENABLE thinking *** {thinking}")
        llm_cfg = {
            "model": model_name,
            "model_server": model_server,  # api_base
            "api_key": model_api_key,
            "seed": 123,
            "temperature": 0.6,
            "repetition_penalty": 1.05,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 32768,
            "min_p": 0,
            "generate_cfg": {
                # Add: When the content is `<think>this is the thought</think>this is the answer`
                # Do not add: When the response has been separated by reasoning_content and content
                # This parameter will affect the parsing strategy of tool call
                "thought_in_content": True,
                # When using OAI API, pass the parameter of whether to enable thinking mode in this way
                "extra_body": {"enable_thinking": True},
            },
        }
    else:
        print(f"***DISABLE thinking *** {thinking}")
        llm_cfg = {
            "model": model_name,
            "model_server": model_server,  # api_base
            "api_key": model_api_key,
            "seed": 123,
            "temperature": 0.7,
            "repetition_penalty": 1.05,
            "top_p": 0.8,
            "top_k": 20,
            "max_new_tokens": 32768,
            "min_p": 0,
            "generate_cfg": {
                # Add: When the content is `<think>this is the thought</think>this is the answer`
                # Do not add: When the response has been separated by reasoning_content and content
                # This parameter will affect the parsing strategy of tool call
                "thought_in_content": False,
                # When using OAI API, pass the parameter of whether to enable thinking mode in this way
                "extra_body": {"enable_thinking": False},
            },
        }

    system = "You act as a smart AI assistant, and you have access to tools to answer to user query."
    # tools = [load_json_file("mcp.json")]
    tools = [
        {
            "mcpServers": {
                "time": {
                    "command": "uvx",
                    "args": ["mcp-server-time", "--local-timezone=America/Toronto"],
                },
                "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
                "sqlite": {
                    "command": "uvx",
                    "args": ["mcp-server-sqlite", "--db-path", "test.db"],
                },
            }
        }
    ]
    bot = Assistant(
        llm=llm_cfg,
        name="Qwen3-Agent",
        description="AI Assistant",
        system_message=system,
        function_list=tools,
    )
    return bot


def agent_gui():
    # set active generation method
    enable_thinking = True
    # Define the agent
    bot = init_agent_service(thinking=enable_thinking)

    chatbot_config = {
        "user.name": "Qwen3 Agent",
        "agent.name": "Qwen3 Agent",
        "input.placeholder": "enter your query here",
        "prompt.suggestions": [
            "https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen",
            "how many tables in the database",
            "Create a student table with the student's name and age.",
            "Add a student named jack smith, who is 16 years old this year.",
        ],
    }
    ui = AgentWebUI(
        bot,
        chatbot_config=chatbot_config,
        think_mode=enable_thinking,
    )

    ui.run(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    agent_gui()
