from utils.device_utils import initialize_device

from tools import sam2, object_detection, clip_tool

import os
from PIL import Image
import numpy as np

from torch import device
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate


import asyncio

async def stream_tokens(agent_executor):
    async for event in agent_executor.astream_events(
        {"messages": 
         [
             HumanMessage(content="Write the criteria for things that you must see for a scene that follow the prompt 'a party with a few guests.'"),
             HumanMessage(content="Now, use image_path='examples/layout.jpg' for the tools you have. Create unit tests for each criteria in the form of a function that takes an image and returns a boolean value. Ensure that you strictly use the image path 'examples/layout.jpg'"),
             ]
        }, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")

                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

def stream_messages(agent_executor, prompt):
    for chunk in agent_executor.stream(
        {"messages": 
         [
             HumanMessage(content=prompt),
            #  HumanMessage(content="Write the criteria for things that you must see for a scene that follow the prompt 'a party with a few guests.'"),
            #  HumanMessage(content="Now, use image_path='examples/layout.jpg' for the tools you have. Create unit tests for each criteria in the form of a function that takes an image and returns a boolean value."),
            #  HumanMessage(content="Execute the tests on the image 'examples/layout.jpg' and return the results. Tell me if the image meets the criteria."),
             ]
        }
    ):
        print(chunk)
        print("----")

async def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


async def main():
    initialize_device()
    segmentation = StructuredTool.from_function(func=sam2, parse_docstring=True)
    object_detect = StructuredTool.from_function(func=object_detection, parse_docstring=True)
    semantic_filtering = StructuredTool.from_function(func=clip_tool, parse_docstring=True)

    tools = [segmentation, object_detect, semantic_filtering]
    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    You have access to the following variables:
    - image_path: 'examples/rendering_traj_000.png'

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question. The final answer should include the criteria and whether each was satisfied or not, along with justification for each.


    Begin!

    Question: "Write and evaluate the criteria for things that you must see for a 3D scene that follow the prompt: {input}. 
        When coming up with the criteria, think about the following: 
        (1) Object Arrangements: Are the objects placed in a way that aligns with the prompt?
        (2) Object Scaling: Are the scale of each object relative to each other? Are the scene appropriate?
        (3) Overall Aesthetics: Does the scene look visually appealing?        
        "
    Thought:{agent_scratchpad}

    '''
#     Thought:{agent_scratchpad}

    prompt = PromptTemplate.from_template(template)



    model = ChatOllama(model='llama3.2', temperature=3.2)
    input = prompt.format(input="'a party with a few guests.'", tools=tools, tool_names=[tool.name for tool in tools], agent_scratchpad="")
    print(input)
    agent_executor = create_react_agent(model=model, tools=tools, state_modifier=input)
    inputs = {"messages": [("user", "'Follow exactly what the system prompt provides."), ("user", "Evaluate the image with the path `image_path` according to these criteria. Return a summary of what the scene satisfies and what it does not satisfy.")]}

    await print_stream(agent_executor.stream(inputs, stream_mode="values"))

    # stream_messages(agent_executor, )
    
    


if __name__ == "__main__":
    asyncio.run(main())
    # main()