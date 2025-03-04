import os
from dotenv import load_dotenv
from ast import Dict
from functools import partial
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
from src.source import get_youtube_query, get_youtube_videos
from src.utils import get_collection, get_source_data, parse_youtube_data
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from settings import OPENAI_API_KEY, YOUTUBE_API_KEY

collection = get_collection("yt_video_data")
mongodb_filters = {}

# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]

DEFAULT_YOUTUBE_FIELDS = {
    "id": [],
    "snippet": [
        "channelTitle",
        "title",
        "description",
        "channelId",
        "defaultAudioLanguage",
        "defaultLanguage",
        "publishedAt",
        "tags",
    ],
    "contentDetails": [
        "definition",
    ],
    "statistics": ["commentCount", "favoriteCount", "likeCount", "viewCount"],
    "duration": [],
    "video_type": [],
    "tags-legnth": [],
    "share_url": [],
    "source": [],
}


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)


@tool("rag_retrieval")
def get_rag_output(query: str):
    """Extracts data from the youtube vector database a natural language query. Compares the
    the query against data stored in the database. Useful for extrating content
    that would be used by the final parser to generate the llm output."""

    # global mongodb_filters
    # global collection
    collection = globals()["collection"]
    mongodb_filters = globals()["mongodb_filters"]

    result = collection.find(
        filter=mongodb_filters,
        sort={"$vectorize": query},
        include_similarity=True,
    ).limit(5)
    # search html for abstract
    parsed_data = get_source_data(result)
    # return abstract text
    return parsed_data


@tool("youtube_retrieval")
def get_fresh_youtube_data(history: str) -> List[Dict]:
    """This function auguments the rag_retrieval function. It is used when the results from
    the rag_retrieval don't match the query passed in to the rag_retrieval function"""
    # youtube_query = get_youtube_query(query)  # returns a list of one element ideally

    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
    )
    formatted_data = "ab"
    return formatted_data


# @tool("final_answer")
# def final_answer(introduction: str, research_steps: str, sources: str):
#     """Returns a natural language response to the user. Note! if the user asks a question
#      that doesn't need a rag_retrieval or youtube retrival, return a normal response.
#      If the user ask a question that needs tome form of information retrieval, the response should be
#      be formatted having the following sections
#     report. There are several sections to this report, those are:
#     - `Videos Content`: Using the YouTube videos provided above, analyze their content to answer the
#         following question. Where applicable, include direct verbatim quotes from the video titles, descriptions, or comments.
#         Add the link to these videos in your response (If no video details is returned, tell the user that his query has  no similiarity in the database)
#     - `Themes and Keywords`: Extract and suggest key themes and keywords and hashtags that can be used for future video content creation,
#       ensuring alignment with trending topics and audience engagement.
#     """
#     if type(research_steps) is list:
#         research_steps = "\n".join([f"- {r}" for r in research_steps])
#     if type(sources) is list:
#         sources = "\n".join([f"- {s}" for s in sources])
#     return ""


@tool("final_answer")
def final_answer(response):
    """Returns a natural language response to the user. Note! if the user asks a question
     that doesn't need a rag_retrieval or youtube retrival, return a normal response.
     If the user ask a question that needs tome form of information retrieval, the response should be
     be formatted having the following sections
    report. There are several sections to this report, those are:
    - `Videos Content`: Using the YouTube videos provided above, analyze their content to answer the
        following question. Where applicable, include direct verbatim quotes from the video titles, descriptions, or comments.
        Add the link to these videos in your response (If no video details is returned, tell the user that his query has  no similiarity in the database)
    - `Themes and Keywords`: Extract and suggest key themes and keywords and hashtags that can be used for future video content creation,
      ensuring alignment with trending topics and audience engagement.
    """
    return ""


def create_prompt():
    """This function abstracts the complexities of creating the promot template
    every single time"""
    system_prompt = """You are the oracle, the great AI decision maker.
                    Given the user's query you must decide what to do with it based on the
                    list of tools provided to you.

                    If you are are asked a question that does not need any information retrival 
                    from the vector database, just return a normal chat response

                    If you see that a tool has been used (in the scratchpad) with a particular
                    query, do NOT use that same tool with the same query again. Also, do NOT use
                    any tool more than twice (ie, if the tool appears in the scratchpad twice, do
                    not use it again).

                    You should aim to collect information from a diverse range of sources before
                    providing the answer to the user. Once you have collected plenty of information
                    to answer the user's question (stored in the scratchpad) use the final_answer
                    tool."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("assistant", "scratchpad: {scratchpad}"),
        ]
    )
    return prompt


def run_oracle(state: list, prompt):
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0,
    )
    tools = [get_rag_output, get_fresh_youtube_data, final_answer]
    oracle = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: create_scratchpad(
                intermediate_steps=x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind_tools(tools)
    )
    # print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    # print(out)
    try:
        tool_name = out.tool_calls[0]["name"]
        tool_args = out.tool_calls[0]["args"]
        action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")

    except IndexError:
        tool_name = "final_answer"
        tool_args = out.content
        action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")
    return {"intermediate_steps": [action_out]}


def router(state: list):
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"


def run_tool(state: list):
    # use this as helper function so we repeat less code
    tool_str_to_func = {
        "rag_retrieval": get_rag_output,
        "youtube_retrieval": get_fresh_youtube_data,
        "final_answer": final_answer,
    }
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    # print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {"intermediate_steps": [action_out]}


def generate_agent():
    """Genegrates an agent based on the tools"""

    graph = StateGraph(AgentState)

    prompt = create_prompt()
    oracle_runner = partial(run_oracle, prompt=prompt)

    graph.add_node("oracle", oracle_runner)
    graph.add_node("rag_retrieval", run_tool)
    graph.add_node("youtube_retrieval", run_tool)
    graph.add_node("final_answer", run_tool)

    graph.set_entry_point("oracle")

    graph.add_conditional_edges(
        source="oracle",  # where in graph to start
        path=router,  # function to determine which node is called
    )

    # create edges from each tool back to the oracle
    tools = [get_rag_output, get_fresh_youtube_data, final_answer]
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # if anything goes to final answer, it must then move to END
    graph.add_edge("final_answer", END)

    runnable = graph.compile()

    return runnable
