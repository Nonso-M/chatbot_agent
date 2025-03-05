import warnings

import streamlit as st
from openai import OpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
from src.source import get_youtube_query, get_youtube_videos
from src.utils import augment_prompt, get_collection, parse_youtube_data
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ast import Dict
from functools import partial
from typing import TypedDict, Annotated, List, Union, Dict
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
from src.source import get_youtube_query, get_youtube_videos
from src.utils import get_collection, get_source_data, parse_youtube_data
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from settings import OPENAI_API_KEY, DEFAULT_YOUTUBE_FIELDS


warnings.filterwarnings("ignore")


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
def get_fresh_youtube_data(query: str) -> List[Dict]:
    """This function auguments the rag_retrieval function. It is used when the results from
    the rag_retrieval don't match the query passed in to the rag_retrieval function"""
    # youtube_query = get_youtube_query(query)  # returns a list of one element ideally

    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
    )
    youtube_data = get_youtube_videos(query)
    formatted_data = parse_youtube_data(
        youtube_data, parse_fields=DEFAULT_YOUTUBE_FIELDS
    )
    return formatted_data


@tool("final_answer")
def final_answer(response):
    """Returns a natural language response to the user. Note! if the user asks a question
     that doesn't need a rag_retrieval or youtube retrival, return a normal response.
     If the user ask a question that needs tome form of information retrieval, the response should be
     be formatted having the following sections
    report. There are several sections to this report, those are:
    - `video_content`: Using the YouTube videos provided above, analyze their content to answer the
        following question. Where applicable, include direct verbatim quotes from the video titles, descriptions, or comments.
        Give reasons why you believe this video would drive engagements
        Add the link to these videos in your response
    - `themes_keywords`: Extract and suggest key themes and keywords and hashtags that can be used for future video content creation,
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

    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {"intermediate_steps": [action_out]}


def build_report(output: Union[Dict, str]):

    if isinstance(output, str):
        return output

    else:
        print(output)
        output = f"""
                VIDEO CONTENT
                ------------
                {output["response"]["video_content"]}

                THEMES AND KEYWORDS
                ------------
                {output["response"]["themes_keywords"]}
                """

    return output


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


runnable = generate_agent()

collection = get_collection("yt_video_data")


messages = [
    SystemMessage(content="You are a helpful assistant that do not hallucinate."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
]
st.title("🐋 Chat Youtube Data 🐋")
st.subheader(
    "🤖 Youtube Video Idea Generator 🤖",
)

st.markdown(
    """
            Introducing the Luminr Prototype App—a cutting-edge tool for brands and influencers to discover content based on your unique query.
             Your input is compared against an extensive database of YouTube videos, delivering tailored suggestions and growth-driving keywords. 
            Give it a try by describing the type of content you’re looking for!
            """
)
st.divider()

# ======= Sidebar UI =========
st.sidebar.markdown("## Parameters")
st.sidebar.divider()
# temp = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.5)

# min_likes = st.sidebar.slider("Minimum Likes", 100, 10000, step=20)

# min_views = st.sidebar.slider("Minimum Views", 1000, 1000000, step=1000)

# min_comments = st.sidebar.slider("Minimum Comments", 0, 500, step=50)

# date_from = st.sidebar.date_input("Date From", value=None)
# st.write("Your birthday is:", date_from)

DEFAULT_MIN_LIKES = 0
DEFAULT_MIN_VIEWS = 0
DEFAULT_MIN_COMMENTS = 0
DEFAULT_DATE_FROM = None  # or date.today() if that’s preferred


def reset_defaults():
    # Reset session state values BEFORE widgets are created
    st.session_state["min_likes"] = DEFAULT_MIN_LIKES
    st.session_state["min_views"] = DEFAULT_MIN_VIEWS
    st.session_state["min_comments"] = DEFAULT_MIN_COMMENTS
    st.session_state["date_from"] = DEFAULT_DATE_FROM
    st.rerun()  # Rerun the app so widgets pick up the new values


# Place the reset button at the top so it runs before widget instantiation
if st.sidebar.button("Reset to Defaults"):
    reset_defaults()

# Initialize session state keys if they don’t exist yet
if "min_likes" not in st.session_state:
    st.session_state.min_likes = DEFAULT_MIN_LIKES
if "min_views" not in st.session_state:
    st.session_state.min_views = DEFAULT_MIN_VIEWS
if "min_comments" not in st.session_state:
    st.session_state.min_comments = DEFAULT_MIN_COMMENTS
if "date_from" not in st.session_state:
    st.session_state.date_from = DEFAULT_DATE_FROM

# Create sidebar sliders with session state values and unique keys
min_likes = st.sidebar.slider(
    "Minimum Likes",
    0,
    10000,
    step=100,
    value=st.session_state.min_likes,
    key="min_likes",
)
min_views = st.sidebar.slider(
    "Minimum Views",
    0,
    1000000,
    step=1000,
    value=st.session_state.min_views,
    key="min_views",
)
min_comments = st.sidebar.slider(
    "Minimum Comments",
    0,
    500,
    step=50,
    value=st.session_state.min_comments,
    key="min_comments",
)
date_from = st.sidebar.date_input(
    "Date From", value=st.session_state.date_from, key="date_from"
)


print(type(date_from))

filter_status = {
    "min_likes": {"value": min_likes, "type": "gte"},
    "min_views": {"value": min_views, "type": "gte"},
    "min_comments": {"value": min_comments, "type": "gte"},
    "date_from": {
        "value": date_from.strftime("%Y-%m") if date_from is not None else None,
        "type": "gte",
    },
}

map_dict = {
    "min_likes": "statistics.likeCount",
    "min_views": "statistics.viewCount",
    "min_comments": "statistics.commentCount",
    "date_from": "publishedAt",
}
print(filter_status)

# ===== API Client =========
client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# ====== Chat History =======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_chat_history_messages():

    print(st.session_state.chat_history)

    if len(st.session_state.chat_history) > 0:
        for message in st.session_state.chat_history:
            with st.chat_message(message.type):
                st.write(message.content)


render_chat_history_messages()


mongodb_filters = {
    f"metadata.{map_dict[field]}": {f'${spec["type"]}': spec["value"]}
    for field, spec in filter_status.items()
    if spec["value"] not in (0, None)
}

print(mongodb_filters)

if prompt := st.chat_input():
    try:
        # display the user's prompt/message
        with st.chat_message("user"):
            st.markdown(prompt)

        # display the llm message
        with st.chat_message("assistant"):
            # placeholder for the llm response
            placeholder = st.empty()

        inputs = {
            "input": prompt,
            "chat_history": st.session_state.chat_history,
            "intermediate_steps": [],
        }
        st.session_state.chat_history.append(HumanMessage(prompt))

        out = runnable.invoke(inputs)

        try:
            # Agent Tools were used during the generation of this
            final_result_interim = out["intermediate_steps"][-1].tool_input["response"]
            final_result = build_report(final_result_interim)

        except:
            # No tool was used during the Agent call. The output is parsed differenntly
            final_result_interim = out["intermediate_steps"][-1].tool_input
            final_result = build_report(final_result_interim)

        full_response = ""

        print(final_result)

        # placeholder.write(final_result)

        for word in final_result:
            full_response += word
            placeholder.write(full_response)

        st.session_state.chat_history.append(AIMessage(full_response))

    except Exception as e:
        print("ERROR: ", e)
