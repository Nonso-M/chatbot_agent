import os
from dotenv import load_dotenv
from apiclient.discovery import build
from openai import OpenAI
from typing import List
import ast
from settings import OPENAI_API_KEY, YOUTUBE_API_KEY


def get_youtube_query(user_search: str) -> List:
    """Takes a user prompt from the app and generate the right youtube query to
    generate better results

    Parameters
    ----------
    user_search : str
        prompt from the gpt app

    Returns
    -------
    List
        A list of search query (single element)
    """

    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    completion = client.chat.completions.create(
        # model="chatgpt-4o-latest",
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Only return what is asked of you, don't add any extra response. Always return a list with square brackets",
            },
            {
                "role": "user",  # here are examples
                "content": f"Change the following text to a list (1 element) of  youtube compatible search queries\
            text: {user_search} ",
            },
        ],
    )

    print(completion.choices[0].message.content)

    list_str = completion.choices[0].message.content

    try:
        list_parsed = ast.literal_eval(list_str)

    except Exception as e:
        print(e)
        return None

    return list_parsed


def get_youtube_videos(search_query: str):
    """Sends request to youtube api and gets video results of a search query

    Parameters
    ----------
    search_query : str
        Search query sent to the API

    Returns
    -------
    _type_
        A result from the youtube API
    """

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    channel = youtube.search().list(
        part="snippet", q=search_query, maxResults=5, type="video"
    )
    a = channel.execute()

    ids = [x["id"]["videoId"] for x in a["items"]]
    id_str = ",".join(ids)
    default_params = {"part": "snippet,contentDetails,statistics,topicDetails,status"}
    default_params["id"] = id_str
    process = youtube.videos().list(**default_params)

    video_data = process.execute()

    return video_data
