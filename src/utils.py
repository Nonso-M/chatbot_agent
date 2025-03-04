import os
import copy
from datetime import datetime
from typing import Dict, List
from astrapy import DataAPIClient
from dotenv import load_dotenv
from openai import OpenAI
import ast
import re

from apiclient.discovery import build


load_dotenv(".env")
BASE_DIR = "."

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


def extract_hashtags(text):
    """
    Extracts all hashtags from the given text and returns them as a list.

    Args:
        text (str): The input text containing hashtags.

    Returns:
        list: A list of hashtags found in the text.
    """
    pattern = r"#\w+"
    tags = re.findall(pattern, text)
    return tags


def youtube_duration_to_seconds(duration):
    """
    Converts a YouTube duration string into seconds.

    Args:
        duration (str): A duration string in the format 'PT#H#M#S'.

    Returns:
        int: The total duration in seconds.
    """
    pattern = re.compile(
        r"PT"  # Starts with 'PT'
        r"(?:(\d+)H)?"  # Optional hours
        r"(?:(\d+)M)?"  # Optional minutes
        r"(?:(\d+)S)?"  # Optional seconds
    )

    match = pattern.fullmatch(duration)
    if not match:
        return 0

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def get_collection(collection_name: str):
    """Creates a connection to the atral DB connection either to query
    or dump data

    Parameters
    ----------
    collection_name : str
        The nae of the collection(table) name
    """
    client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
    database = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
    print(f"* Database: {database.info().name}\n")

    collection = database.get_collection(collection_name)
    print(f"* Collection: {collection.info().name}\n")

    return collection


def get_source_data(similiarity_search: List[Dict]):

    text_list = []
    for i, a in enumerate(similiarity_search):
        print(i)
        text = f"Video with the title {a['metadata']['snippet']['title']} is a {a['metadata']['video_type']} video by {a['metadata']['snippet']['channelTitle']} \
        published on {a['metadata']['snippet']['publishedAt']}. The video has {a['metadata']['statistics']['viewCount']} views, {a['metadata']['statistics']['likeCount']} \
        likes, and {a['metadata']['statistics']['commentCount']} comments. Video Duration: {a['metadata']['duration']}. \
            Description: {a['metadata']['snippet']['description']}. Video Link {a['metadata']['share_url']} \n------------------------------------------\n"

        text_list.append(text)

    source_knowledge = " ".join(text_list)

    return source_knowledge


def augment_prompt(results: str):
    # get top 3 results from knowledge base
    # get the text from the results
    source_knowledge = get_source_data(results)
    # feed into an augmented prompt

    augmented_prompt = f"""
                    Below are a number of real, relevant YouTube videos retrieved from embedded data.

                    These videos cover a variety of topics and may contain content in multiple languages. If necessary, translate key points into English and summarize the main insights.

                    Each video includes metadata such as title, description, engagement metrics (views, likes, comments), and publication date.

                    Here are the video details:

                    {source_knowledge}
                    --------------------------

                    Using the YouTube videos provided above, analyze their content to answer the following question. Where applicable, include direct verbatim quotes from the video titles, descriptions, or comments.

                    Add the link to these videos in your response (If no video details is returned, tell the user that his query has  no similiarity in the database)

                    Additionally, extract and suggest key themes and keywords and hashtags that can be used for future video content creation, ensuring alignment with trending topics and audience engagement.


                    Answer:

                    """

    return augmented_prompt


def parse_youtube_data(data_list, parse_fields):
    """Parses out neccesary field from the tiktok api result
    Args:
        data_list (List[List[Dict]]): API result straight from the tiktok apify API
        parse_fields (Dict[str,List]): FIelds that will be parsed out of the API result

    Returns:
        List[Dict]: _description_
    """

    data_unpacked = copy.deepcopy(data_list["items"])

    final_list = []
    for i, dat in enumerate(data_unpacked):
        print(i)
        final_dict = {}

        for field in parse_fields.keys():
            if parse_fields[field]:
                final_dict[field] = {}
                for nested_field in parse_fields[field]:
                    # Check if the keys exist before accessing them
                    if field in dat.keys() and nested_field in dat[field].keys():
                        final_dict[field][nested_field] = dat[field][nested_field]
                    else:
                        # Handle the case where the keys do not exist
                        final_dict[field][nested_field] = None

            else:
                # Check if the key exists before accessing it
                if field in dat.keys():
                    final_dict[field] = dat[field]
                else:
                    # Handle the case where the key does not exist
                    final_dict[field] = None

        final_list.append({"metadata": final_dict})
    for dat in final_list:
        dat["metadata"]["snippet"]["publishedAt"] = datetime.strptime(
            dat["metadata"]["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            time = youtube_duration_to_seconds(
                dat["metadata"]["contentDetails"]["duration"]
            )
        except:
            time = 0
        dat["metadata"]["duration"] = time

        try:
            dat["metadata"]["snippet"]["tags"] = extract_hashtags(
                dat["metadata"]["snippet"]["title"]
            )
            dat["metadata"]["tags-legnth"] = len(dat["metadata"]["snippet"]["tags"])

        except:
            dat["metadata"]["tags-legnth"] = 0

        dat["metadata"]["share_url"] = (
            "https://www.youtube.com/watch?v=" + dat["metadata"]["id"]
        )

        dat["metadata"]["video_type"] = (
            "promo"
            if time == 0
            else "short" if time <= 120 else "medium" if time < 360 else "long"
        )

    return final_list
