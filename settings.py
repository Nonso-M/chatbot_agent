import os
from dotenv import load_dotenv


load_dotenv(".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

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
