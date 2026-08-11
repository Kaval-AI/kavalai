"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import html
import os
import secrets
from typing import Annotated, Optional

import feedparser
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from kavalai.functionkernel import pythontool
from pydantic import BaseModel

app = FastAPI(title="RSS-Parser", version="1.0.0")
security = HTTPBasic()


# These will be set from command line arguments
AUTH_USER = os.environ.get("RSS_AUTH_USER", "admin")
AUTH_PASSWORD = os.environ.get("RSS_AUTH_PASSWORD", "password")


def validate_auth(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    is_correct_username = secrets.compare_digest(credentials.username, AUTH_USER)
    is_correct_password = secrets.compare_digest(credentials.password, AUTH_PASSWORD)

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class RssFeedItem(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    summary: Optional[str] = None


class Feed(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    items: list[RssFeedItem] = []


@app.get("/get_rss_feed", response_model=Feed)
def get_rss_feed_endpoint(
    url: str,
    max_results: int = 5,
    username: str = Depends(validate_auth),
) -> Feed:
    """Fetches and parses an RSS or Atom feed from a given URL via FastAPI.

    Args:
        url: The valid URL of the RSS feed.
        max_results: Number of recent articles to return (default 5).
        username: Authenticated username.
    """
    try:
        return get_rss_feed(url=url, max_results=max_results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@pythontool
def get_rss_feed(
    url: str,
    max_results: int = 5,
) -> Feed:
    """Fetches and parses an RSS or Atom feed from a given URL.

    Args:
        url: The valid URL of the RSS feed.
        max_results: Number of recent articles to return (default 5).
    """
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return Feed()
        result = Feed(title=feed.feed.get("title"), url=url)
        for entry in feed.entries[:max_results]:
            title = entry.get("title", "No Title")
            link = entry.get("link", "No Link")
            summary = html.unescape(entry.get("summary", "No summary available."))

            result.items.append(RssFeedItem(title=title, link=link, summary=summary))
        return result
    except Exception as e:
        raise Exception(f"Error parsing feed: {str(e)}")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the RSS service command line."""
    parser = argparse.ArgumentParser(description="RSS-Parser Service")
    parser.add_argument(
        "--port", type=int, default=10000, help="Port to run the service on"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.environ.get("RSS_AUTH_USER", "admin"),
        help="Basic auth username",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("RSS_AUTH_PASSWORD", "password"),
        help="Basic auth password",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the RSS service with credentials from the command line."""
    global AUTH_USER, AUTH_PASSWORD

    args = parse_args(argv)
    AUTH_USER = args.user
    AUTH_PASSWORD = args.password

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
