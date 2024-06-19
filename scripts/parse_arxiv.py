"""Requirements
    - feedparser installed: pip install feedparser
"""

import argparse
import json
import logging
import os
import urllib
import urllib.request

import feedparser

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)s-%(funcName)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)


def get_bibtex(feed: dict):
    lines = ["@article{" + feed["id"].split("/")[-1]]
    for k, v in [
        ("author", " and ".join([a["name"] for a in feed["authors"]])),
        ("title", feed["title"]),
        # ("PrimaryClass", feed["category"]),
        # ("Abstract", feed["summary"]),
        ("year", str(feed["published_parsed"][0])),
        ("month", str(feed["published_parsed"][1])),
        ("note", feed["arxiv_comment"]),
        ("archiveprefix", "arXiv"),
        ("url", feed["link"]),
    ]:
        if len(v):
            lines.append("%-13s = {%s}" % (k, v))

    return ("," + os.linesep).join(lines) + os.linesep + "}"


def main(id, verbose=False):
    id = id.split("/")[-1]
    url = f"http://export.arxiv.org/api/query?search_query=all:{id}&start=0&max_results=1"

    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    feed = feedparser.parse(data)["entries"][0]

    json_feed = json.dumps(feed, indent=2)
    _logger.debug(json_feed)

    title = feed["title"].replace("\n", "")
    authors = ", ".join(author["name"] for author in feed["authors"])
    link = feed["link"]
    abstract = feed["summary"]
    bibtext = get_bibtex(feed)

    _logger.debug(f"{title}\nby {authors} ({link})")
    _logger.debug(bibtext)

    obj = {
        "title": title,
        "authors": authors,
        "link": link,
        "abstract": abstract,
        "bibtext": bibtext,
    }
    if verbose:
        print(json.dumps(obj, indent=2))
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--id", type=str, default="2203.07798")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args.id, args.verbose)
