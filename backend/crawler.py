import requests
import json
import time
from pymongo import MongoClient, UpdateOne
from urllib.parse import urljoin

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MONGO_URI       = "mongodb://localhost:27017/"
DB_NAME         = "rag_database"
COLLECTION_NAME = "little_articles"

mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[DB_NAME]
articles_col = db[COLLECTION_NAME]


GHOST_API_BASE = "https://dl-staging-website.ghost.io/ghost/api/content"
GHOST_API_KEY  = "a4b216e975091c63cc39c1ac98"
POSTS_PER_PAGE = 15
DEEPLEARNING_AI_BASE = "https://www.deeplearning.ai/the-batch/"


INITIAL_TAGS = [
    "large-language-models-llms"
]

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------
def fetch_posts_for_tag(tag_slug, page=1):
    """
    Fetches one page of posts for a given tag slug.
    """
    url = (
        f"{GHOST_API_BASE}/posts/"
        f"?key={GHOST_API_KEY}"
        f"&include=tags%2Cauthors"
        f"&filter=tag:{tag_slug}"
        f"&page={page}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    posts = data.get("posts", [])
    return posts


def transform_post_to_document(post_json):
    """
    Given one `post_json` dictionary from Ghost API, build a MongoDB document.
    We extract the fields we care about:
      - id (Ghost’s internal ID)
      - uuid
      - title
      - slug
      - html       (full body HTML)
      - feature_image
      - feature_image_alt       (if present)
      - excerpt
      - published_at
      - updated_at
      - reading_time
      - url        (canonical URL to the post)
      - tags       (list of {id, name, slug, url, …})
      - authors    (list of {id, name, slug, url, …})
      - primary_author (object)
      - primary_tag    (object)
    Then we return a dict suitable for insertion/upsert into MongoDB.
    """
    doc = {
        "_id":            post_json["id"],   # use Ghost’s numeric ID as our Mongo _id
        "uuid":           post_json.get("uuid"),
        "title":          post_json.get("title"),
        "slug":           post_json.get("slug"),
        "html":           post_json.get("html"),
        "feature_image":  post_json.get("feature_image"),
        "feature_image_alt": post_json.get("feature_image_alt"),
        "excerpt":        post_json.get("excerpt"),
        "published_at":   post_json.get("published_at"),
        "updated_at":     post_json.get("updated_at"),
        "reading_time":   post_json.get("reading_time"),
        "tags":           post_json.get("tags", []),
        "authors":        post_json.get("authors", []),
        "primary_author": post_json.get("primary_author"),
        "primary_tag":    post_json.get("primary_tag"),
        # record when we inserted/updated this document
        "scraped_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    url_parts = post_json.get("url", "").strip().split("/")
    if len(url_parts) > 0:
        doc["url"] = urljoin(DEEPLEARNING_AI_BASE, url_parts[-2])
    else:
        doc["url"] = DEEPLEARNING_AI_BASE
    return doc


def upsert_articles(docs):
    """
    Given a list of document dicts (each having a unique "_id"),
    perform a bulk upsert into MongoDB so we insert new or update existing.
    """
    if not docs:
        return
    operations = []
    for d in docs:
        operations.append(
            UpdateOne(
                {"_id": d["_id"]},
                {"$set": d},
                upsert=True
            )
        )
    result = articles_col.bulk_write(operations)
    print(f"  Upserted {result.upserted_count} new, modified {result.modified_count}")
    return result

def get_num_articles(tag_slug):
    """
    Given a tag slug, return the number of articles under that tag.
    """
    resp = requests.get(f'https://www.deeplearning.ai/_next/data/2HpDvYc1z_k6duU9sMQWm/the-batch/tag/{tag_slug}.json?slug={tag_slug}')
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        print(f"  Failed to decode JSON for tag '{tag_slug}': {resp.text}")
        return 0
    try:
        num_posts = data['pageProps']['tag']['count']['posts']
    except KeyError:
        print(f"  Failed to fetch tag '{tag_slug}': {data}")
        return 0
    return num_posts


# ------------------------------------------------------------
# MAIN CRAWL LOOP
# ------------------------------------------------------------
def crawl_all_tags(initial_tags):
    """
    Crawl Ghost posts by tag, starting from `initial_tags`.
    Maintains a queue of tag slugs to process, and a set of already‐processed tags.
    For each tag, pages through all posts, inserts/upserts each post to MongoDB,
    and enqueues any newly discovered tag slugs found in each post’s "tags" array.
    """
    tags_to_process = list(initial_tags)
    seen_tags       = set()

    while tags_to_process:
        current_tag = tags_to_process.pop(0)
        if current_tag in seen_tags:
            continue

        print(f"\n=== Crawling tag: '{current_tag}' ===")
        page = 1
        all_docs_for_this_tag = []

        total_articles = get_num_articles(current_tag)
        if total_articles == 0:
            print(f"  No articles found for tag '{current_tag}'.")
            seen_tags.add(current_tag)
            continue

        # Calculate the number of pages based on the total articles
        total_pages = (total_articles // POSTS_PER_PAGE) + (1 if total_articles % POSTS_PER_PAGE > 0 else 0)

        print(f"  Found {total_articles} posts under tag '{current_tag}', across {total_pages} pages.")

        while page <= total_pages:
            try:
                posts = fetch_posts_for_tag(current_tag, page)
            except requests.HTTPError as e:
                print(f"  Failed to fetch tag '{current_tag}' (page {page}): {e}")
                break

            print(f"  Processing page {page}/{total_pages}, {len(posts)} posts.")

            docs_this_page = []
            for post in posts:
                doc = transform_post_to_document(post)
                docs_this_page.append(doc)


                for tag_obj in post.get("tags", []):
                    slug = tag_obj.get("slug")
                    if slug and slug not in seen_tags and slug not in tags_to_process:
                        print(f"    Discovered new tag: '{slug}'")
                        tags_to_process.append(slug)

            upsert_articles(docs_this_page)

            page += 1
            # Don't hit the API too hard
            time.sleep(1)

        seen_tags.add(current_tag)
        print(f"=== Finished tag: '{current_tag}' ===")

    print("\nAll tags processed. Crawl complete.")


if __name__ == "__main__":
    crawl_all_tags(INITIAL_TAGS)
