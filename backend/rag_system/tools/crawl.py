import requests
import json
import time
import argparse
from pymongo import MongoClient, UpdateOne
from urllib.parse import urljoin

# ------------------------------------------------------------
# CONFIGURATION (now via argparse)
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Crawl Ghost blog and upsert articles to MongoDB.")
    parser.add_argument('--mongo-uri', type=str, default="mongodb://localhost:27017/", help='MongoDB URI')
    parser.add_argument('--db-name', type=str, default="rag_database", help='MongoDB database name')
    parser.add_argument('--collection-name', type=str, default="articles", help='MongoDB collection name')
    parser.add_argument('--ghost-api-base', type=str, default="https://dl-staging-website.ghost.io/ghost/api/content", help='Ghost API base URL')
    parser.add_argument('--ghost-api-key', type=str, default="a4b216e975091c63cc39c1ac98", help='Ghost API key')
    parser.add_argument('--deeplearning-ai-base', type=str, default="https://www.deeplearning.ai/the-batch/", help='Base URL for deeplearning.ai')
    parser.add_argument('--posts-per-page', type=int, default=15, help='Posts per page')
    parser.add_argument('--initial-tags', type=str, nargs='+', default=["large-language-models-llms"], help='Initial tag slugs to crawl')
    parser.add_argument('--cookie', type=str, default="v_7Wdn7Y_3E7b3v5FzsK6", help='Read in README, how to get this cookie')
    return parser.parse_args()

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

def upsert_articles(docs, articles_col):
    """
    Given a list of document dicts (each having a unique "_id"),
    perform a bulk upsert into MongoDB so we insert new or update existing.
    Now: skip if _id exists in DB (do not update).
    """
    if not docs:
        return
    operations = []
    skipped = 0
    for d in docs:
        if articles_col.count_documents({"_id": d["_id"]}, limit=1):
            skipped += 1
            continue
        operations.append(
            UpdateOne(
                {"_id": d["_id"]},
                {"$set": d},
                upsert=True
            )
        )
    if operations:
        result = articles_col.bulk_write(operations)
        print(f"  Upserted {result.upserted_count} new, modified {result.modified_count}, skipped {skipped}")
        return result
    else:
        print(f"  Skipped {skipped} (already in DB)")
        return None

def crawl_all_tags(initial_tags, articles_col, ghost_api_base, ghost_api_key, deeplearning_ai_base, posts_per_page, cookie):
    """
    Crawl Ghost posts by tag, starting from `initial_tags`.
    Maintains a queue of tag slugs to process, and a set of already‐processed tags.
    For each tag, pages through all posts, inserts/upserts each post to MongoDB,
    and enqueues any newly discovered tag slugs found in each post’s "tags" array.
    """
    def fetch_posts_for_tag(tag_slug, page=1):
        """
        Fetches one page of posts for a given tag slug.
        """
        url = (
            f"{ghost_api_base}/posts/"
            f"?key={ghost_api_key}"
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
        """
        doc = {
            "_id":            post_json["id"],
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
            "scraped_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        url_parts = post_json.get("url", "").strip().split("/")
        if len(url_parts) > 0:
            doc["url"] = urljoin(deeplearning_ai_base, url_parts[-2])
        else:
            doc["url"] = deeplearning_ai_base
        return doc

    def get_num_articles(tag_slug):
        """
        Given a tag slug, return the number of articles under that tag.
        """
        resp = requests.get(f'https://www.deeplearning.ai/_next/data/{cookie}/the-batch/tag/{tag_slug}.json?slug={tag_slug}')
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

        total_pages = (total_articles // posts_per_page) + (1 if total_articles % posts_per_page > 0 else 0)

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

            upsert_articles(docs_this_page, articles_col)

            page += 1
            time.sleep(0.3)  # Avoid hitting API too hard

        seen_tags.add(current_tag)
        print(f"=== Finished tag: '{current_tag}' ===")

    print("\nAll tags processed. Crawl complete.")

if __name__ == "__main__":
    args = parse_args()
    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client[args.db_name]
    articles_col = db[args.collection_name]
    crawl_all_tags(
        initial_tags=args.initial_tags,
        articles_col=articles_col,
        ghost_api_base=args.ghost_api_base,
        ghost_api_key=args.ghost_api_key,
        deeplearning_ai_base=args.deeplearning_ai_base,
        posts_per_page=args.posts_per_page,
        cookie=args.cookie
    )
