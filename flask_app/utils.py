import requests
from bs4 import BeautifulSoup
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle

# Embedding model
model_name = "all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)

def search_articles(query):
    """
    Searches for articles related to the query using Serper API.
    """
    params = {
        "q": query,
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": '86cb3126078e6a54e7382695f09764d25e153980590b163ee96120b4dc16fbc3'
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results['organic_results']
    links = [result["link"] for result in organic_results[:6]]  # Top 6 links
    return links

def scrape_article_content(url):
    """
    Scrapes content from the given URL.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([p.get_text(strip=True) for p in paragraphs])
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def create_vector_store(article_contents):
    """
    Creates a vector store from article contents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_with_metadata = [
        Document(page_content=content, metadata={"source": url})
        for url, content in article_contents.items()
        for content in text_splitter.split_text(content)
    ]
    vector_store = FAISS.from_documents(docs_with_metadata, embedding)
    with open("index.pkl", "wb") as f:
        pickle.dump(vector_store, f)

def load_vector_store():
    """
    Loads the pre-built vector store.
    """
    with open("index.pkl", "rb") as f:
        return pickle.load(f)
