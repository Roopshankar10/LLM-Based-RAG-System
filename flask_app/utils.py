import os
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import time
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import  RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import langchain
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Embedding model
model_name = "all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)

def search_articles(query):
    """
    Searches for articles related to the query using Serper API.
    Returns a list of dictionaries containing article URLs, headings, and text.
    """
    serp_api=os.getenv('serp_api_key')
    params = {
    "q": query,
    "location": "Austin, Texas, United States",
    "hl": "en",
    "gl": "us",
    "google_domain": "google.com",
    "api_key": serp_api
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results['organic_results']
    links=[]

    for i in organic_results:
        # print(i)
        links.append(i["link"])
    top_links=links[:6]

    return top_links

def fetch_article_content(top_links):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
}
    def scrape_p_tags(url):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Check for request errors

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all <p> tags
            p_tags = soup.find_all("p")
            return [p.get_text(strip=True) for p in p_tags]  # Get clean text from each <p> tag
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
    all_p_tags = {}
    # Iterate over URLs and scrape <p> tags
    for link in top_links:
        all_p_tags[link] = scrape_p_tags(link)
        time.sleep(1) 
    res=""

    # Print or process the scraped <p> tags
    for url, p_texts in all_p_tags.items():
        # print(f"\n<p> tags from {url}:")
        for p_text in p_texts:
            if p_text:
                
                res=res+'\n'+p_text
    return all_p_tags

def create_vecotrs(all_p_tags):
    text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
    docs_with_metadata = [
    Document(page_content=doc, metadata={"source": url})
    for url, p_texts in all_p_tags.items()
    for doc in text_splitter.split_text("\n".join(p_texts))
]
    

    vector_store = FAISS.from_documents(docs_with_metadata, embedding)
    with open("index.pkl",'wb') as f:
        pickle.dump(vector_store,f)


def load_vector_store():
    """
    Loads the pre-built vector store.
    """
    with open("index.pkl", "rb") as f:
        return pickle.load(f)
