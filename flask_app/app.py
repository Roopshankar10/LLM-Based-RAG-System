from flask import Flask, request, jsonify
from utils import search_articles, fetch_article_content, create_vecotrs, load_vector_store
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain

app = Flask(__name__)

# Load vector store and LLM
llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

@app.route('/query', methods=['POST'])
def query():
    """
    Handles the POST request to '/query'.
    Extracts the query from the request, retrieves top relevant chunks,
    and generates an answer using the LLM.
    """
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    
    links=search_articles(user_query)
    content=fetch_article_content(links)
    create_vecotrs(content)
    vector_store = load_vector_store()

    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    results = retriever.get_relevant_documents(user_query)
    
    # Select top-k relevant chunks
    top_k = 5  # Number of top chunks to use
    ranked_chunks = results[:top_k]

    # Concatenate selected chunks
    concatenated_content = ""
    token_limit = 1024  # LLM token limit
    for doc in ranked_chunks:
        chunk_tokens = len(doc.page_content.split())
        if len(concatenated_content.split()) + chunk_tokens <= token_limit:
            concatenated_content += doc.page_content + "\n"
        else:
            break

    # Create final prompt
    final_prompt = f"Query: {user_query}\nContext: {concatenated_content}"

    # Generate answer
    try:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        answer = chain.invoke({"question": user_query, "context": final_prompt})
        return jsonify({"answer": answer["answer"], "sources": answer["sources"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)
