import streamlit as st
import requests

st.title("LLM-based RAG Search")

# Input for user query
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query.strip():
        st.write(f"Searching for: {query}")
        
        # Flask API endpoint
        flask_url = "http://localhost:5001/query"  # URL of the Flask app

        try:
            # Make a POST request to the Flask API
            response = requests.post(flask_url, json={"query": query})

            if response.status_code == 200:
                # Display the generated answer
                data = response.json()
                answer = data.get('answer', "No answer received.")
                sources = data.get('sources', [])

                st.subheader("Answer:")
                st.write(answer)

                # Display sources if available
                if sources:
                    st.subheader("Sources:")
                    for source in sources:
                        st.write(f"- {source}")
                else:
                    st.write("No sources available.")
            else:
                st.error(f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the Flask API: {e}")
    else:
        st.warning("Please enter a query.")
