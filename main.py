from langchain_groq import ChatGroq


llm=ChatGroq(groq_api_key="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",model_name="Llama-3.2-90b-text-preview")

if __name__=='__main__':
    response=llm.invoke("suggest some indian dishes")
    print(response)