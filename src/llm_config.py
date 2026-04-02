from langchain_openai import ChatOpenAI

def get_llm():
    """
    Initializes the ChatOpenAI client pointing to local llama-server.
    The local llama.cpp server is expected to run on http://127.0.0.1:8080/v1
    """
    return ChatOpenAI(
        openai_api_base="http://127.0.0.1:8080/v1",
        max_retries=1,
        timeout=1200.00,
        openai_api_key="not-needed", # Local API does not require real key
        streaming=True,
        temperature=0.1
    )
