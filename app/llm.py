import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# Fail fast instead of hanging a request worker; the openai client retries
# connection errors, timeouts, 429 and 5xx with exponential backoff.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=LLM_TIMEOUT,
    max_retries=LLM_MAX_RETRIES,
)


def generate(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "If the context does not contain enough information, say so. "
                    "Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": f"""
CONTEXT:
{context}

QUESTION:
{question}
"""
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content