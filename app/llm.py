import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
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