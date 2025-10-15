from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "I love Data Science."
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)   

print("Vector:", response.data[0].embedding)