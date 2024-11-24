from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from qdrant.search_in_collection import get_search_result

template = """You are a knowledgeable assistant specializing in human anatomy and physiology.
Your role is to provide accurate, concise, and easy-to-understand explanations about body parts and their functions."""
human_template = """
Question: {question}

Content: {content}

Answer the question using Content
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
model = OllamaLLM(model="qwen2.5:14b", temperature=0.9)

chain = prompt | model

question = "Which part protects the heart and lungs?"
search_result = get_search_result(question)


print(chain.invoke({"question": question, "content": search_result}))