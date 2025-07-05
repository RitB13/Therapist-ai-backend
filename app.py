from fastapi import FastAPI
from pydantic import BaseModel
from llm_config import load_llm
from retriever_config import get_retriever
from prompts.few_shot_examples import few_shot_examples
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ✅ Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:5173"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and retriever once
llm_pipeline = load_llm()
# retriever = get_retriever()
chat_history = few_shot_examples.strip()

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Request schema
class ChatInput(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(input_data: ChatInput):
    global chat_history

    user_input = input_data.user_input

    # RAG Retrieval
    relevant_docs = retriever.invoke(user_input)  # ✅ Replaces deprecated get_relevant_documents
    retrieved_knowledge = "\n".join([doc.page_content for doc in relevant_docs])
    rag_context = f"Helpful therapeutic information:\n{retrieved_knowledge}"

    # Final Prompt Construction
    full_input = f"{rag_context}\n\n{chat_history}\nUser: {user_input}\nTherapist:"

    # LLM Response (Groq returns plain string)
    response = llm_pipeline.invoke(full_input)

    # Optional: Clean up response if it repeats the prefix
    if "Therapist:" in response:
        response = response.split("Therapist:")[-1].strip()

    # Update History
    chat_history += f"\nUser: {user_input}\nTherapist: {response}"

    with open("logs/chat_history.txt", "a") as f:
        f.write(f"\nUser: {user_input}\nTherapist: {response}\n")

    return {"response": response}
