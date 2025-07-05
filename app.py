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

# ✅ Load model once
llm_pipeline = load_llm()
chat_history = few_shot_examples.strip()

# ✅ Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# ✅ Define request schema
class ChatInput(BaseModel):
    user_input: str

# ✅ Define POST endpoint
@app.post("/chat")
async def chat(input_data: ChatInput):
    global chat_history

    user_input = input_data.user_input

    # ✅ Initialize retriever here (on-demand)
    retriever = get_retriever()

    # ✅ Retrieve relevant knowledge
    relevant_docs = retriever.invoke(user_input)
    retrieved_knowledge = "\n".join([doc.page_content for doc in relevant_docs])
    rag_context = f"Helpful therapeutic information:\n{retrieved_knowledge}"

    # ✅ Construct prompt
    full_input = f"{rag_context}\n\n{chat_history}\nUser: {user_input}\nTherapist:"

    # ✅ LLM Response
    response = llm_pipeline.invoke(full_input)

    # ✅ Clean response if needed
    if "Therapist:" in response:
        response = response.split("Therapist:")[-1].strip()

    # ✅ Update chat history
    chat_history += f"\nUser: {user_input}\nTherapist: {response}"

    # ✅ Log history to file
    with open("logs/chat_history.txt", "a") as f:
        f.write(f"\nUser: {user_input}\nTherapist: {response}\n")

    return {"response": response}
