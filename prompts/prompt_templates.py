from langchain.prompts import PromptTemplate
from prompts.personality import therapist_role

base_prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=f"""{therapist_role}

Conversation history:
{{chat_history}}

User: {{user_input}}
Therapist:"""
)
