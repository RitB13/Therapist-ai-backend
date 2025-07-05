from transformers import GenerationConfig

gen_config = GenerationConfig(
    temperature=0.7,        # Controls creativity
    top_k=40,               # Avoid extreme tokens
    top_p=0.9,              # Focus on likely responses
    repetition_penalty=1.1, # Prevent repeating "You're not alone"
    max_new_tokens=150,     # Length of response
    do_sample=True          # Sampling instead of greedy decoding
)
