from transformers import AutoTokenizer

def analyze_input(text):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokens = tokenizer.tokenize(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token Count: {len(tokens)}")
