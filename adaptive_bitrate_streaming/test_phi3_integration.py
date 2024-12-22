from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
model_path = "/Users/raja/Documents/GitHub/netslm/downloaded_plms/Phi-3-mini-4k-instruct"
tokenizer_path = "/Users/raja/Documents/GitHub/netslm/downloaded_plms/Phi-3-mini-4k-instruct/tokenizers/microsoft/Phi-3-mini-4k-instruct"

# Load tokenizer
print(f"Loading Phi-3 tokenizer from: {tokenizer_path}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# Ensure unique pad token
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    print("Added unique pad token:", tokenizer.pad_token)

# Load model
print(f"Loading Phi-3 model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Resize model embeddings to accommodate new pad token
model.resize_token_embeddings(len(tokenizer))

# Test input
text = "Hello, world! How can I assist you today?"
inputs = tokenizer(text, return_tensors="pt")

print("Tokenized input:", inputs)

# Generate output
outputs = model.generate(inputs.input_ids, max_length=50)
print("Generated output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
