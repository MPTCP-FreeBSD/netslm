from transformers import AutoModelForCausalLM, AutoTokenizer

class Phi3Model:
    def __init__(self, model_path):
        """
        Initialize the Phi-3-mini-4k-instruct model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    def generate(self, inputs, max_length=128, **kwargs):
        """
        Generate text from the Phi-3 model.
        """
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
