from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, device='cuda'):
    return AutoModelForCausalLM.from_pretrained(model_name).to(device)

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
