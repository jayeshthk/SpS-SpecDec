import torch
from specdec.main import SpeculativeDecoder

if __name__ == "__main__":
    decoder = SpeculativeDecoder(
        approx_model_name="meta-llama/Llama-2-7b-hf",
        target_model_name="meta-llama/Llama-2-70b-hf",
        gamma=4, max_tokens=20, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    input_text = "Tell me a fun fact about space."
    output = decoder.speculative_sample(input_text)
    print("Generated Text:", output)
