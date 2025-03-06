import time
from specdec.main import SpeculativeDecoder

if __name__ == "__main__":
    decoder = SpeculativeDecoder(
        approx_model_name="meta-llama/Llama-2-7b-hf",
        target_model_name="meta-llama/Llama-2-70b-hf",
        gamma=4, max_tokens=50, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    input_text = "How do neural networks learn?"
    
    start_time = time.time()
    output = decoder.speculative_sample(input_text)
    end_time = time.time()
    
    print("Generated Text:", output)
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
