import argparse
import torch
from specdec.main import SpeculativeDecoder
from specdec.utils import log_message, benchmark

def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding with approximate and target models.")
    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?", help='Input text for generation')
    parser.add_argument('--approx_model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='Name of the approximate model')
    parser.add_argument('--target_model_name', type=str, default="meta-llama/Llama-2-70b-hf", help='Name of the target model')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='Number of speculative guesses per step')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='Maximum number of tokens to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on (cpu/cuda)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Show benchmark results')
    
    args = parser.parse_args()
    
    if args.verbose:
        log_message(f"Initializing models: {args.approx_model_name} (approx) → {args.target_model_name} (target)")
    
    decoder = SpeculativeDecoder(
        approx_model_name=args.approx_model_name,
        target_model_name=args.target_model_name,
        gamma=args.gamma,
        max_tokens=args.max_tokens,
        device=args.device
    )
    
    if args.benchmark:
        output = benchmark(decoder.speculative_sample, args.input)
    else:
        output = decoder.speculative_sample(args.input)
    
    print("Generated Text:", output)

if __name__ == "__main__":
    main()
