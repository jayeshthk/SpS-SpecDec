import torch
import torch.nn.functional as F
from .sampler import generate_speculative_tokens, validate_tokens
from .models import load_model, load_tokenizer

class SpeculativeDecoder:
    def __init__(self, approx_model_name, target_model_name, gamma=4, max_tokens=20, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load models
        self.approx_model = load_model(model_name=approx_model_name, device=self.device)
        self.target_model = load_model(model_name=target_model_name, device=self.device)
        self.tokenizer = load_tokenizer(model_name=target_model_name)
        
        self.gamma = gamma
        self.max_tokens = max_tokens

    def speculative_sample(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        output_ids = input_ids.clone()
        
        for _ in range(self.max_tokens):
            speculative_tokens = generate_speculative_tokens(self.approx_model, output_ids, self.gamma)
            accepted_tokens = validate_tokens(self.target_model, output_ids, speculative_tokens)
            
            if len(accepted_tokens) == 0:
                break  # No tokens accepted, stop generation

            output_ids = torch.cat([output_ids, accepted_tokens.unsqueeze(0)], dim=-1)
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)