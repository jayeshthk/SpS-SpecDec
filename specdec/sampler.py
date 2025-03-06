import torch
import torch.nn.functional as F
def generate_speculative_tokens(approx_model, input_ids, gamma):
    with torch.no_grad():
        approx_output = approx_model.generate(input_ids, max_length=input_ids.shape[1] + gamma)
    return approx_output[:, -gamma:]

def validate_tokens(target_model, input_ids, speculative_tokens):
    target_logits = target_model(input_ids).logits[:, -1, :]
    target_probs = F.softmax(target_logits, dim=-1)
    accept_mask = torch.rand_like(target_probs) < target_probs
    return speculative_tokens[accept_mask]
