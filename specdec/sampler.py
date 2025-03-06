import torch
import torch.nn.functional as F
def generate_speculative_tokens(approx_model, input_ids, gamma):
    with torch.no_grad():
        approx_output = approx_model.generate(input_ids, max_length=input_ids.shape[1] + gamma)
    return approx_output[:, -gamma:]


def validate_tokens(target_model, input_ids, speculative_tokens):
   
    num_speculative = speculative_tokens.shape[1]  

    target_logits = target_model(input_ids).logits[:, -num_speculative:, :]  # Shape: [1, num_speculative, vocab_size]
    target_probs = F.softmax(target_logits, dim=-1)  # Convert logits to probabilities
    speculative_probs = torch.gather(target_probs, 2, speculative_tokens.unsqueeze(-1)).squeeze(-1)  # Shape: [1, num_speculative]

    rand_probs = torch.rand_like(speculative_probs)

    accept_mask = rand_probs < speculative_probs  # Shape: [1, num_speculative]

    accepted_tokens = speculative_tokens[accept_mask]

    if accepted_tokens.numel() == 0:
        next_token = torch.multinomial(target_probs[:, -1, :], 1)  # Pick top token per position
        accepted_tokens = next_token.squeeze(0)  # Shape: [num_speculative]

    return accepted_tokens