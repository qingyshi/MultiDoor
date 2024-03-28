import torch


class TokenSelection(object):
    def __init__(self, k) -> None:
        self.k = k
    
    def __call__(self, tokens: torch.Any) -> torch.Any:
        '''
        Arguments:
            tokens: (b, 256, 1536)
        Return:
            tokens: (b, k, 1536)
        '''
        b, n, c = tokens.shape
        norm_tokens = tokens / ((tokens ** 2).sum(-1, keepdim=True) ** 0.5)
        sim = torch.einsum("bnc, bmc -> bnm", norm_tokens, norm_tokens.clone())
        sim = sim.softmax(-1)
        scores = sim.sum(-1)
        _, indices = torch.topk(scores, self.k, dim=-1, largest=False)  # (b, k)
        batch = torch.arange(b).unsqueeze(-1).repeat(1, self.k)
        tokens = tokens[batch, indices]
        return tokens

tokens = torch.randn(4, 256, 1536)
ts = TokenSelection(32)
print(ts(tokens).shape)       