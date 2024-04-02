import torch


class TokenSelection(object):
    def __init__(self, k) -> None:
        self.k = k
    
    def __call__(self, tokens: torch.Any) -> torch.Any:
        '''
        Arguments:
            tokens: (b, n, 256, 1536)
        Return:
            tokens: (b, n, k, 1536)
        '''
        b, n, l, c = tokens.shape
        norm_tokens = tokens / ((tokens ** 2).sum(-1, keepdim=True) ** 0.5)
        sim = torch.einsum("bnlc, bnmc -> bnlm", norm_tokens, norm_tokens.clone())
        sim = sim.softmax(-1)
        scores = sim.sum(-1)    # (b, n, 256)
        _, indices = torch.topk(scores, self.k, dim=-1, largest=False)  # (b, n, k)
        batch = torch.arange(b).reshape(b, 1, 1).repeat(1, n, self.k)
        obj = torch.arange(n).reshape(1, n, 1).repeat(b, 1, self.k)
        tokens = tokens[batch, obj, indices]
        return tokens

if __name__ == "__main__":
    tokens = torch.randn(4, 2, 256, 1536)
    ts = TokenSelection(24)
    print(ts(tokens).shape)       