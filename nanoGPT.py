import torch
import torch.nn as nn
from torch.nn import functional as F

## hyperparameters
batch_size = 16
max_iters = 5000
block_size = 32
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ----------

torch.manual_seed(1337)

## load dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

## unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

## train and test splits
data = torch.tensor(encode(text), dtype=torch.int)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

## data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(k)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

## Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            ScaleDotProductAttention(head_size) for _ in range(num_head)
        )
        self.linear = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        out = self.dropout(self.linear(out))
        return out

class ScaleDotProductAttention(nn.Module):
    ## linear map can be either set at here or MHA
    def __init__(self, head_size):
        super().__init__()

        self.w_k = nn.Linear(n_embd, head_size)
        self.w_q = nn.Linear(n_embd, head_size)
        self.w_v = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        K = self.w_k(x)
        Q = self.w_q(x)
        V = self.w_v(x)

        Q_T = Q.transpose(1, 2)
        score = K@Q_T* C**(-0.5) # (B,T,T)
        score = score.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        out = self.dropout(F.softmax(score, dim=-1))@V
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-12
        
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = ((x - mu)/torch.sqrt(var+self.eps))*self.gamma + self.beta
        return x

class Block(nn.Module):
    ## decoder only
    def __init__(self, n_embd, num_head):
        super().__init__()
        head_size = n_embd//num_head
        self.MHA = MultiHeadAttention(num_head, head_size)
        self.layernorm1 = LayerNorm(n_embd)
        self.layernorm2 = LayerNorm(n_embd)
        self.out_head = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.MHA(self.layernorm1(x))
        x = x + self.out_head(self.layernorm2(x))
        return x

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd) ## why embedding, not RoPE
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head)
              for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets.long())

        return logits, loss

    def generate(self, idx, max_new_tokens):
        ## idx is (B,T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


## training loop
model = BigramLM()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M paramters')

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for iter in range(max_iters):

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))