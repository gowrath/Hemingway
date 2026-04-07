import argparse
import torch

with open('data/hemingway.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])

block_size = 64
vocab_size = len(chars)

model = torch.nn.Sequential(
    torch.nn.Embedding(vocab_size, 64),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * block_size, vocab_size)
)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='He ')
parser.add_argument('--max-new-tokens', type=int, default=300)
args = parser.parse_args()

context = args.prompt[-block_size:]
context_ids = [stoi.get(ch, 0) for ch in context]
if len(context_ids) < block_size:
    context_ids = [0] * (block_size - len(context_ids)) + context_ids

out = list(context_ids)
for _ in range(args.max_new_tokens):
    x = torch.tensor([out[-block_size:]], dtype=torch.long)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1).item()
    out.append(idx_next)

print(decode(out).lstrip())
