import torch

with open('data/hemingway.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]

block_size = 64
batch_size = 32

def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

model = torch.nn.Sequential(
    torch.nn.Embedding(len(chars), 64),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * block_size, len(chars))
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(500):
    xb, yb = get_batch()
    logits = model(xb)
    loss = torch.nn.functional.cross_entropy(logits, yb[:, -1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

torch.save(model.state_dict(), "model.pt")
