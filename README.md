# Hemingway

A minimalist NLP project exploring how much of literary style can be learned by extremely small language models.

---

🧠 Core Idea

Modern language models use billions of parameters.

This project asks a simpler question:

«How much of "good writing" is just statistical structure?»

We train a tiny model on public-domain works by Ernest Hemingway and observe:

- what it learns
- what it fails to learn
- where "style" ends and "meaning" begins

---

⚙️ What’s inside

- Character-level language model (PyTorch)
- Training loop ("src/train.py")
- Sampling script ("src/sample.py")
- Public-domain Hemingway corpus ("data/hemingway.txt")

---

🚀 Quick Start

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Train

python src/train.py

Generate text

python src/sample.py --prompt "He looked at the river" --max-new-tokens 300

---

📚 Dataset

This project uses public-domain works by Ernest Hemingway sourced from Project Gutenberg, including:

- In Our Time (1925)
- The Sun Also Rises (1926)
- Men Without Women (1927)
- A Farewell to Arms (1929)
- Three Stories & Ten Poems

---

🧪 What to expect

This model will learn:

- short declarative sentences
- punctuation rhythm
- surface-level prose style

It will NOT learn:

- deep meaning
- long narrative structure
- true literary intent

---

🧭 Philosophy

This project treats language as compression.

If style can be compressed into a small set of weights, then:

«voice may be mostly statistical»

The remaining gap is where meaning lives.

---

🔜 Next steps

- Upgrade to Transformer architecture
- Add tokenization (BPE)
- Compare model sizes vs quality
- Build a demo

---

⚠️ Note

Use only public-domain text if you plan to share or distribute this project.