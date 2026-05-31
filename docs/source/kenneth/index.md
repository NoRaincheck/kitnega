---
title: Kenneth — Kneser-Ney Language Model
---

# Kenneth

Interpolated Kneser-Ney language model with pure-Python training,
NLTK-compatible scoring, and optional PyTorch/ONNX export.

## Usage

```python
from kenneth.model import KneserNeyModel, CharacterKneserNeyModel

# Word-level model
model = KneserNeyModel(order=3)
model.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
model.score("cat", ("the",))       # log prob given context
model.perplexity(["the", "cat", "sat"])

# Character-level model
cmodel = CharacterKneserNeyModel(order=4)
cmodel.fit(["hello", "world"])
cmodel.perplexity("hello")
```

## API

| Method                   | Description                                   |
| ------------------------ | --------------------------------------------- |
| `fit(sentences, vocab?)` | Train on tokenised (or raw) sentences         |
| `score(word, context?)`  | Log probability of word given preceding words |
| `perplexity(texts)`      | Perplexity of a sentence or corpus            |
| `get_counts()`           | Raw n-gram counts per order                   |
| `get_vocab()`            | Known vocabulary list                         |

### PyTorch / ONNX Export

```python
from kenneth.export import to_torch, to_onnx

model = KneserNeyModel(order=3)
model.fit(training_sentences)

# PyTorch module
torch_module = to_torch(model)

# ONNX export
to_onnx(model, "model.onnx")
```

See `kenneth/kenneth/export.py` for details. Requires `torch` and `onnx` dev
dependencies.
