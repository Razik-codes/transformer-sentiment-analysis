# Transformer Architectures for Movie-Review Sentiment Classification

An educational deep-learning project that studies the transformer at three levels of abstraction: an encoder--decoder implementation built from first principles, a BERT-style encoder, and a RoBERTa-based classifier with a lightweight bottleneck adapter. The notebooks use binary sentiment classification on IMDB movie reviews as a concrete application.

This repository is presented as coursework and a learning artifact, not as a claim of novel research. It is intended to demonstrate practical familiarity with PyTorch, attention mechanisms, tokenisation, pretrained language models, and evaluation workflows.

## Project scope

| Notebook | Focus | Main components |
| --- | --- | --- |
| `transformer_final_proj_14004110.ipynb` | Transformer mechanics | Multi-head attention, positional encoding, masking, encoder--decoder stack |
| `Bert_final_proj_14004110.ipynb` | BERT-style encoder construction | Token/position embeddings, transformer encoder layers, IMDB-style CSV dataset interface |
| `adapter_final_proj_14004110.ipynb` | Transfer learning for sentiment analysis | `roberta-base`, a 64-dimensional adapter bottleneck, and classification metrics |

## Research questions

1. How do masking, positional representations, residual connections, and multi-head attention combine in a transformer?
2. How does a compact BERT-style encoder differ from a pretrained transformer used for downstream sentiment classification?
3. What is the practical role of an adapter bottleneck when adapting a pretrained RoBERTa model to IMDB sentiment analysis?

## Methods

The RoBERTa notebook downloads the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and reports accuracy, macro precision, macro recall, and macro F1 on its held-out test split. The BERT-style notebook expects `train_reviews.csv` and `test_reviews.csv`, each with `text` and `label` columns.

The notebooks are instructional implementations, so results should be reproduced on the target hardware before being cited or compared. In particular, model configuration, random seeds, trainable parameters, and hardware are not yet logged systematically; the repository deliberately does not report unverified benchmark numbers.

## Reproducibility

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks from the repository root. The adapter notebook downloads and extracts `aclImdb/` on first run; this dataset and downloaded model files are intentionally excluded from version control. A CUDA-capable PyTorch installation is optional but recommended for training.

For a fair experimental comparison, use a fixed seed, preserve the IMDB train/test split, record package and hardware versions, and evaluate only after model-selection decisions are complete.

## Technical notes and limitations

- The adapter notebook currently fine-tunes the RoBERTa backbone as well as training the adapter. Freezing the backbone would be required to make this a parameter-efficient-adaptation experiment.
- The BERT-style notebook is an architectural exercise rather than a reproduction of pretrained BERT. It should be extended with a dedicated classification head and held-out evaluation before using it for performance claims.
- Dataset licenses and the Hugging Face model terms apply to their respective assets; neither is redistributed here.

## References

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- Devlin et al. (2019), [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://aclanthology.org/N19-1423/).
- Liu et al. (2019), [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692).
- Maas et al. (2011), [*Learning Word Vectors for Sentiment Analysis*](https://aclanthology.org/P11-1015/).

## Repository hygiene

Large datasets, model checkpoints, local CSV files, notebook checkpoints, and environment files are excluded through `.gitignore`. Please do not commit downloaded datasets, access tokens, or generated model weights.
