# Retrieval benchmark

- **questions**: 36
- **corpus**: 12 documents from corpus
- **chunking**: recursive, ~128 tokens, 16 overlap
- **embedding**: fake:fake-embed
- **reranker**: fake:fake-rerank
- **vector store**: memory

| config | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | Hit@1 | p95 ms |
|---|---|---|---|---|---|---|---|---|
| **dense** | 0.514 | 0.639 | 0.639 | 0.764 | 0.603 | 0.634 | 0.528 | 18 |
| **sparse** | 0.764 | 0.931 | 0.958 | 0.972 | 0.882 | 0.894 | 0.833 | 24 |
| **hybrid** | 0.722 | 0.806 | 0.889 | 0.917 | 0.819 | 0.824 | 0.778 | 42 |
| **hybrid+rerank** | 0.611 | 0.833 | 0.889 | 0.958 | 0.769 | 0.802 | 0.667 | 58 |

## By question type (MRR / recall@5)

| type | dense | sparse | hybrid | hybrid+rerank |
|---|---|---|---|---|
| keyword | 0.89 / 0.93 | 1.00 / 1.00 | 1.00 / 1.00 | 0.95 / 1.00 |
| multi | 0.30 / 0.29 | 0.90 / 0.93 | 0.69 / 0.71 | 0.71 / 0.86 |
| paraphrase | 0.47 / 0.53 | 0.76 / 0.93 | 0.71 / 0.87 | 0.63 / 0.80 |

## Questions `sparse` did not answer in the top 5

- `p01` (paraphrase): How long do I have to ask for my money back?
