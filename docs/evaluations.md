# Evaluations

This document summarizes the current benchmark after expanding the test set and tuning the `Smart` ranker to be more hard-constraint aware.

## Benchmark Setup

- Number of evaluation queries: 19
- Coverage:
  - exact constraints such as price, year, gear, and owners
  - location-heavy queries such as Haifa, Nahariya, Kiryat, Tel Aviv, and Modi'in
  - soft intents such as family, luxury, city driving, and first-owner preference
  - slangy and ambiguous user intent

## NLU

| Metric | Value |
|---|---:|
| Hard Precision | 0.947 |
| Hard Recall | 0.947 |
| Hard F1 | 0.947 |
| Soft Precision | 0.947 |
| Soft Recall | 0.789 |
| Soft F1 | 0.840 |
| Combined Precision | 0.947 |
| Combined Recall | 0.868 |
| Combined F1 | 0.894 |
| Combined P/R Mean | 0.908 |
| Soft P/R Mean | 0.868 |

## Retrieval @5

| Variant | P@5 | R@5 | NDCG@5 | Cases |
|---|---:|---:|---:|---:|
| Smart | 0.411 | 0.842 | 0.648 | 19 |
| Baseline | 0.347 | 0.737 | 0.564 | 19 |
| No rerank | 0.253 | 0.684 | 0.603 | 19 |

## Retrieval Across k

| Variant | P@10 | R@10 | NDCG@10 | P@15 | R@15 | NDCG@15 | P@20 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Smart | 0.200 | 0.769 | 0.665 | 0.164 | 0.846 | 0.669 | 0.154 | 1.000 | 0.697 |
| Baseline | 0.277 | 0.846 | 0.714 | 0.215 | 0.846 | 0.693 | 0.177 | 0.846 | 0.691 |
| Auto | 0.231 | 0.769 | 0.656 | 0.179 | 0.769 | 0.638 | 0.158 | 0.923 | 0.665 |

## Auto Routing

| Metric | Value |
|---|---:|
| Auto P@5 | 0.368 |
| Auto R@5 | 0.789 |
| Auto NDCG@5 | 0.549 |
| Baseline chosen | 9 cases |
| Smart chosen | 10 cases |

### Auto Across k

| k | P@k | R@k | NDCG@k |
|---|---:|---:|---:|
| 5 | 0.368 | 0.789 | 0.549 |
  | 10 | 0.263 | 0.895 | 0.603 |
  | 15 | 0.214 | 0.947 | 0.618 |
  | 20 | 0.174 | 0.947 | 0.620 |

### What Auto Means

`Auto` is the routing layer of the system, not a third independent ranker. It chooses between the hard-constraint `Baseline` and the more flexible `Smart` path depending on the query.

- When the query is strict and well-structured, Auto tends to choose `Baseline`.
- When the query is more ambiguous or preference-heavy, Auto can choose `Smart`.
- This is why Auto is best evaluated as a control policy that balances quality and robustness, not only as a single ranking score.

## Timing

| Variant | Mean ms | Median ms | Peak KB |
|---|---:|---:|---:|
  | Smart | 184.25 | 146.70 | 126.6 |
  | Baseline | 52.96 | 9.09 | 8.6 |
  | Auto | 164.42 | 129.80 | 86.6 |

## Interpretation

- The `Baseline` pipeline remains the strongest point of reference on the benchmark, especially in latency and memory.
- The tuned `Smart` pipeline now performs better on the retrieval benchmark at `@5`, which means the additional semantic and hard-fit weighting is helping.
- `Auto` is not meant to beat `Baseline` in every aggregate metric. Its value is that it routes each query to the more appropriate path, which makes the overall system more robust and more realistic for mixed user intent.
- The benchmark is now more balanced because it includes location-heavy and slangy queries, so the reported numbers are less biased toward clean exact-match cases.
- `Auto` should be read as the system's decision layer: it is allowed to pick the best available retrieval path per query, rather than being judged only as a standalone ranker.

## Practical Conclusion

For the final project narrative, it is most accurate to describe the system as:

- `Baseline`: the strong, reliable hard-constraint engine
- `Smart`: the semantic enhancement layer that helps with ambiguous or preference-heavy queries
- `Auto`: the routing layer that chooses between them dynamically
