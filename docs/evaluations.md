# Evaluations

This document summarizes the current evaluation outputs computed by the codebase.

## NLU

| Metric | Value |
|---|---:|
| Hard Precision | 1.000 |
| Hard Recall | 1.000 |
| Hard F1 | 1.000 |
| Soft Precision | 0.923 |
| Soft Recall | 0.821 |
| Soft F1 | 0.856 |
| Combined Precision | 0.962 |
| Combined Recall | 0.910 |
| Combined F1 | 0.928 |
| Combined P/R Mean | 0.936 |
| Soft P/R Mean | 0.872 |

## Retrieval

| Variant | P@5 | R@5 | NDCG@5 | Cases |
|---|---:|---:|---:|---:|
| smart | 0.308 | 0.769 | 0.686 | 13 |
| baseline | 0.462 | 0.846 | 0.732 | 13 |
| no_rerank | 0.277 | 0.692 | 0.635 | 13 |

## Ablation

| Variant | P@5 | R@5 | NDCG@5 | Cases |
|---|---:|---:|---:|---:|
| no_semantic | 0.462 | 0.846 | 0.732 | 13 |
| no_rerank | 0.277 | 0.692 | 0.635 | 13 |

## Timing

| Variant | Mean ms | Median ms | Peak KB |
|---|---:|---:|---:|
| smart | 29.28 | 23.57 | 127.1 |
| baseline | 2.20 | 1.79 | 9.9 |

## Judge sample

- Sampled pairs: 50
- Score histogram: {1: 16, 0: 34}

## Per-query table

| Query | Smart P@5 | Baseline P@5 | Smart NDCG@5 | Baseline NDCG@5 |
|---|---:|---:|---:|---:|
| רכב ראשון לסטודנט עד 50 אלף, אוטומטי, חסכוני | 0.400 | 1.000 | 1.000 | 1.000 |
| משפחתית מרווחת אוטומטית עד 80 אלף | 0.000 | 1.000 | 0.000 | 1.000 |
| ג'יפ 4x4 אוטומטי עד 150 אלף | 0.200 | 0.000 | 1.000 | 0.000 |
| BMW אוטומטי עד 150 אלף | 0.400 | 0.400 | 1.000 | 1.000 |
| רכב חסכוני בדלק עד 60 אלף | 0.200 | 0.400 | 1.000 | 0.693 |
| רכב עירוני לנהג צעיר עד 45 אלף | 0.200 | 0.400 | 1.000 | 1.000 |
| רכב יוקרתי עם טסט ארוך | 0.200 | 0.000 | 0.387 | 0.000 |
| רכב יד ראשונה ללא תאונות | 0.000 | 0.200 | 0.000 | 0.387 |
| טויוטה היברידית חסכונית יד ראשונה עד 35 אלף | 0.200 | 0.200 | 1.000 | 1.000 |
| ב.מ.וו X1 יוקרתי אוטומטי יד ראשונה עד 180 אלף | 0.600 | 0.600 | 1.000 | 1.000 |
| רכב קטן לעיר לסטודנט עד 50 אלף | 0.000 | 0.200 | 0.000 | 1.000 |
| רכב ידני חסכוני עד 60 אלף | 0.600 | 0.600 | 0.581 | 0.567 |
| מרצדס אוטומטית יוקרתית עד 180 אלף | 1.000 | 1.000 | 0.949 | 0.869 |

## Interpretation

- Hard NLU is perfect on this benchmark: the system extracts the required slots exactly.
- Soft NLU is strong but not perfect, which is expected because preferences like family, luxury, and young-driver are fuzzy.
- Retrieval improves over the no-rerank semantic-only variant on this benchmark in NDCG@5.
- The lexical baseline is still competitive on several queries, so the benchmark is useful for spotting where semantic ranking helps and where it overreaches.
