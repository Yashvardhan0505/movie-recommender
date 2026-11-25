# Results

MovieLens 1M: 1,000,209 ratings from 6,040 users on 3,706 movies (2000-04-25 to 2003-02-28). Test period from 2000-12-29: 1,180 returning users (on average 81 movies each to predict) and 29 new users.

## Returning users (top 10)

| Method | Recall@10 | NDCG@10 | Hit rate@10 | Catalog coverage |
|---|---:|---:|---:|---:|
| Most popular (baseline) | 0.0394 | 0.2540 | 0.671 | 3.8% |
| Most popular, last 30 days | 0.0497 | 0.2734 | 0.727 | 3.3% |
| Item-to-item | 0.0402 | 0.2564 | 0.653 | 7.8% |
| Matrix factorisation (SVD) | 0.0443 | 0.2425 | 0.675 | 23.8% |
| EASE | 0.0494 | 0.2559 | 0.695 | 24.2% |
| Two-stage: retrieve + rerank | 0.0637 | 0.2900 | 0.777 | 21.0% |

## Candidate retrieval (how many of the next movies made the shortlist)

| Source | Candidates per user | Recall |
|---|---:|---:|
| Most popular (baseline) | 50 | 0.143 |
| Most popular, last 30 days | 50 | 0.161 |
| Item-to-item | 100 | 0.210 |
| Matrix factorisation (SVD) | 100 | 0.239 |
| EASE | 100 | 0.250 |
| All five combined | 187 | 0.388 |

## New users with no history (top 10)

| Method | Recall@10 | NDCG@10 | Hit rate@10 |
|---|---:|---:|---:|
| Most popular (baseline) | 0.0577 | 0.5458 | 0.966 |
| Most popular, last 30 days | 0.0596 | 0.5567 | 0.966 |
| Popular with similar people | 0.0635 | 0.5835 | 0.966 |

## What the reranker relies on (permutation importance)

- Popular with people of the same age and gender (`segment_share`): 0.1005
- Movie: years since release (`item_age`): 0.0881
- Movie: average rating (`item_avg_rating`): 0.0379
- Movie: times rated (`item_log_pop`): 0.0218
- Rank in SVD (`svd_rank`): 0.0129
- User: days since first rating (`user_days_active`): 0.0109
- EASE score (`ease_score`): 0.0106
- SVD score (`svd_score`): 0.0062
- User: occupation (`occupation`): 0.0056
- Genre match with the user's history (`genre_affinity`): 0.0045
- Item-to-item score (`itemcf_score`): 0.0036
- Number of models proposing it (`n_sources`): 0.0034
