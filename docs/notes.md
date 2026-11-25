# Notes

## 1.2: SciFact hybrid vs best single retriever
Hybrid nDCG@10 on SciFact (0.7095) is 0.003 below exact vectors (0.7128); recall@10 is higher (0.8243 vs 0.8212). On NFCorpus hybrid is above both (0.3598 vs 0.3463 vector, 0.3057 BM25).
Differences between the runs: the vector baseline is exact brute force, hybrid's vector side is approximate HNSW (ef=200); and RRF with k=60 gives BM25 (0.655 on SciFact, 0.057 below vectors) equal weight. Neither effect was isolated in a separate run, so the cause is not established. The gap is within what RRF with an unweighted weaker lexical list can produce; no tuning was done to close it.
