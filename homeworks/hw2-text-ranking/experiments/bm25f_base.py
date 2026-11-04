import time
import numpy as np
import pandas as pd

from utils import (
    load_queries, load_qrels, load_docs_lemma, get_bm25f_stats,
    bm25f_score_query, lemmatize_text, cross_validate, append_result, show_registry,
)

PARAMS = dict(k1=1.5, b_title=0.5, b_body=0.75, w_title=3.0, w_body=1.0)


def main():
    t0 = time.time()
    train_queries = load_queries()
    train_qrels = load_qrels()
    docs_lemma = load_docs_lemma()
    stats = get_bm25f_stats(docs_lemma, max_body_lemmas=300)

    print(f"docs={len(stats['title_tf'])}, vocab={len(stats['term_to_id'])}, "
          f"avg_title={stats['avg_title_len']:.2f}, avg_body={stats['avg_body_len']:.2f}, "
          f"load={time.time()-t0:.1f}s")

    doc_id_to_idx = {d: i for i, d in enumerate(docs_lemma.DocumentId)}
    q_texts_map = train_queries.set_index("QueryId")["QueryText"].to_dict()
    q_lemma_cache = {}

    def score_fn(queries_df, pairs_df):
        pairs_by_qid = {qid: sub for qid, sub in pairs_df.groupby("QueryId")}
        results = []
        for qid, sub in pairs_by_qid.items():
            qtext = q_lemma_cache.get(qid)
            if qtext is None:
                qtext = lemmatize_text(q_texts_map[qid])
                q_lemma_cache[qid] = qtext
            q_terms = qtext.split()
            idx = [doc_id_to_idx[d] for d in sub.DocumentId if d in doc_id_to_idx]
            if not idx:
                continue
            scores = bm25f_score_query(q_terms, idx, stats, **PARAMS)
            for did, s in zip(sub.DocumentId.values, scores):
                results.append((qid, did, float(s)))
        return pd.DataFrame(results, columns=["QueryId", "DocumentId", "Score"])

    print("cross-validating...")
    full_ndcg, fold_scores = cross_validate(score_fn, train_queries, train_qrels)
    print(f"\nfull-train nDCG@10 = {full_ndcg:.4f}")
    print(f"CV nDCG@10: mean = {fold_scores.mean():.4f}, std = {fold_scores.std():.4f}")
    print(f"total elapsed: {time.time()-t0:.1f}s")

    append_result("bm25f_base", {**PARAMS, "lemmatization": True, "max_body_lemmas": 300},
                  full_ndcg, fold_scores)
    show_registry()


if __name__ == "__main__":
    main()
