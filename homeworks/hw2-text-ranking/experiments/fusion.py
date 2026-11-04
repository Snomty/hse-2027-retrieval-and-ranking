"""
Fusion BM25F + LM: нормализация min-max по запросу, подбор beta.
В реестр добавляется только лучший конфиг.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import (
    load_queries, load_qrels, load_docs_lemma,
    get_bm25f_stats, get_lm_stats,
    bm25f_score_query, lm_score_query, lemmatize_text,
    evaluate_run, append_result, show_registry, SEED,
)


BM25F_PARAMS = dict(k1=0.7, b_title=0.5, b_body=0.75, w_title=12.0, w_body=1.0)
LM_PARAMS = dict(mu=3000.0)
GRID_BETA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fused_score_query(q_terms, doc_indices, bm25f_stats, lm_stats, beta):
    s_bm = bm25f_score_query(q_terms, doc_indices, bm25f_stats, **BM25F_PARAMS)
    s_lm = lm_score_query(q_terms, doc_indices, lm_stats, **LM_PARAMS)
    return beta * minmax_norm(s_bm) + (1.0 - beta) * minmax_norm(s_lm)


def cv_score(bm25f_stats, lm_stats, doc_id_to_idx, train_queries, train_qrels, beta, n_folds=5):
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
            scores = fused_score_query(q_terms, idx, bm25f_stats, lm_stats, beta)
            for did, s in zip(sub.DocumentId.values, scores):
                results.append((qid, did, float(s)))
        return pd.DataFrame(results, columns=["QueryId", "DocumentId", "Score"])

    unique_qids = train_qrels.QueryId.unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores = []
    for _, val_idx in kf.split(unique_qids):
        val_qids = unique_qids[val_idx]
        val_pairs = train_qrels[train_qrels.QueryId.isin(val_qids)]
        run_val = score_fn(train_queries, val_pairs)
        fold_scores.append(evaluate_run(run_val, train_qrels, k=10))
    fold_scores = np.array(fold_scores)
    return float(fold_scores.mean()), float(fold_scores.std()), fold_scores


def main():
    t0 = time.time()
    train_queries = load_queries()
    train_qrels = load_qrels()
    docs_lemma = load_docs_lemma()
    bm25f_stats = get_bm25f_stats(docs_lemma, max_body_lemmas=300)
    lm_stats = get_lm_stats(docs_lemma, max_body_lemmas=300)
    doc_id_to_idx = {d: i for i, d in enumerate(docs_lemma.DocumentId)}
    print(f"loaded in {time.time()-t0:.1f}s")

    best_beta = None
    best_mean = None
    all_runs = []

    print("\n=== stage: beta ===")
    for beta in GRID_BETA:
        mean, std, _ = cv_score(bm25f_stats, lm_stats, doc_id_to_idx,
                                train_queries, train_qrels, beta)
        all_runs.append({"beta": beta, "cv_mean": mean, "cv_std": std})
        marker = ""
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_beta = beta
            marker = "  <- best"
        print(f"  beta={beta:<4}  cv_mean={mean:.4f}  std={std:.4f}{marker}")
    print(f"  -> best beta={best_beta}, cv_mean={best_mean:.4f}")

    print("\n=== final ===")
    q_texts_map = train_queries.set_index("QueryId")["QueryText"].to_dict()
    q_lemma_cache = {}

    def score_fn_full(queries_df, pairs_df):
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
            scores = fused_score_query(q_terms, idx, bm25f_stats, lm_stats, best_beta)
            for did, s in zip(sub.DocumentId.values, scores):
                results.append((qid, did, float(s)))
        return pd.DataFrame(results, columns=["QueryId", "DocumentId", "Score"])

    full_run = score_fn_full(train_queries, train_qrels)
    full_ndcg = evaluate_run(full_run, train_qrels, k=10)
    final_mean, final_std, _ = cv_score(bm25f_stats, lm_stats, doc_id_to_idx,
                                        train_queries, train_qrels, best_beta)
    print(f"full-train nDCG@10 = {full_ndcg:.4f}")
    print(f"CV mean = {final_mean:.4f}, std = {final_std:.4f}")
    print(f"best beta = {best_beta}")
    print(f"total elapsed: {time.time()-t0:.1f}s")

    runs_df = pd.DataFrame(all_runs)
    print("\nall runs:")
    print(runs_df.to_string(index=False))

    append_result(
        "fusion_bm25f_lm",
        {"bm25f": BM25F_PARAMS, "lm": LM_PARAMS, "beta": best_beta,
         "norm": "minmax_per_query", "lemmatization": True, "max_body_lemmas": 300},
        full_ndcg,
        np.array([final_mean]),
    )
    reg = pd.read_pickle("results/registry.pkl")
    reg.loc[reg.index[-1], "cv_mean"] = final_mean
    reg.loc[reg.index[-1], "cv_std"] = final_std
    reg.to_pickle("results/registry.pkl")

    show_registry()


if __name__ == "__main__":
    main()
