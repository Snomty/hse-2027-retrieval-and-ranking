"""LM Dirichlet по объединённому Title+Body, подбор mu.
В реестр добавляется только лучший конфиг.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import (
    load_queries, load_qrels, load_docs_lemma, get_lm_stats,
    lm_score_query, lemmatize_text, evaluate_run, append_result, show_registry,
    SEED,
)


BASE_PARAMS = dict(mu=2000.0)
GRID_MU = [100.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0]


def cv_score(lm_stats, doc_id_to_idx, train_queries, train_qrels, params, n_folds=5):
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
            scores = lm_score_query(q_terms, idx, lm_stats, **params)
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
    lm_stats = get_lm_stats(docs_lemma, max_body_lemmas=300)
    doc_id_to_idx = {d: i for i, d in enumerate(docs_lemma.DocumentId)}
    print(f"loaded in {time.time()-t0:.1f}s, vocab={len(lm_stats['term_to_id'])}, "
          f"avg_doc_len={lm_stats['avg_doc_len']:.2f}, total_terms={lm_stats['total_terms']:.0f}")

    best_params = dict(BASE_PARAMS)
    best_mean = None
    all_runs = []

    print("\n=== stage: mu ===")
    for v in GRID_MU:
        params = {**best_params, "mu": v}
        mean, std, _ = cv_score(lm_stats, doc_id_to_idx, train_queries, train_qrels, params)
        all_runs.append({**params, "cv_mean": mean, "cv_std": std})
        marker = ""
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_params["mu"] = v
            marker = "  <- best"
        print(f"  mu={v:<6}  cv_mean={mean:.4f}  std={std:.4f}{marker}")
    print(f"  -> best mu={best_params['mu']}, cv_mean={best_mean:.4f}")

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
            scores = lm_score_query(q_terms, idx, lm_stats, **best_params)
            for did, s in zip(sub.DocumentId.values, scores):
                results.append((qid, did, float(s)))
        return pd.DataFrame(results, columns=["QueryId", "DocumentId", "Score"])

    full_run = score_fn_full(train_queries, train_qrels)
    full_ndcg = evaluate_run(full_run, train_qrels, k=10)
    final_mean, final_std, _ = cv_score(lm_stats, doc_id_to_idx, train_queries, train_qrels, best_params)
    print(f"full-train nDCG@10 = {full_ndcg:.4f}")
    print(f"CV mean = {final_mean:.4f}, std = {final_std:.4f}")
    print(f"best params: {best_params}")
    print(f"total elapsed: {time.time()-t0:.1f}s")

    runs_df = pd.DataFrame(all_runs)
    print("\nall runs:")
    print(runs_df[["mu", "cv_mean", "cv_std"]].to_string(index=False))

    append_result(
        "lm_dirichlet2",
        {**best_params, "lemmatization": True, "max_body_lemmas": 300,
         "fields": "Title + Body (combined)",
         "p_c": "(tf_corpus + 0.5) / (total + 0.5V)"},
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
    