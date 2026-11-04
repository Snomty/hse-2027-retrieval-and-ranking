"""Жадный подбор параметров BM25F: w_title -> k1 -> (опционально) b_body.
В реестр добавляется только лучший конфиг по CV mean.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import (
    load_queries, load_qrels, load_docs_lemma, get_bm25f_stats,
    bm25f_score_query, lemmatize_text, evaluate_run, append_result, show_registry,
    SEED,
)


BASE_PARAMS = dict(k1=0.9, b_title=0.5, b_body=0.75, w_title=8.0, w_body=1.0)

GRID_W_TITLE = [8.0, 12.0, 16.0, 20.0]
GRID_K1      = [0.3, 0.5, 0.7, 0.9]
GRID_B_BODY  = [0.5, 0.75, 0.9]

TUNE_B_BODY = False


def cv_score(stats, doc_id_to_idx, train_queries, train_qrels, params, n_folds=5):
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
            scores = bm25f_score_query(q_terms, idx, stats, **params)
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


def run_stage(name, grid, key, best_params, best_mean, stats, doc_id_to_idx,
              train_queries, train_qrels, all_runs):
    print(f"\n=== stage: {key} ===")
    for v in grid:
        params = {**best_params, key: v}
        mean, std, _ = cv_score(stats, doc_id_to_idx, train_queries, train_qrels, params)
        all_runs.append({**params, "cv_mean": mean, "cv_std": std, "stage": key})
        marker = ""
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_params[key] = v
            marker = "  <- best"
        print(f"  {key}={v:<5}  cv_mean={mean:.4f}  std={std:.4f}{marker}")
    print(f"  -> best {key}={best_params[key]}, cv_mean={best_mean:.4f}")
    return best_params, best_mean


def main():
    t0 = time.time()
    train_queries = load_queries()
    train_qrels = load_qrels()
    docs_lemma = load_docs_lemma()
    stats = get_bm25f_stats(docs_lemma, max_body_lemmas=300)
    doc_id_to_idx = {d: i for i, d in enumerate(docs_lemma.DocumentId)}
    print(f"loaded in {time.time()-t0:.1f}s, vocab={len(stats['term_to_id'])}")

    best_params = dict(BASE_PARAMS)
    best_mean = None
    all_runs = []

    best_params, best_mean = run_stage(
        "w_title", GRID_W_TITLE, "w_title", best_params, best_mean,
        stats, doc_id_to_idx, train_queries, train_qrels, all_runs,
    )
    best_params, best_mean = run_stage(
        "k1", GRID_K1, "k1", best_params, best_mean,
        stats, doc_id_to_idx, train_queries, train_qrels, all_runs,
    )
    if TUNE_B_BODY:
        best_params, best_mean = run_stage(
            "b_body", GRID_B_BODY, "b_body", best_params, best_mean,
            stats, doc_id_to_idx, train_queries, train_qrels, all_runs,
        )

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
            scores = bm25f_score_query(q_terms, idx, stats, **best_params)
            for did, s in zip(sub.DocumentId.values, scores):
                results.append((qid, did, float(s)))
        return pd.DataFrame(results, columns=["QueryId", "DocumentId", "Score"])

    full_run = score_fn_full(train_queries, train_qrels)
    full_ndcg = evaluate_run(full_run, train_qrels, k=10)
    final_mean, final_std, _ = cv_score(stats, doc_id_to_idx, train_queries, train_qrels, best_params)
    print(f"full-train nDCG@10 = {full_ndcg:.4f}")
    print(f"CV mean = {final_mean:.4f}, std = {final_std:.4f}")
    print(f"best params: {best_params}")
    print(f"total elapsed: {time.time()-t0:.1f}s")

    runs_df = pd.DataFrame(all_runs)
    cols = ["stage", "k1", "b_title", "b_body", "w_title", "cv_mean", "cv_std"]
    print("\nall runs:")
    print(runs_df[cols].to_string(index=False))

    append_result(
        "bm25f_tuned2",
        {**best_params, "lemmatization": True, "max_body_lemmas": 300},
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
    