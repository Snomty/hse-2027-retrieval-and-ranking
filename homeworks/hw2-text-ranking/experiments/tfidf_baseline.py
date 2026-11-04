import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    load_docs, load_queries, load_qrels,
    cross_validate, append_result, show_registry,
)


def main():
    t0 = time.time()
    docs = load_docs()
    train_queries = load_queries()
    train_qrels = load_qrels()

    docs["Text"] = docs.Title.fillna("") + " " + docs.Body.fillna("")
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=2, max_df=0.95,
    )
    doc_matrix = vectorizer.fit_transform(docs.Text)
    doc_id_to_idx = {d: i for i, d in enumerate(docs.DocumentId)}
    print(f"docs={doc_matrix.shape}, elapsed={time.time()-t0:.1f}s")

    def score_fn(queries_df, pairs_df):
        qids = pairs_df.QueryId.unique()
        q_texts = queries_df.set_index("QueryId").loc[qids, "QueryText"]
        q_matrix = vectorizer.transform(q_texts.values)
        pairs_by_qid = {qid: sub for qid, sub in pairs_df.groupby("QueryId")}
        results = []
        for qid, qvec in zip(q_texts.index, q_matrix):
            sub = pairs_by_qid.get(qid)
            if sub is None:
                continue
            idx = [doc_id_to_idx[d] for d in sub.DocumentId if d in doc_id_to_idx]
            if not idx:
                continue
            sims = cosine_similarity(qvec, doc_matrix[idx]).ravel()
            for did, s in zip(sub.DocumentId.values, sims):
                results.append((qid, did, float(s)))
        return pd.DataFrame(results, columns=["QueryId", "DocumentId", "Score"])

    print("cross-validating...")
    full_ndcg, fold_scores = cross_validate(score_fn, train_queries, train_qrels)
    print(f"\nfull-train nDCG@10 = {full_ndcg:.4f}")
    print(f"CV nDCG@10: mean = {fold_scores.mean():.4f}, std = {fold_scores.std():.4f}")

    append_result("tfidf_baseline", {
        "fields": "Title + ' ' + Body",
        "sublinear_tf": True, "min_df": 2, "max_df": 0.95,
        "lemmatization": False,
    }, full_ndcg, fold_scores)
    show_registry()


if __name__ == "__main__":
    main()
    