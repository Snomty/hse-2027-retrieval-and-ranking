import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pymorphy3
from tqdm.auto import tqdm


MAX_BODY_LEMMAS = 300
PARAMS = dict(k1=0.7, b_title=0.5, b_body=0.75, w_title=12.0, w_body=1.0)

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_morph = pymorphy3.MorphAnalyzer()
_lemma_cache = {}


def lemmatize_text(text):
    if not isinstance(text, str) or not text:
        return ""
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        lem = _lemma_cache.get(tok)
        if lem is None:
            lem = _morph.parse(tok)[0].normal_form
            _lemma_cache[tok] = lem
        out.append(lem)
    return " ".join(out)


def load_docs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            rows.append(tuple(parts[:4]))
    return pd.DataFrame(rows, columns=["DocumentId", "URL", "Title", "Body"])


def load_queries(path):
    return pd.read_csv(path, sep="\t", header=None,
                       names=["QueryId", "QueryText"], encoding="utf-8")


def build_stats(docs, max_body_lemmas=MAX_BODY_LEMMAS):
    n_docs = len(docs)
    term_to_id = {}

    def get_tid(term):
        tid = term_to_id.get(term)
        if tid is None:
            tid = len(term_to_id)
            term_to_id[term] = tid
        return tid

    title_tf, body_tf = [], []
    title_lens = np.zeros(n_docs, dtype=np.float32)
    body_lens = np.zeros(n_docs, dtype=np.float32)

    for i, (t, b) in enumerate(tqdm(zip(docs.Title, docs.Body), total=n_docs, desc="lemmatize+tf")):
        t_toks = lemmatize_text(t).split()
        b_toks = lemmatize_text(b).split()[:max_body_lemmas]
        title_lens[i] = len(t_toks)
        body_lens[i] = len(b_toks)
        title_tf.append({get_tid(tok): cnt for tok, cnt in Counter(t_toks).items()})
        body_tf.append({get_tid(tok): cnt for tok, cnt in Counter(b_toks).items()})

    n_terms = len(term_to_id)
    title_df = np.zeros(n_terms, dtype=np.int32)
    body_df = np.zeros(n_terms, dtype=np.int32)
    for d in title_tf:
        for tid in d:
            title_df[tid] += 1
    for d in body_tf:
        for tid in d:
            body_df[tid] += 1
    combined_df = title_df + body_df
    idf_array = np.log((n_docs - combined_df + 0.5) / (combined_df + 0.5) + 1.0).astype(np.float32)

    return {
        "term_to_id": term_to_id,
        "title_tf": title_tf,
        "body_tf": body_tf,
        "title_lens": title_lens,
        "body_lens": body_lens,
        "avg_title_len": float(title_lens.mean()),
        "avg_body_len": float(body_lens.mean()),
        "idf_array": idf_array,
    }


def score_query(q_terms, doc_indices, stats, k1, b_title, b_body, w_title, w_body):
    term_to_id = stats["term_to_id"]
    title_tf = stats["title_tf"]
    body_tf = stats["body_tf"]
    title_lens = stats["title_lens"]
    body_lens = stats["body_lens"]
    avg_title_len = stats["avg_title_len"]
    avg_body_len = stats["avg_body_len"]
    idf_array = stats["idf_array"]

    q_tids = [term_to_id[t] for t in q_terms if t in term_to_id]
    if not q_tids:
        return np.zeros(len(doc_indices), dtype=np.float32)

    scores = np.zeros(len(doc_indices), dtype=np.float32)
    for pos, di in enumerate(doc_indices):
        t_dict = title_tf[di]
        b_dict = body_tf[di]
        t_len = title_lens[di]
        b_len = body_lens[di]
        s = 0.0
        for tid in q_tids:
            idf = idf_array[tid]
            if idf <= 0:
                continue
            tf_t = t_dict.get(tid, 0)
            tf_b = b_dict.get(tid, 0)
            tf_tilde = (
                w_title * tf_t / (1 - b_title + b_title * t_len / avg_title_len)
                + w_body * tf_b / (1 - b_body + b_body * b_len / avg_body_len)
            )
            if tf_tilde > 0:
                s += idf * tf_tilde * (k1 + 1) / (tf_tilde + k1)
        scores[pos] = s
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="path to data directory")
    parser.add_argument("--submission_file", type=str, default="submission.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    submission_path = Path(args.submission_file)

    t0 = time.time()
    print(f"[1/4] loading docs from {data_dir / 'vkmarco-docs.tsv'}...")
    docs = load_docs(data_dir / "vkmarco-docs.tsv")
    doc_ids = docs.DocumentId.values
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    print(f"      docs={len(docs)}, elapsed={time.time()-t0:.1f}s")

    print("[2/4] building BM25F stats...")
    stats = build_stats(docs)
    del docs
    print(f"      vocab={len(stats['term_to_id'])}, "
          f"avg_title={stats['avg_title_len']:.2f}, avg_body={stats['avg_body_len']:.2f}, "
          f"elapsed={time.time()-t0:.1f}s")

    print(f"[3/4] scoring objects from {data_dir / 'objects.csv'}...")
    eval_queries = load_queries(data_dir / "vkmarco-doceval-queries.tsv")
    objects = pd.read_csv(data_dir / "objects.csv")
    q_texts_map = eval_queries.set_index("QueryId")["QueryText"].to_dict()
    print(f"      eval_queries={len(eval_queries)}, objects={len(objects)}")

    scores = pd.Series(np.nan, index=objects.ObjectId.values, dtype=np.float64)
    missing_docs = set()

    pairs_by_qid = {qid: sub for qid, sub in objects.groupby("QueryId")}
    for qid, sub in tqdm(pairs_by_qid.items(), desc="eval"):
        q_terms = lemmatize_text(q_texts_map.get(qid, "")).split()
        doc_indices, obj_ids = [], []
        for obj_id, did in zip(sub.ObjectId.values, sub.DocumentId.values):
            di = doc_id_to_idx.get(did)
            if di is None:
                missing_docs.add(did)
                continue
            doc_indices.append(di)
            obj_ids.append(obj_id)
        if not doc_indices:
            continue
        s = score_query(q_terms, doc_indices, stats, **PARAMS)
        scores.loc[obj_ids] = s

    scores = scores.fillna(0.0)
    submission = pd.DataFrame({
        "ObjectId": objects.ObjectId.values,
        "Score": scores.values,
    })

    assert len(submission) == len(objects), "row count mismatch"
    assert submission.ObjectId.is_unique, "duplicate ObjectId"
    assert submission.Score.notna().all(), "NaN in Score"
    assert np.isfinite(submission.Score).all(), "inf in Score"
    assert (submission.Score >= 0).all(), "negative Score"

    submission.to_csv(submission_path, index=False)
    print(f"\n[4/4] saved {submission_path}, rows={len(submission)}, "
          f"missing_docs={len(missing_docs)}, total_elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
