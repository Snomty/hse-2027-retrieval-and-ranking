import re
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pymorphy3
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

# Корень проекта — папка, где лежит utils.py
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

_morph = pymorphy3.MorphAnalyzer()
_lemma_cache = {}


# ---------- Загрузка ----------
def load_docs(path=None):
    path = path or DATA_DIR / "vkmarco-docs.tsv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            rows.append(tuple(parts[:4]))
    return pd.DataFrame(rows, columns=["DocumentId", "URL", "Title", "Body"])


def load_queries(path=None):
    path = path or DATA_DIR / "vkmarco-doctrain-queries.tsv"
    return pd.read_csv(path, sep="\t", header=None,
                       names=["QueryId", "QueryText"], encoding="utf-8")


def load_qrels(path=None):
    path = path or DATA_DIR / "vkmarco-doctrain-qrels.tsv"
    return pd.read_csv(path, sep=r"\s+", header=None,
                       names=["QueryId", "Q0", "DocumentId", "Label"],
                       encoding="utf-8")[["QueryId", "DocumentId", "Label"]]


def load_eval_queries(path=None):
    path = path or DATA_DIR / "vkmarco-doceval-queries.tsv"
    return pd.read_csv(path, sep="\t", header=None,
                       names=["QueryId", "QueryText"], encoding="utf-8")


# ---------- Лемматизация ----------
def lemmatize_text(text: str) -> str:
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


def load_docs_lemma(rebuild=False):
    path = CACHE_DIR / "docs_lemmatized.pkl"
    if path.exists() and not rebuild:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("docs_lemmatized.pkl corrupted, rebuilding...")
            path.unlink()
    docs = load_docs()
    out = docs[["DocumentId"]].copy()
    out["Title_lemma"] = [lemmatize_text(t) for t in tqdm(docs.Title, desc="title")]
    out["Body_lemma"]  = [lemmatize_text(b) for b in tqdm(docs.Body, desc="body")]
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


# ---------- Статистики BM25F (компактные, dict[int,int]) ----------
def build_bm25f_stats(docs_lemma, max_body_lemmas=300, save=True):
    """
    Строит статистики BM25F в компактном виде:
    - term_to_id: dict[str, int]
    - title_tf, body_tf: list[dict[int, int]] — tid -> count
    - idf_array: np.ndarray[float32] длины len(term_to_id)
    Память: ~100-200 МБ вместо ~1.5 ГБ у Counter[str].
    """
    n_docs = len(docs_lemma)
    term_to_id = {}

    def get_tid(term):
        tid = term_to_id.get(term)
        if tid is None:
            tid = len(term_to_id)
            term_to_id[term] = tid
        return tid

    title_tf_ids = []
    body_tf_ids = []
    title_lens = np.zeros(n_docs, dtype=np.float32)
    body_lens  = np.zeros(n_docs, dtype=np.float32)

    for i, t in enumerate(tqdm(docs_lemma.Title_lemma, desc="title_tf")):
        toks = t.split() if isinstance(t, str) else []
        title_lens[i] = len(toks)
        title_tf_ids.append({get_tid(tok): cnt for tok, cnt in Counter(toks).items()})

    for i, b in enumerate(tqdm(docs_lemma.Body_lemma, desc="body_tf")):
        toks = b.split() if isinstance(b, str) else []
        if max_body_lemmas:
            toks = toks[:max_body_lemmas]
        body_lens[i] = len(toks)
        body_tf_ids.append({get_tid(tok): cnt for tok, cnt in Counter(toks).items()})

    n_terms = len(term_to_id)
    print(f"vocab size: {n_terms}")

    # df по полям — массивы уже финальной длины
    title_df = np.zeros(n_terms, dtype=np.int32)
    body_df  = np.zeros(n_terms, dtype=np.int32)
    for d in title_tf_ids:
        for tid in d:
            title_df[tid] += 1
    for d in body_tf_ids:
        for tid in d:
            body_df[tid] += 1

    combined_df = title_df + body_df
    idf_array = np.log((n_docs - combined_df + 0.5) / (combined_df + 0.5) + 1.0).astype(np.float32)

    stats = {
        "term_to_id": term_to_id,
        "title_tf": title_tf_ids,
        "body_tf": body_tf_ids,
        "title_lens": title_lens,
        "body_lens": body_lens,
        "avg_title_len": float(title_lens.mean()),
        "avg_body_len":  float(body_lens.mean()),
        "idf_array": idf_array,
        "max_body_lemmas": max_body_lemmas,
    }
    if save:
        path = CACHE_DIR / "bm25f_stats.pkl"
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        print(f"saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return stats


def get_bm25f_stats(docs_lemma, max_body_lemmas=300):
    """Загружает статистики из кэша или строит заново. Пересобирает при повреждении."""
    path = CACHE_DIR / "bm25f_stats.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                stats = pickle.load(f)
            # проверка формата: должен быть компактный (term_to_id + idf_array)
            if "term_to_id" not in stats or "idf_array" not in stats:
                print("cache in old format, rebuilding...")
                path.unlink()
            else:
                return stats
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"cache corrupted ({e}), rebuilding...")
            path.unlink()
    print("building bm25f stats...")
    return build_bm25f_stats(docs_lemma, max_body_lemmas=max_body_lemmas, save=True)


def bm25f_score_query(q_terms, doc_indices, stats,
                      k1=1.5, b_title=0.5, b_body=0.75, w_title=3.0, w_body=1.0):
    """Скорит один запрос по списку doc_indices."""
    term_to_id = stats["term_to_id"]
    idf_array = stats["idf_array"]
    title_tf = stats["title_tf"]
    body_tf = stats["body_tf"]
    title_lens = stats["title_lens"]
    body_lens = stats["body_lens"]
    avg_title_len = stats["avg_title_len"]
    avg_body_len = stats["avg_body_len"]

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
                + w_body  * tf_b / (1 - b_body  + b_body  * b_len / avg_body_len)
            )
            if tf_tilde > 0:
                s += idf * tf_tilde * (k1 + 1) / (tf_tilde + k1)
        scores[pos] = s
    return scores


# ---------- Метрики ----------
def ndcg_at_k(rels_sorted, k=10):
    rels = np.asarray(rels_sorted, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    gains = 2.0 ** rels - 1.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    dcg = (gains / discounts).sum()
    ideal = np.sort(rels)[::-1][:k]
    gains_i = 2.0 ** ideal - 1.0
    disc_i = np.log2(np.arange(2, ideal.size + 2))
    idcg = (gains_i / disc_i).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_run(run_df, qrels_df, k=10):
    merged = run_df.merge(qrels_df, on=["QueryId", "DocumentId"], how="left")
    merged["Label"] = merged["Label"].fillna(0).astype(int)
    merged = merged.sort_values(["QueryId", "Score", "DocumentId"],
                                ascending=[True, False, True])
    scores = []
    for _, grp in merged.groupby("QueryId", sort=False):
        scores.append(ndcg_at_k(grp.Label.values, k=k))
    return float(np.mean(scores)) if scores else 0.0


def cross_validate(score_fn, train_queries, train_qrels, n_folds=5):
    full_run = score_fn(train_queries, train_qrels)
    full_ndcg = evaluate_run(full_run, train_qrels, k=10)

    unique_qids = train_qrels.QueryId.unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold_idx, (_, val_idx) in enumerate(kf.split(unique_qids)):
        val_qids = unique_qids[val_idx]
        val_pairs = train_qrels[train_qrels.QueryId.isin(val_qids)]
        run_val = score_fn(train_queries, val_pairs)
        fold_scores.append(evaluate_run(run_val, train_qrels, k=10))
        print(f"  fold {fold_idx}: nDCG@10 = {fold_scores[-1]:.4f}")
    return full_ndcg, np.array(fold_scores)


# ---------- Реестр ----------
REGISTRY_PATH = RESULTS_DIR / "registry.pkl"
REGISTRY_COLUMNS = ["name", "config", "full_train_ndcg", "cv_mean", "cv_std"]


def append_result(name, config, full_train_ndcg, fold_scores):
    if REGISTRY_PATH.exists():
        df = pd.read_pickle(REGISTRY_PATH)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)
    df.loc[len(df)] = {
        "name": name,
        "config": config,
        "full_train_ndcg": float(full_train_ndcg),
        "cv_mean": float(fold_scores.mean()),
        "cv_std": float(fold_scores.std()),
    }
    df.to_pickle(REGISTRY_PATH)
    return df


def show_registry():
    if not REGISTRY_PATH.exists():
        print("registry empty")
        return None
    df = pd.read_pickle(REGISTRY_PATH)
    cols = [c for c in df.columns if c != "config"]
    print(df[cols].to_string(index=False))
    return df


# ---------- LM Dirichlet (по объединённому тексту) ----------
def build_lm_stats(docs_lemma, max_body_lemmas=300, save=True):
    n_docs = len(docs_lemma)
    term_to_id = {}

    def get_tid(term):
        tid = term_to_id.get(term)
        if tid is None:
            tid = len(term_to_id)
            term_to_id[term] = tid
        return tid

    doc_tf = []
    doc_lens = np.zeros(n_docs, dtype=np.float32)

    for i, (t, b) in enumerate(tqdm(zip(docs_lemma.Title_lemma, docs_lemma.Body_lemma),
                                     total=n_docs, desc="lm_tf")):
        t_toks = t.split() if isinstance(t, str) else []
        b_toks = b.split()[:max_body_lemmas] if isinstance(b, str) else []
        toks = t_toks + b_toks
        doc_lens[i] = len(toks)
        doc_tf.append({get_tid(tok): cnt for tok, cnt in Counter(toks).items()})

    n_terms = len(term_to_id)
    print(f"vocab size: {n_terms}")

    term_freq_array = np.zeros(n_terms, dtype=np.float32)
    for d in doc_tf:
        for tid, cnt in d.items():
            term_freq_array[tid] += cnt

    total_terms = float(term_freq_array.sum())

    stats = {
        "term_to_id": term_to_id,
        "doc_tf": doc_tf,
        "doc_lens": doc_lens,
        "avg_doc_len": float(doc_lens.mean()),
        "term_freq_array": term_freq_array,
        "total_terms": total_terms,
        "max_body_lemmas": max_body_lemmas,
    }
    if save:
        path = CACHE_DIR / "lm_stats.pkl"
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        print(f"saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return stats


def get_lm_stats(docs_lemma, max_body_lemmas=300):
    path = CACHE_DIR / "lm_stats.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"lm cache corrupted ({e}), rebuilding...")
            path.unlink()
    print("building lm stats...")
    return build_lm_stats(docs_lemma, max_body_lemmas=max_body_lemmas, save=True)


def lm_score_query(q_terms, doc_indices, lm_stats, mu=2000.0):
    term_to_id = lm_stats["term_to_id"]
    doc_tf = lm_stats["doc_tf"]
    doc_lens = lm_stats["doc_lens"]
    term_freq_array = lm_stats["term_freq_array"]
    total_terms = lm_stats["total_terms"]
    n_terms = len(term_to_id)
    denom_pc = total_terms + 0.5 * n_terms

    q_tids = [term_to_id[t] for t in q_terms if t in term_to_id]
    if not q_tids:
        return np.zeros(len(doc_indices), dtype=np.float32)

    p_c = np.array([(term_freq_array[tid] + 0.5) / denom_pc for tid in q_tids],
                   dtype=np.float64)

    scores = np.zeros(len(doc_indices), dtype=np.float32)
    for pos, di in enumerate(doc_indices):
        d_dict = doc_tf[di]
        d_len = doc_lens[di]
        s = 0.0
        for k, tid in enumerate(q_tids):
            tf = d_dict.get(tid, 0)
            p = (tf + mu * p_c[k]) / (d_len + mu)
            s += np.log(p)
        scores[pos] = s
    return scores
