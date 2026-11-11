import argparse
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

SEED = 42
EPS = 1e-6


def add_norm(df, cols):
    g = df.groupby("QueryId", sort=False)
    means = g[cols].transform("mean")
    stds = g[cols].transform("std").fillna(0)
    z = (df[cols] - means) / (stds + EPS)
    z.columns = [c + "_z" for c in cols]
    return pd.concat([df, z], axis=1), list(cols) + list(z.columns)


def get_groups(df):
    uniq = np.unique(df["QueryId"].values)
    mapping = {q: i for i, q in enumerate(uniq)}
    return df["QueryId"].map(mapping).values


def train(data_dir, model_file):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    feat_cols = [c for c in train_df.columns if c not in ("QueryId", "Label")]
    train_df, feat_cols = add_norm(train_df, feat_cols)

    qids = np.unique(train_df["QueryId"].values)
    rng = np.random.RandomState(SEED)
    val_qids = rng.choice(qids, size=max(1, len(qids) // 10), replace=False)
    val_mask = train_df["QueryId"].isin(val_qids).values

    tr = train_df.loc[~val_mask].reset_index(drop=True)
    va = train_df.loc[val_mask].reset_index(drop=True)

    model = CatBoostRanker(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=100,
        use_best_model=True,
    )
    model.fit(
        Pool(tr[feat_cols].values, label=tr["Label"].values, group_id=get_groups(tr)),
        eval_set=Pool(va[feat_cols].values, label=va["Label"].values, group_id=get_groups(va)),
    )
    model.save_model(model_file)
    print("saved model:", model_file)


def predict(data_dir, model_file, submission_file):
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    feat_cols = [c for c in test_df.columns if c not in ("ObjectId", "QueryId")]
    test_df, feat_cols = add_norm(test_df, feat_cols)

    model = CatBoostRanker()
    model.load_model(model_file)
    scores = model.predict(test_df[feat_cols].values)

    out = pd.DataFrame({"ObjectId": test_df["ObjectId"].values, "Score": scores})
    out.to_csv(submission_file, index=False)
    print("saved submission:", submission_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--model_file", required=True)
    p.add_argument("--submission_file")
    p.add_argument("data_dir")
    args = p.parse_args()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        predict(args.data_dir, args.model_file, args.submission_file)


if __name__ == "__main__":
    main()
    