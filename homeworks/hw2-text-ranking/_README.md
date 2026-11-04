
# HW2 — VK MARCO text ranking

- Kaggle nick: **Snomty**
- Kaggle команда (в лидерборде результат под ней) **Minakov Danil**
- Public LB: **0.61670**

## Модель

BM25F с лемматизацией pymorphy3, Body усечён до 300 лемм.
Параметры: k1=0.7, b_title=0.5, b_body=0.75, w_title=12.0, w_body=1.0.

## nDCG@10 на train (CV 5 folds)

- tfidf_baseline: 0.8109 ± 0.0080
- bm25f_base: 0.8219 ± 0.0053
- bm25f_tuned2 (финал): 0.8283 ± 0.0068

## Время работы

~5–6 минут, пик памяти ~2–3 ГБ.
