
# ДЗ 3. Обучение ранжированию

## Kaggle

- Ник: `Snomty`
- Скор на публичном лидерборде: **0.623** (побит бейзлайн 0.61621)

## Подход

Модель: `CatBoostRanker` с функцией потерь `YetiRank`.

Признаки: 67 исходных + 67 z-score внутри запроса (нормировка по `QueryId`).
Итого 134 признака.

Гиперпараметры:

- `iterations=1000`
- `learning_rate=0.05`
- `depth=6`
- `l2_leaf_reg=3.0`
- `loss_function="YetiRank"`
- `eval_metric="NDCG:top=10"`
- `early_stopping_rounds=100`, `use_best_model=True`

Валидация: 10% уникальных запросов из train отложены для early stopping.
Seed: 42.

## Как воспроизвести

```bash
pip install -r requirements.txt

python solution.py --train --model_file=model.bin data
python solution.py --model_file=model.bin --submission_file=submission.csv data
```
