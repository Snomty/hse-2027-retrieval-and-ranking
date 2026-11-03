import pandas as pd

# Чтение TXT
df = pd.read_csv('ObjectId,Label.txt')  # pandas сам определит разделитель

# Сохранение в CSV
df.to_csv('ObjectId,Label.csv', index=False, encoding='utf-8-sig')