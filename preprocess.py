import pandas as pd

df = pd.read_csv("movies_genres_1.csv", delimiter='\t')

df.drop('Lifestyle', axis=1, inplace=True)
df.to_csv("movies_genres_en.csv", sep='\t', encoding='utf-8', index=False)