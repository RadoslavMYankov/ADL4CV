import pandas as pd

df = pd.read_csv('utils/density_results.csv')

sorted_df = df.sort_values(by='num_projected_points', ascending=True)
print(sorted_df.head(5))

threshold = 1000

sparse_frames = sorted_df[sorted_df['num_projected_points'] < threshold]['image_id'].tolist()
print(sorted(sparse_frames))
