import os
import pandas as pd

def process_csv(file_path):
    df = pd.read_csv(file_path)
    method_name = os.path.splitext(os.path.basename(file_path))[0]
    df['Training Time'] = df['Wall time'] - df['Wall time'].iloc[0]

    # Add (Nerf training) seconds if the method is "ours"
    if method_name == "Ours":
        df['Training Time'] += 40 # change based on Scene and NeRF (to direct from JSON not done due to file transfer from remote server)

    df.rename(columns={'Step': 'Step'}, inplace=True)
    df = df[['Training Time', 'Step']]
    df.insert(0, 'Method', method_name)
    return df

def merge_csvs(csv_file_1, csv_file_2, output_file):
    # Process both CSV files
    df1 = process_csv(csv_file_1)
    df2 = process_csv(csv_file_2)
    
    # Concatenate them
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the merged DataFrame
    merged_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    scene = "bicycle"
    csv_file_1 = f'{scene}//tensorboard//GSplat SfM.csv'
    csv_file_2 = f'{scene}//tensorboard//Ours.csv'
    output_file = f'{scene}//tensorboard//merged.csv'
    
    merge_csvs(csv_file_1, csv_file_2, output_file)
    print("Merged CSV saved to:", output_file)