import pandas as pd

# Concateneting 20 csv files with features from dinov3 of approximately 5000 rows
df_list = []  # List to hold dataframes 

for i in range(20):
    df = pd.read_csv(f"/home/oagile/hons_physics_uwc/hons_project_2026/dinov3_features_extracted_duplicated/features_chunk_{i}.csv")
    df_list.append(df)
    print(f"features_chunk_{i} appended")

# Concatenating all dataframes
df_all = pd.concat(df_list, ignore_index=True)
print(df_all.shape)

# # Saving the new csv with all features from 99566 fits file
# df_dir = "/home/oagile/hons_physics_uwc/hons_project_2026/dinov3_features_extracted_duplicated/all_dinov3_features.csv"  # directory to save the new csv file
# df_all.to_csv(df_dir, index=False)
# print("csv file saved")









