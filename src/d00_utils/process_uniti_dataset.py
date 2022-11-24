import pandas as pd
from src.d00_utils import helpers


def load_uniti_dataset(file_path: str = "data/d01_raw/uniti/uniti_dataset_22.09.28.csv", user_id_col: str = 'user_id',
                       timestamp_col: str = 'created_at', target_col: str = 'cumberness') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = helpers.create_target_shift(df, target_name=target_col)
    df.sort_values(by=[timestamp_col, user_id_col], inplace=True)
    df = df.reset_index().rename({'index':'answer_id'}, axis=1)
    return df


def main():
    uniti_df = load_uniti_dataset()
    uniti_df.to_csv("data/d02_processed/uniti.csv", index=False)


if __name__ == "__main__":
    main()
