import pandas as pd
from src.d00_utils import helpers


def get_tyt_test_users() -> list:
    return [1, 2, 11, 36, 39, 40, 41, 42, 43, 46, 47, 48, 49, 54, 64, 66, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            103, 104, 108, 109, 115, 116, 132, 158, 191, 196, 218, 374, 461, 553, 728, 845, 1119, 1563, 2186,
            2242, 2244]


def get_tyt_columns() -> list:
    return ['id', 'user_id', 'created_at', 'question1', 'question2', 'question3', 'question4', 'question5',
            'question6', 'question7', 'question8']


def load_tyt_dataset(filepath: str = "data/d01_raw/tyt/22-01-17_standardanswers.csv", user_id_col: str = 'user_id',
                     timestamp_col: str = 'created_at', target_col: str = 'question3'):
    columns_to_keep = get_tyt_columns()
    test_users = get_tyt_test_users()
    df = pd.read_csv(filepath)
    df = df[~df.user_id.isin(test_users)]
    df = df[columns_to_keep]
    df = helpers.create_target_shift(df, target_name=target_col)
    df = df.sort_values(by=[timestamp_col, user_id_col]).rename({'id': 'answer_id'}, axis=1)
    return df


def main():
    tyt_dataset = load_tyt_dataset()
    tyt_dataset.to_csv("data/d02_processed/tyt.csv", index=False)


if __name__ == "__main__":
    main()
