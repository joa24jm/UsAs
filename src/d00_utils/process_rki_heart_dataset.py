import pandas as pd
from src.d00_utils import helpers


def get_rki_heart_columns():
    return ['answer_id', 'user_id', 'created_at', 'smoke1', 'smoke2', 'alcoh2', 'alcoh3', 'fruit2', 'fruit3', 'veget2',
            'veget3', 'fastf1', 'fastf2', 'sport2', 'sport3', 'platf1', 'weigh2', 'hyper1', 'hyper2', 'hyper4',
            'hyper5', 'diabe1', 'diabe2', 'diabe4', 'diabe5', 'blood1', 'blood2', 'blood3', 'medic1', 'medic2',
            'physi1', 'physi2', 'hospi1', 'hospi2', 'medic3', 'pain1', 'pain2']


def load_rki_heart_dataset(filepath: str = "data/d01_raw/ch/22-07-01_rki_heart_followup.csv",
                           user_id_col: str = 'user_id', timestamp_col: str = 'created_at'):
    """
    Creates the RKI heart dataset. Not following convention - excluding the target_col param because it is computed in this case.
    :param filepath: path to raw CSV file
    :param user_id_col: column which contains user id
    :param timestamp_col: column which contains timestamp
    :return: the dataset obj with the target variable
    """
    df = pd.read_csv(filepath)
    df = df[get_rki_heart_columns()]
    df = helpers.create_target_shift(df, 'pain2')
    df.sort_values(by=[timestamp_col, user_id_col], inplace=True)
    return df


def main():
    rki_heart_dataset = load_rki_heart_dataset()
    rki_heart_dataset.to_csv("data/d02_processed/rki_heart.csv", index=False)


if __name__ == "__main__":
    main()
