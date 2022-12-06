import pandas as pd
from src.d00_utils import helpers


def get_ch_stress_columns():
    return ['answer_id', 'questionnaire_id', 'user_id', 'created_at',
            'pss1', 'pss2', 'pss3', 'pss4', 'pss5', 'pss6', 'pss7', 'pss8', 'pss9', 'pss10']


def load_ch_stress_dataset(filepath: str = "../../data/d01_raw/ch/22-10-05_rki_stress_followup.csv",
                           user_id_col: str = 'user_id', timestamp_col: str = 'created_at',
                           target_col: str = 'pss10'):
    df = pd.read_csv(filepath)
    df = df[get_ch_stress_columns()]
    df = helpers.create_target_shift(df, target_name=target_col)
    df.sort_values(by=[timestamp_col, user_id_col], inplace=True)
    return df


def main():
    ch_stress_dataset = load_ch_stress_dataset()
    ch_stress_dataset.to_csv("../../data/d02_processed/ch_stress.csv", index=False)

if __name__ == "__main__":
    main()
