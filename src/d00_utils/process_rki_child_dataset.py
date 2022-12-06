import pandas as pd
from src.d00_utils import helpers


def get_rki_child_columns():
    return ['answer_id', 'questionnaire_id', 'user_id', 'created_at', 'kj_cv_inf', 'kj_cv_fam', 'kj_cvad', 'kj_school4',
            'kj_restr_cur', 'kj_restr_day', 'kj_restr_out', 'kj_sport', 'kj_olfac', 'kj_famclim2', 'kj_famarg2',
            'kj_viol', 'kj_anx2', 'kj_anx3', 'kj_media2', 'kj_qol1', 'kj_qol2', 'kj_qol3', 'kj_qol4', 'kj_qol5',
            'kj_qol6', 'kj_qol7', 'kj_qol8', 'kj_qol9', 'kj_qol10', 'kj_scas1', 'kj_scas2', 'kj_scas3', 'kj_scas4',
            'kj_scas5', 'kj_scas6', 'kj_scas7', 'kj_scas8', 'kj_phq_hope', 'kj_phq_interest', 'kj_phq_sleep']


def phq9_score_to_level(x):
    try:
        if x <= 4:
            return "None"
        elif x <= 9:
            return "Mild"
        elif x <= 14:
            return "Moderate"
        elif x <= 19:
            return "Moderate-to-Severe"
        elif x <= 27:
            return "Severe"
        else:
            return "Error"
    except:
        print(x)
        input("Error")


def compute_phq9_score(df: pd.DataFrame):
    """
   Computes the score for the phq9 questionnaire as the sum of all questions.
   :param df: the data frame with the questionnaire
   :return: data frame with new column 'phq9_score' that has the total
   """
    phq9_columns = ['phq9_a', 'phq9_b', 'phq9_c', 'phq9_d', 'phq9_e', 'phq9_f', 'phq9_g', 'phq9_h', 'phq9_i']
    scores = df[phq9_columns].sum(axis=1).apply(phq9_score_to_level)
    return scores


def load_rki_child_dataset(filepath: str = "../../data/d01_raw/ch/22-10-05_rki_children_followup.csv",
                           user_id_col: str = 'user_id', timestamp_col: str = 'created_at'):
    """
    Creates the RKI child dataset. Not following convention - excluding the target_col param because it is computed in this case.
    :param filepath: path to raw CSV file
    :param user_id_col: column which contains user id
    :param timestamp_col: column which contains timestamp
    :return: the dataset obj with the target variable
    """
    df = pd.read_csv(filepath)
    df = df[get_rki_child_columns()]
    df = helpers.create_target_shift(df, 'kj_phq_hope')
    df.sort_values(by=[timestamp_col, user_id_col], inplace=True)
    return df


def main():
    rki_child_dataset = load_rki_child_dataset()
    rki_child_dataset.to_csv("../../data/d02_processed/rki_children.csv", index=False)


if __name__ == "__main__":
    main()
