import pandas as pd
from src.d00_utils import helpers


def get_rki_parent_columns():
    return ['answer_id', 'questionnaire_id', 'user_id', 'created_at', 'pers',
            'a_ort', 'a_lohn', 'covid1', 'covid2', 'covid3', 'mhem1', 'pt_aktuell2', 'klima1', 'gewalt1',
            'phqd_a', 'phqd_a1', 'phqd_a2', 'phqd_a3', 'phqd_a4', 'phqd_b', 'phqd_c', 'phqd_d', 'phqd_e',
            'phqd_f', 'phqd_g', 'phqd_h', 'phqd_i', 'phqd_j', 'phqd_k', 'einsam1', 'einsam2', 'einsam3',
            'interact1', 'interact2', 'phq9_a', 'phq9_b', 'phq9_c', 'phq9_d', 'phq9_e', 'phq9_f', 'phq9_g',
            'phq9_h', 'phq9_i', 'phqpd_a', 'gad7_a', 'gad7_b', 'gad7_c', 'gad7_d', 'gad7_e', 'gad7_f',
            'gad7_g', 'tinnitus', 'phqd_belast', 'qol1', 'qol2', 'qol10', 'qol12', 'qol17', 'qol19', 'qol20',
            'qol23', 'schlaf_1', 'schlaf_2', 'schlaf_3', 'schlaf_4', 'schlaf_5', 'schlaf_6', 'schlaf_7',
            'sport2', 'alk_1', 'alk_2', 'feedback', 'phqpd_b', 'phqpd_c', 'phqpd_d', 'phq9_score']


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


def phq9_level_to_category(df, cols):
    """
    Convert level to category
    :param df: dataframe
    :param cols: list of columns to apply
    :return: categorical columns
    """

    dic = {'None': 0,
           'Mild': 1,
           'Moderate': 2,
           'Moderate-to-Severe': 3,
           'Severe': 4}

    df[cols] = df[cols].replace(dic)

    return df


def compute_phq9_score(df: pd.DataFrame):
    """
    Computes the score for the phq9 questionnaire as the sum of all questions.
    :param df: the data frame with the questionnaire
    :return: data frame with new column 'phq9_score' that has the total
    """
    phq9_columns = ['phq9_a', 'phq9_b', 'phq9_c', 'phq9_d', 'phq9_e', 'phq9_f', 'phq9_g', 'phq9_h', 'phq9_i']
    scores = df[phq9_columns].sum(axis=1).apply(phq9_score_to_level)
    return scores


def get_features(df: pd.DataFrame):
    features = [col for col in df.columns if ('phq9' in col) and ('_t1' not in col)]

    return features


def load_rki_parent_dataset(filepath: str = "../../data/d01_raw/ch/22-10-05_rki_parent_followup.csv",
                            user_id_col: str = 'user_id', timestamp_col: str = 'created_at'):
    """
    Creates the RKI parent dataset. Not following convention - excluding the target_col param because it is computed in this case.
    :param filepath: path to raw CSV file
    :param user_id_col: column which contains user id
    :param timestamp_col: column which contains timestamp
    :return: the dataset obj with the target variable
    """
    df = pd.read_csv(filepath)
    df['phq9_score'] = compute_phq9_score(df)
    df = df[get_rki_parent_columns()]
    df = phq9_level_to_category(df, ['phq9_score'])
    df = helpers.create_target_shift(df, 'phq9_score')
    df.sort_values(by=[timestamp_col, user_id_col], inplace=True)
    return df


def main():
    rki_parent_dataset = load_rki_parent_dataset()
    rki_parent_dataset.to_csv("../../data/d02_processed/rki_parent.csv", index=False)


if __name__ == "__main__":
    main()
