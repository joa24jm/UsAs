# Author Vishnu Unnikrishnan

# imports
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import sys


def insert_lagged_target(df: pd.DataFrame, target_col, user_id_col="user_id", n_lags=1):
    df_target = None
    for user in df[user_id_col].unique():
        df_user = df[df[user_id_col] == user]
        df_user['target'] = df_user[target_col].shift(-n_lags)
        if df_target is None:
            df_target = df_user[:-n_lags]
        else:
            df_target = pd.concat([df_target, df_user[:-n_lags]])
    return df_target


def load_corona_check():
    def drop_ambiguous_users_cc(df):
        """
        Takes the whole df, filters one user and drops assessments from that user, if
        - the age varies from the most common age of the users
        - the gender varies from the most common gender of the users
        - the education varies from the most common education of the users
        - the user filled out the questionnaire for others
        :param df: corona check dataframe
        :param user_id: user id of that users
        :return: reduced dataframe
        """

        user_ids = list(df.user_id.unique())

        for user_id in user_ids:

            sub_df = df[df['user_id'] == user_id]

            # filled out for him-/herself
            sub_df = sub_df[sub_df.author == 'MYSELF']
            all_assessments = sub_df.index

            if sub_df.shape[0] > 0:

                try:
                    # most common age
                    mca = sub_df.age.value_counts().index[0]
                    f1 = (sub_df.age == mca)
                    # most common education
                    mce = sub_df.education.value_counts().index[0]
                    f2 = (sub_df.education == mce)
                    # most common gender
                    mcg = sub_df.gender.value_counts().index[0]
                    f3 = (sub_df.gender == mcg)

                except:  # some users skipped those baseline questions - we drop all assessments from those
                    continue

                filtered_assessments = sub_df[f1 & f2 & f3].index
                assessments_to_drop = list(set(all_assessments) - set(filtered_assessments))

                df.drop(index=assessments_to_drop, inplace=True)

        return df

    def drop_one_time_users_cc(df):
        """
        Drops users with less than 2 assessments.
        :param df:
        :return: dataframe with users that have at lest two assessments.
        """

        s = df.user_id.value_counts() > 1
        users = s[s == True].index

        return df[df['user_id'].isin(users)]

    df = pd.read_csv('data/d01_raw/cc/22-06-29_corona-check-data.csv')
    df = drop_one_time_users_cc(df)
    df = drop_ambiguous_users_cc(df)
    df = df[(df.questionnaire_id == 3) & (df.research == "YES")]

    columns_to_keep = ['answer_id', 'user_id', 'created_at', 'person', 'age', 'gender', 'education', 'author',
                       'fever', 'sorethroat', 'runnynose', 'cough', 'losssmell',
                       'losstaste', 'shortnessbreath', 'headace', 'musclepain', 'diarrhea', 'generalweakness',
                       'corona_result']
    df = df[columns_to_keep]
    return insert_lagged_target(df, 'corona_result')


def load_rki_parent_dataset():
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

    df = pd.read_csv("data/d01_raw/ch/22-07-01_rki_parent_followup.csv")
    df = df[df.questionnaire_id == 10]  # TODO @Johannes, I guess followup should be only 1 questionnaire? ~,How to lag?
    df['phq9_score'] = compute_phq9_score(df)  # .apply(phq9_score_to_level)
    df = insert_lagged_target(df, 'phq9_score')
    columns_to_keep = ['answer_id', 'questionnaire_id', 'user_id', 'created_at', 'pers',
                       'a_ort', 'a_lohn', 'covid1', 'covid2', 'covid3', 'mhem1', 'pt_aktuell2', 'klima1', 'gewalt1',
                       'phqd_a', 'phqd_a1', 'phqd_a2', 'phqd_a3', 'phqd_a4', 'phqd_b', 'phqd_c', 'phqd_d', 'phqd_e',
                       'phqd_f', 'phqd_g', 'phqd_h', 'phqd_i', 'phqd_j', 'phqd_k', 'einsam1', 'einsam2', 'einsam3',
                       'interact1', 'interact2', 'phq9_a', 'phq9_b', 'phq9_c', 'phq9_d', 'phq9_e', 'phq9_f', 'phq9_g',
                       'phq9_h', 'phq9_i', 'phqpd_a', 'gad7_a', 'gad7_b', 'gad7_c', 'gad7_d', 'gad7_e', 'gad7_f',
                       'gad7_g', 'tinnitus', 'phqd_belast', 'qol1', 'qol2', 'qol10', 'qol12', 'qol17', 'qol19', 'qol20',
                       'qol23', 'schlaf_1', 'schlaf_2', 'schlaf_3', 'schlaf_4', 'schlaf_5', 'schlaf_6', 'schlaf_7',
                       'sport2', 'alk_1', 'alk_2', 'feedback', 'phqpd_b', 'phqpd_c', 'phqpd_d', 'phq9_score', 'target']
    df = df[columns_to_keep]
    return df


def load_rki_children_dataset():
    df = pd.read_csv("data/d01_raw/ch/22-07-01_rki_children_followup.csv")
    df = df[df.questionnaire_id == 14]  # todo Johannes

    columns_to_keep = ['answer_id', 'questionnaire_id', 'user_id', 'created_at', 'kj_cv_inf', 'kj_cv_fam',
                       'kj_cvad', 'kj_school4', 'kj_restr_cur', 'kj_restr_day', 'kj_restr_out', 'kj_sport', 'kj_olfac',
                       'kj_famclim2', 'kj_famarg2', 'kj_viol', 'kj_anx2', 'kj_anx3', 'kj_media2', 'kj_qol1', 'kj_qol2',
                       'kj_qol3', 'kj_qol4', 'kj_qol5', 'kj_qol6', 'kj_qol7', 'kj_qol8', 'kj_qol9', 'kj_qol10',
                       'kj_scas1',
                       'kj_scas2', 'kj_scas3', 'kj_scas4', 'kj_scas5', 'kj_scas6', 'kj_scas7', 'kj_scas8',
                       'kj_phq_hope',
                       'kj_phq_interest', 'kj_phq_sleep']
    df = df[columns_to_keep]
    df = insert_lagged_target(df, target_col="kj_phq_hope")
    return df


def load_rki_heart_dataset():
    df = pd.read_csv("data/d01_raw/ch/22-07-01_rki_heart_followup.csv")
    df = df[df.questionnaire_id == 12]  # TODO Johannes, help!
    columns_to_keep = ['answer_id', 'user_id', 'created_at',
                       'smoke1', 'smoke2', 'alcoh2', 'alcoh3', 'fruit2', 'fruit3', 'veget2', 'veget3',
                       'fastf1', 'fastf2', 'sport2', 'sport3', 'platf1', 'weigh2', 'hyper1', 'hyper2',
                       'hyper4', 'hyper5', 'diabe1', 'diabe2', 'diabe4', 'diabe5', 'blood1', 'blood2', 'blood3',
                       'medic1', 'medic2', 'physi1', 'physi2', 'hospi1', 'hospi2', 'medic3', 'pain1', 'pain2']
    df = df[columns_to_keep]
    df = insert_lagged_target(df, target_col='pain2')
    return df


def load_ch_stress_dataset():
    df = pd.read_csv("data/d01_raw/ch/22-07-01_rki_stress_followup.csv")
    columns_to_keep = ['answer_id', 'questionnaire_id', 'user_id', 'created_at',
                       'pss1', 'pss2', 'pss3', 'pss4', 'pss5', 'pss6', 'pss7', 'pss8', 'pss9', 'pss10']
    df = df[columns_to_keep]
    df = insert_lagged_target(df, target_col='pss10')
    return df


def load_tyt_dataset():
    test_users = [1, 2, 11, 36, 39, 40, 41, 42, 43, 46, 47, 48, 49, 54, 64, 66, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                  103, 104, 108, 109, 115, 116, 132, 158, 191, 196, 218, 374, 461, 553, 728, 845, 1119, 1563, 2186,
                  2242, 2244]
    df = pd.read_csv("data/d01_raw/tyt/22-01-17_standardanswers.csv")
    df = df[~df.user_id.isin(test_users)]
    columns_to_keep = ['id', 'user_id', 'created_at', 'question1', 'question2', 'question3', 'question4', 'question5',
                       'question6', 'question7', 'question8']
    df = df[columns_to_keep]
    df = insert_lagged_target(df, target_col='question3')
    return df


def load_uniti_dataset():
    df = pd.read_csv("data/d01_raw/uniti/uniti_dataset_22.09.28.csv")
    df = insert_lagged_target(df, target_col='cumberness')
    return df

def main():
    cc = load_corona_check()
    rki_parent = load_rki_parent_dataset()
    rki_children = load_rki_children_dataset()
    rki_heart = load_rki_heart_dataset()
    ch_stress = load_ch_stress_dataset()

    tyt = load_tyt_dataset()
    uniti = load_uniti_dataset()

    cc.to_csv("data/d02_processed/cc.csv", sep=",", index=False)
    rki_parent.to_csv("data/d02_processed/rki_parent.csv", sep=",", index=False)
    rki_children.to_csv("data/d02_processed/rki_children.csv", sep=",", index=False)
    rki_heart.to_csv("data/d02_processed/rki_heart.csv", sep=",", index=False)
    ch_stress.to_csv("data/d02_processed/ch_stress.csv", sep=",", index=False)
    tyt.to_csv("data/d02_processed/tyt.csv", sep=",", index=False)
    uniti.to_csv("data/d02_processed/uniti.csv", sep=",", index=False)


if __name__ == '__main__':
    main()
