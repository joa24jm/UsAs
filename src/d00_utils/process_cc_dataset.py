import pandas as pd
from src.d00_utils import helpers


def get_cc_columns() -> list:
    return ['answer_id', 'user_id', 'created_at', 'person', 'age', 'gender', 'education', 'author', 'fever',
            'sorethroat', 'runnynose', 'cough', 'losssmell', 'losstaste', 'shortnessbreath', 'headace', 'musclepain',
            'diarrhea', 'generalweakness', 'corona_result']


def load_corona_check(filepath: str = 'data/d01_raw/cc/22-06-29_corona-check-data.csv', user_id_col: str = 'user_id',
                      timestamp_col: str = 'created_at', target_col: str = 'question3'):
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

    df = pd.read_csv(filepath)
    df = drop_one_time_users_cc(df)
    df = drop_ambiguous_users_cc(df)
    df = df[(df.questionnaire_id == 3) & (df.research == "YES")]

    columns_to_keep = get_cc_columns()
    df = df[columns_to_keep]
    return helpers.create_target_shift(df, target_name='corona_result')


def main():
    cc_dataset = load_corona_check()
    cc_dataset.to_csv("data/d02_processed/cc.csv", index=False)


if __name__ == "__main__":
    main()
