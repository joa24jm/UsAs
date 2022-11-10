# Author Johannes Allgaier

# imports
import pandas as pd
from datetime import date, datetime
import numpy as np


class baseline_model:

    def get_baseline_user_prediction(user_id='None', df_train='None', target_name='None', approach='last'):
        """
        Gets a baseline prediction on a user level. Can either return the last known target of this user or all targets.
        :param user_id: user id of user
        :param df_train: training data of this fold
        :param target_name: name of target
        :param approach: 'last' or 'all'
        :return: prediction for target at t1
        """

        user_data = df_train[df_train.user_id == user_id]

        if approach == 'last':
            prediction = user_data.sort_values(by='created_at').iloc[-1, :][target_name]
            return prediction

        if approach == 'all':
            prediction = user_data[target_name].mean()
            return prediction

        # if arrive here, something is wrong.
        raise ValueError('Something with function call is wrong')

    def get_baseline_assessment_prediction(self, data='None', target_name='None', approach='last'):
        """
        Gets a baseline prediction on an assessment level. Can either return the last known target of this user or all targets.
        :param df_train: train data of this fold
        :param target_name: name of target
        :param approach: 'all' or 'last'
        :return: prediction for target at t1
        """

        data['baseline_estimate'] = None

        for i in range(data.shape[0]):

                if approach == 'last':
                    if i == 0:
                        pred = data[target_name].mean()
                    else:
                        pred = data.iloc[i - 1, :][target_name]
                    data['baseline_estimate'].iloc[i] = pred

                if approach == 'all':
                    if i == 0:
                        # cold start problem, so take mean of all assessments
                        pred = data[target_name].mean()
                    else:
                        # mean of all so far known assessments
                        pred = data.iloc[:i, :][target_name].mean()
                    data['baseline_estimate'].iloc[i] = pred

        return data['baseline_estimate']


def find_schedule_pattern(df, form='%Y-%m-%d %H:%M:%S', date_col_name='created_at'):
    """
    Takes a dataframe df and returns a dict that describes the duration of two filled out
    assessments of one user.
    :param df: dataframe that contains assessments of all users
           form: Format of the time stamp of the date column
           date_col_name: Name of the column containing the collection time stamp
    :return: dict like {hours: , days: , weeks: }
    """
    # find all users with more than two assessments
    s = df.user_id.value_counts() > 2
    user_ids = s[s == True].index

    hours_means, days_means = list(), list()

    for user_id in user_ids:
        sub_df = df[df['user_id'] == user_id]

        # for aggregation
        hours, days = list(), list()

        for i in np.arange(0, sub_df.shape[0] - 1):
            date_start = sub_df[date_col_name].iloc[i]
            date_start = datetime.strptime(date_start, form)
            date_end = sub_df[date_col_name].iloc[i + 1]
            date_end = datetime.strptime(date_end, form)

            delta = date_end - date_start

            hours.append(delta.total_seconds() / 3600)
            days.append(delta.total_seconds() / 3600 / 24)

        hours_means.append(np.array(hours).mean())
        days_means.append(np.array(days).mean())

    return {'avg hours between two assessments': np.array(hours_means).mean(),
            # average length between two filled out assessments in hours
            'avg days between two assessments': np.array(days_means).mean(),
            # average length between two filled out assessments in days
            'std_hours': np.array(hours_means).std(),  # std of length between two filled out assessments in hours
            'std_days': np.array(days_means).std()}  # std of length between two filled out assessments in days


def create_target_shift(df, target_name='target'):
    """
    Find the next target_t1 for each user and add it as a single column.
    :param df: dataframe with user_id, answer_id, created_at, features, target_t0
    :return: df with added column target_t1
    """

    # for each user, get target value of the next assessment
    df[f'{target_name}_t1'] = df.groupby('user_id')[f'{target_name}'].shift(periods=1, axis='index')

    # drop assessments where target is unknown
    df.dropna(subset=[f'{target_name}_t1'], inplace=True)

    return df


def main():
    # # test find schedule pattern
    # # read in a sample dataframe, uncomment to test
    # df = pd.read_csv('../../data/d01_raw/ch/22-10-05_rki_stress_followup.csv')
    # # df = pd.read_csv('../../data/d01_raw/ch/22-10-05_rki_parent_followup.csv')
    # # df = pd.read_csv('../../data/d01_raw/ch/22-10-05_rki_heart_followup.csv')
    # res = find_schedule_pattern(df, form='%Y-%m-%d %H:%M:%S', date_col_name='created_at')
    # print(res)

    # df = pd.read_csv('../../data/d01_raw/tyt/22-10-24_standardanswers.csv', index_col='Unnamed: 0')
    df = pd.read_csv('../../data/d01_raw/uniti/uniti_dataset_22.09.28.csv')
    # print(df.shape)
    #
    # # test target shift
    # df = create_target_shift(df, target_name='cumberness')
    # print(df.shape)

    # test baseline approach
    df_train = df
    sample = df_train.sample(n=1)
    time_column = 'created_at'
    time_stamp = sample[time_column]
    target_name = 'cumberness'
    user_id = df_train.sample(n=1).user_id.iloc[0]

    model = baseline_model()

    # print('user')
    # for approach in ['last', 'all']:
    #     pred = model.get_baseline_user_prediction(user_id=user_id, df_train=df_train, target_name=target_name,
    #                                                        approach=approach)
    #     print(approach, '\t', pred)
    print('assessment')
    for approach in ['last', 'all']:
        pred = model.get_baseline_assessment_prediction(data=df_train, target_name=target_name, approach=approach)
        print(approach, '\t', pred)


if __name__ == '__main__':
    main()
