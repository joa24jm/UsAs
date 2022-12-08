# Author Johannes Allgaier

from datetime import datetime

import numpy as np
# imports
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class baseline_model:

    def get_baseline_user_prediction(self, data='None', target_name='None', approach='last', time_col='created_at'):
        """
        Gets a baseline prediction on a user level. Can either return the last known target of this user or all targets.
        :param data: training data of this fold
        :param target_name: name of target
        :param approach: 'last' or 'all'
        :return: prediction for target at t1
        """

        gb = data.groupby('user_id')
        groups = dict(list(gb))

        data['baseline_estimate'] = None

        for user_id in groups.keys():

            user_data = data.loc[gb.groups[user_id]]
            user_data = user_data.sort_values(by=time_col)
            user_data['baseline_estimate'] = None

            for i, idx in enumerate(user_data.index):
                if i == 0:
                    # for first data of this user, there is no former data known
                    user_data.loc[idx, 'baseline_estimate'] = data.loc[:idx, target_name].mean()
                else:
                    if approach == 'last':
                        # last assessment of this user
                        user_data.loc[idx, 'baseline_estimate'] = user_data.iloc[i - 1][target_name]

                    if approach == 'all':
                        # all assessments of this user
                        user_data.loc[idx, 'baseline_estimate'] = user_data.iloc[:i + 1][target_name].mean()

            data.loc[user_data.index, 'baseline_estimate'] = user_data['baseline_estimate']

        return data['baseline_estimate']

    def get_baseline_assessment_prediction(self, data='None', target_name='None', approach='last',
                                           time_col='created_at'):
        """
        Gets a baseline prediction on an assessment level. Can either return the last known target of this user or all targets.
        :param data: train data of this fold
        :param target_name: name of target
        :param approach: 'all' or 'last'
        :param time_col: name of timestamp column
        :return: prediction for target at t1
        """

        data['baseline_estimate'] = None
        data = data.sort_values(by=time_col)

        for i, a_id in enumerate(data.index):

            if approach == 'last':
                if i == 0:
                    pred = data[target_name].mean()
                else:
                    pred = data.iloc[i - 1][target_name]
                data.loc[a_id, 'baseline_estimate'] = pred

            if approach == 'all':
                if i == 0:
                    # cold start problem, so take mean of all assessments
                    pred = data[target_name].mean()
                else:
                    # mean of all so far known assessments
                    pred = data.iloc[:i + 1][target_name].mean()
                data.loc[a_id, 'baseline_estimate'] = pred

        return data['baseline_estimate']


def shuffle_array(user_array, n, seed=1994):
    """
    Shuffles array of user_id into n lists.
    :param user_array: array of user ids
    :param n: number of chunks
    :return: n arrays with user ids
    """

    np.random.seed(seed)
    np.random.shuffle(user_array)
    return np.array_split(user_array, n)


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


def calc_cum_mean(df, features, user_id='user_id'):
    """
    Grouped per user, calculate a cumulative mean for each user. For the first assessment, the mean is the reported value.
    For the second assessment, the mean is the mean of the last and the current assessment.
    :param df: train df containing user_id, features, and target.
    :param features: list of feature names
    :param user_id: name of user_id column in df_train
    :return: df_train with the same shape as the input but with cumulative user-wise means in each row
    """

    # calculate
    for feature in features:
        grp = df.groupby(user_id, group_keys=False)[feature]
        df[f'{feature}_cum_sum'] = grp.apply(lambda p: p.cumsum())
        df[f'{feature}_cum_mean'] = df[f'{feature}_cum_sum'] / \
                                    grp.apply(lambda x: pd.Series(np.arange(1, len(x) + 1), x.index))
        # drop cache columns
        df.drop(columns=[f'{feature}_cum_sum', f'{feature}'], inplace=True)
        # declare grouped cumulative mean column as new feature
        df.rename(columns={f'{feature}_cum_mean': f'{feature}'}, inplace=True)

    return df


def visualize_confusion_matrix(y_test, y_pred, mapping, final_score):
    """
    Prints a confusion matrix to console.
    :param final_score: final score of this approach
    :param y_test: array of ground truth classifictions
    :param y_pred: array of predicted classes
    :param mapping: dict that maps class encodings to readable classes
    :return: None
    """
    cf_matrix = confusion_matrix(y_test, y_pred)
    cm_array_df = pd.DataFrame(cf_matrix, index=list(mapping.values()), columns=list(mapping.values()))
    sns.heatmap(cm_array_df, annot=True, cmap='Blues', fmt='')
    plt.title(f'Final score {round(final_score, 3)}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def calc_final_score(scores, y_pred_test, y_test):
    """
    Calculates the final score of an approach
    :param y_test: ground truth arrray of test set
    :param scores: np array with scores of train folds during cross validation
    :param y_pred_test: y_pred of the test set
    :return: f1 for test set and final score
    """
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    f1_final = f1_test - 0.5 * scores.std()
    return f1_test, f1_final


def fit_and_calc_score(model, X_train, X_test, y_train, y_test, scores):
    """
    fit model and all X_train, predict on y_test and calc score
    :param model:
    :param X_train:
    :param y_train:
    :return:
    """

    # refit on all training data
    model.fit(X_train, y_train)

    # evaluate on test set
    y_pred = model.predict(X_test)

    f1_test, f1_final = calc_final_score(scores, y_pred, y_test)
    # print('f1_weighted testscore: ', round(f1_test, 3))

    return y_pred, f1_final


def create_target_shift(df, target_name='target'):
    """
    Find the next target_t1 for each user and add it as a single column.
    :param df: dataframe with user_id, answer_id, created_at, features, target_t0
    :return: df with added column target_t1
    """

    # for each user, get target value of the next assessment

    df[f'{target_name}_t1'] = df.sort_values(by=['user_id', 'created_at']).groupby('user_id')[f'{target_name}'].shift(
        periods=-1, axis='index')

    # drop assessments where target is unknown
    df.dropna(subset=[f'{target_name}_t1'], inplace=True)

    return df


def cut_target(y_train, y_test, bins, LE, fit=False):
    """
    Cuts regression target into bins
    :param y_train: target series
    :param y_test: target series
    :param bins: list of bins
    :param LE: Label Encoder object
    :param fit: Whether to refit Label Encoder
    :return: y_train, y_test
    """

    # make cumberness a classification instead of regression for the target, not for the feature
    y_train = pd.cut(y_train, bins=bins, include_lowest=True)
    y_test = pd.cut(y_test, bins=bins, include_lowest=True)

    if fit:
        LE = LabelEncoder()
        y_train = LE.fit_transform(y_train)
    else:
        y_train = LE.transform(y_train)
    y_test = LE.transform(y_test)

    return y_train, y_test, LE


def create_user_dfs(data, min_assessments=10):
    """
    Group a large dataframe into sub dataframes with only one user per dataframe.
    Return those dfs in a list
    :param data: dataframe with user_id
    :param min_assessments: minimum number of assessments to retain the user_df in the list, defaults to 10
    """
    # create a dict of dfs grouped by users id
    df_groups = dict(list(data.groupby('user_id')))
    # drop user_dfs with less than 11 assessments per user
    df_groups = [df_groups[user_id] for user_id in df_groups.keys() if df_groups[user_id].shape[0] > min_assessments]

    return df_groups


def create_train_and_test_set(df):
    """
    Uses 80 % of the first users as train data and 20 % of the last users as test data.
    Users are ordered by id. That is, the smaller a user id, the longer the user participates in the study.
    :param df: dataframe with all data
    :return:  df_train, df_test
    """

    # define random state
    random_state = 1994

    # define list of users
    users_list = sorted(df.user_id.unique())
    # 20 % into train users
    s = pd.Series(users_list)
    split_idx = int(len(s) * 0.8)
    test_users = s[split_idx:].values
    train_users = s[:split_idx].values
    # unit test 1
    assert set([x for x in users_list if x not in set(test_users)]) == set(s[:split_idx].values)
    # unit test 2
    assert set(list(train_users) + list(test_users)) == set(users_list)

    # get train and test dataframe
    # use train to check approaches, use test to validate approaches'
    df_train = df[df['user_id'].isin(train_users)]
    df_test = df[df['user_id'].isin(test_users)]

    return df_train, df_test


def user_wise_missing_value_treatment(df, features, target, user_id_col='user_id'):
    """
    If a value from a user is missing, fill with the mean of all values from this user for this answer.
    :param df: dataframe containing a 'user_id' column
    :param features: list of features to treat
    :param target: name of target column
    :param user_id_col: defaults to 'user_id'

    :return: df with filled na values
    """

    # drop row if target is missing
    df.dropna(axis='index', subset=[target], inplace=True)

    grp_by = df.groupby(user_id_col)

    for feature in features:
        # this applies for users with more than one assessment
        df[feature] = df[feature].fillna(grp_by[feature].transform('mean'))

    # this applies for users with only one assessment
    for i in df.columns[df.isnull().any(axis=0)]:
        df[i].fillna(df[i].mean(), inplace=True)


    return df


def prepare_and_instantiate(df_train, df_test, features, target, bins, LE, fit=False, cut=True):
    """

    :param df_train: Whole dataframe with features, target, created_at, user_id for train rows
    :param df_test: Whole dataframe with features, target, created_at, user_id for test rows
    :param features: list of features
    :param target: name of target column
    :param bins: intervals to cut a regression target into classification, i.e. [0, 20, 40, 60, 80, 100]
    :param LE: Label Encoder object
    :return:
    """

    # missing value treatment
    df_train = user_wise_missing_value_treatment(df_train, features, target)
    df_test = user_wise_missing_value_treatment(df_test, features, target)

    # get train and test subsets for X and y
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    if cut: # some targets are already categorical and don't need to get binned
        y_train, y_test, LE = cut_target(y_train, y_test, bins, LE, fit=fit)

    # instantiate model
    model = RandomForestClassifier(random_state=1994)

    return model, X_train, X_test, y_train, y_test, LE


################################################################# tests

def test_find_schedule_pattern():
    # test find schedule pattern
    # read in a sample dataframe, uncomment to test
    df = pd.read_csv('../../data/d01_raw/ch/22-10-05_rki_stress_followup.csv')
    # df = pd.read_csv('../../data/d01_raw/ch/22-10-05_rki_parent_followup.csv')
    # df = pd.read_csv('../../data/d01_raw/ch/22-10-05_rki_heart_followup.csv')
    res = find_schedule_pattern(df, form='%Y-%m-%d %H:%M:%S', date_col_name='created_at')
    print(res)


def test_create_target_shift():
    # df = pd.read_csv('../../data/d01_raw/tyt/22-10-24_standardanswers.csv', index_col='Unnamed: 0')
    df = pd.read_csv('../../data/d01_raw/uniti/uniti_dataset_22.09.28.csv')
    # print(df.shape)
    #
    # # test target shift
    df = create_target_shift(df, target_name='cumberness')
    # print(df.shape)


def test_calc_cum_mean():
    # test cumulative mean
    cols = ['user_id', 'values', 'values2']
    test_df = df = pd.DataFrame([['A', 1, 10], ['A', 2, 20], ['A', 3, 30], ['B', 2, 20], ['B', 4, 40], ['B', 5, 50]],
                                columns=cols)
    return calc_cum_mean(test_df, features=cols[1:], user_id=cols[0])


def test_class_model(df):
    # test baseline approach
    target_name = 'cumberness'
    model = baseline_model()
    for approach in ['last', 'all']:
        # pred_series = model.get_baseline_user_prediction(data=df, target_name=target_name, approach=approach)
        pred_series = model.get_baseline_assessment_prediction(data=df, target_name=target_name, approach=approach)
        print(approach, '\t', pred_series)



def test_visualize_confusion_matrix():
    y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_test = [1, 0, 0, 2, 1, 1, 3, 2, 2]
    final_score = .8
    mapping = {0: 'Class 0',
               1: 'Class 1',
               2: 'Class 2',
               3: 'Class 3'}

    visualize_confusion_matrix(y_test, y_pred, mapping, final_score)


def main():
    df = pd.read_csv('../../data/d02_processed/cc.csv', index_col='answer_id')

    # first 80 % of users into train, second 20 % into test
    df_train, df_test = create_train_and_test_set(df)

    features = None #TODO: tbd
    target = 'corona_result'
    time_col = 'created_at'

    # No Label encoding required but some functions need it as arguments
    LE = None
    bins = None

    # preare dataset and model
    model, X_train, X_test, y_train, y_test, _ = prepare_and_instantiate(df_train, df_test, features, target,
                                                                                 bins, LE, fit=False, cut=False)


if __name__ == '__main__':
    main()
