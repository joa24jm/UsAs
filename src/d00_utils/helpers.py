# Author Johannes Allgaier

# imports
import pandas as pd
from datetime import date, datetime
import numpy as np

def find_mode_period_length(df, form = '%Y-%m-%d %H:%M:%S', date_col_name='created_at'):
    """
    Takes a dataframe df and returns a dict that describes the duration of two filled out
    assessments of one user.
    :param df: dataframe that contains assessments of all users
           form: Format of the time stamp of the date column
           date_col_name: Name of the column containing the collection time stamp
    :return: dict like {hours: , days: , weeks: }
    """
    # find most frequent user mfu
    mfu = df.user_id.value_counts().index[0]

    # reduce dataframe to most frequent user
    df = df[df.user_id == mfu]


    # find gap of two assessments
    form = form

    # for aggregation
    hours, days = list(), list()

    for i in np.arange(0, df.shape[0]-1):

        date_start = df[date_col_name].iloc[i]
        date_start = datetime.strptime(date_start, form)
        date_end = df[date_col_name].iloc[i+1]
        date_end = datetime.strptime(date_end, form)

        delta = date_end-date_start

        hours.append(delta.total_seconds()/3600)
        days.append(delta.total_seconds()/3600/24)

    return {'avg hours between two assessments':np.array(hours).mean(), # average length between two filled out assessments in hours
            'avg days between two assessments':np.array(days).mean(),   # average length between two filled out assessments in days
            'std_hours': np.array(hours).std(), # std of length between two filled out assessments in hours
            'std_days': np.array(days).std()}   # std of length between two filled out assessments in days

def main():

    # read in a sample dataframe, uncomment to test
    # df = pd.read_csv('../../data/d01_raw/ch/22-07-01_rki_stress_followup.csv')
    # df = pd.read_csv('../../data/d01_raw/ch/22-07-01_rki_parent_followup.csv')
    df = pd.read_csv('../../data/d01_raw/ch/22-07-01_rki_heart_followup.csv')

    res = find_mode_period_length(df, form='%Y-%m-%d %H:%M:%S', date_col_name='created_at')

    print(res)


if __name__ == '__main__':
    main()
