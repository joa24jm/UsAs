# Author Johannes Allgaier
import os
import sys

sys.path.insert(0, "../..")
import pandas as pd
from src.d00_utils import helpers

def get_dataset_names():
    return ['cc', 'ch_stress', 'rki_children', 'rki_heart', 'rki_parent', 'tyt', 'uniti']

def format_f1_and_std(df, col):
    """
    :param df: pandas dataframe
    :param col: name of column
    """

    return df[f'{col}_f1'].map('{:.3f}'.format) + " (" + df[f'{col}_std'].map('{:.3f}'.format) + ")"


def load_approach_tables(path='../../results/tables/approaches', save=False):
    """
    :param path:
    :return:
    """
    files = [file for file in os.listdir(path) if '.csv' in file]
    idxs = helpers.get_approaches()
    dfs = pd.DataFrame(index=idxs)
    for file in files:
        df = pd.read_csv(path + '/' + file,
                         index_col='approach')
        name = file.split('.')[0]
        df.columns = [f'{name}_{col}' for col in df.columns]
        dfs = pd.concat([dfs, df], axis=1)

    if save:
        dfs.to_excel(path + '/approaches.xlsx')

    foo = 1

    return dfs


def rank_approaches(path='../../results/tables/approches/approaches.xlsx'):
    df = pd.read_excel(path)

    
def prepare_results(df):

    # prepare data
    idxs = [idx for idx in df.transpose().index if 'f1' in idx]
    df_ranks = df.rank(ascending=False).transpose().loc[idxs, :]

    # calculate results
    result = pd.DataFrame(df_ranks.mean().apply(lambda x: round(x, 2)))
    result.rename(columns={0:'average_rank'}, inplace=True)
    result['average_rank_std'] = df_ranks.std().apply(lambda x: round(x, 2))

    return result.sort_values(by='average_rank')
    

def main():
    path = '../../results/tables/approaches'

    dfs = load_approach_tables(path, save=True)

    foo = 1


if __name__ == '__main__':
    main()

