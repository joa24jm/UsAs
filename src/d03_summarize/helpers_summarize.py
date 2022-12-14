# Author Johannes Allgaier
import os
import sys

sys.path.insert(0, "../..")
import pandas as pd
from src.d00_utils import helpers


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


def main():
    path = '../../results/tables/approaches'

    dfs = load_approach_tables(path, save=True)

    foo = 1


if __name__ == '__main__':
    main()
