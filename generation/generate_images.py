from pathlib import Path
import pandas as pd
from pprint import pprint


def main():
    data_directory = Path(__file__).parent.joinpath('data')
    item_df = pd.read_parquet(data_directory.joinpath('item_df_base.parquet'))
    pprint(item_df)


if __name__ == '__main__':
    main()