import datetime as dt
import numpy as np
from csv import writer
import pandas as pd


def clean(dirty_df):
    # get tlv rows
    dirty_df = dirty_df[dirty_df['linqmap_city'] == 'תל אביב - יפו'].copy()

    # remove unwanted cols
    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments', 'linqmap_city']
    dirty_df.drop(columns=cols_to_remove, inplace=True)

    # remove nans from chosen cols
    dirty_df.dropna(subset=['linqmap_type', 'linqmap_subtype', 'linqmap_reportRating', 'linqmap_roadType',
                            'linqmap_magvar', 'linqmap_reliability', 'linqmap_street'], inplace=True)

    # add NS & WE cols
    dirty_df['NS'] = np.where((dirty_df['linqmap_magvar'] < 90) | (dirty_df['linqmap_magvar'] > 270), 1, 0)
    dirty_df['EW'] = np.where((dirty_df['linqmap_magvar'] < 180), 1, 0)

    # convert date & time & timestamp to datetime object
    dirty_df['pubDate'] = dirty_df.apply(lambda row: dt.datetime.strptime(row.pubDate, "%m/%d/%Y %H:%M:%S"), axis=1)
    dirty_df['update_date'] = dirty_df.apply(lambda row: dt.datetime.fromtimestamp(row.update_date / 1000), axis=1)
    dirty_df['event_time_hours'] = dirty_df.apply(lambda row: (row.update_date - row.pubDate).total_seconds() / 3600,
                                                  axis=1)

    # add dummies for street
    dums = pd.get_dummies(dirty_df['linqmap_street'])
    dirty_df = pd.concat([dirty_df, dums], axis=1)

    # remove magvar & pubdate & street col
    dirty_df.drop(columns=['linqmap_magvar', 'pubDate', 'linqmap_street'], inplace=True)

    dirty_df.sort_values(by='update_date', inplace=True)
    return dirty_df


def get_comb(df):
    new_cols = []
    for i in range(1, 6):
        new_cols += list(df.columns + '_X' + str(i))
    with open('learn_df.csv', 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_cols)

        for i in range(df.shape[0] - 4):
            print('\r', str(i / df.shape[0] - 4) + ' %', end='')
            cur_row = []
            cur_row += list(df.loc[i, :])
            cur_row += list(df.loc[i + 1, :])
            cur_row += list(df.loc[i + 2, :])
            cur_row += list(df.loc[i + 3, :])
            cur_row += list(df.loc[i + 4, :])
            writer_object.writerow(cur_row)
        print('100 %')
    f_object.close()
    return pd.read_csv('learn_df.csv')
