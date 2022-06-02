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

    # create saved df file of types and subtype
    subtypes = dirty_df.groupby('linqmap_type').aggregate({'linqmap_subtype': np.unique})
    subtypes.to_csv('subtypes.csv')

    dirty_df.sort_values(by='update_date', inplace=True)

    # remove magvar & pubdate & street colb
    dirty_df.drop(columns=['linqmap_magvar', 'pubDate', 'linqmap_street', 'update_date', 'linqmap_type'], inplace=True)
    # add dummies for subtype
    dums = pd.get_dummies(dirty_df['linqmap_subtype'])
    dirty_df = pd.concat([dirty_df, dums], axis=1)
    return dirty_df


def get_comb(df):
    new_cols = []
    for i in range(1, 6):
        new_cols += list(df.columns + '_S' + str(i))
    with open('learn_df.csv', 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_cols)

        for i in range(df.shape[0] - 4):
            print('\r', str(round(i / (df.shape[0] - 4) * 100, 2)) + ' %', end='')
            cur_row = []
            cur_row += list(df.iloc[i, :])
            cur_row += list(df.iloc[i + 1, :])
            cur_row += list(df.iloc[i + 2, :])
            cur_row += list(df.iloc[i + 3, :])
            cur_row += list(df.iloc[i + 4, :])
            writer_object.writerow(cur_row)
        print('\r100 %')
    f_object.close()
    temp = pd.read_csv('learn_df.csv')

    cols_to_remove = []
    for col in temp.columns:
        if col.endswith('_S5'):
            cols_to_remove.append(col)
    cols_to_remove.remove('linqmap_subtype_S5')
    cols_to_remove.remove('x_S5')
    cols_to_remove.remove('y_S5')
    print(cols_to_remove)

    temp.drop(columns=['linqmap_subtype_S1', 'linqmap_subtype_S2', 'linqmap_subtype_S3',
                       'linqmap_subtype_S4'] + cols_to_remove, inplace=True)
    return temp


def get_comb_for_pred(df):
    new_cols = []
    for i in range(1, 5):
        new_cols += list(df.columns + '_S' + str(i))
    with open('to_predict.csv', 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_cols)

        for i in range(0, df.shape[0] - 3, 4):
            print('\r', str(round(i / (df.shape[0] - 4) * 100, 2)) + ' %', end='')
            cur_row = []
            cur_row += list(df.iloc[i, :])
            cur_row += list(df.iloc[i + 1, :])
            cur_row += list(df.iloc[i + 2, :])
            cur_row += list(df.iloc[i + 3, :])
            writer_object.writerow(cur_row)
        print('\r100 %')
    f_object.close()
    temp = pd.read_csv('to_predict.csv')
    return temp


def split_X_y_type_pred(df):
    X = df.copy()
    responses = ['linqmap_subtype_S5', 'x_S5', 'y_S5']

    y = df.loc[:, responses]
    X.drop(columns=responses, inplace=True)

    dict_to_replace = dict()
    c = 1
    for i in np.unique(y['linqmap_subtype_S5']):
        dict_to_replace[i] = c
        c += 1
    y['linqmap_subtype_S5'].replace(dict_to_replace, inplace=True)
    pd.DataFrame(dict_to_replace, index=range(len(dict_to_replace))).to_csv('response_type_dict.csv')

    return X, y['linqmap_subtype_S5']


def split_X_y_cord_pred(df):
    X = df.copy()
    responses = ['linqmap_subtype_S5', 'x_S5', 'y_S5']
    y = df.loc[:, responses]
    X.drop(columns=responses, inplace=True)

    return X, y[:, 'x_S5', 'y_S5']
