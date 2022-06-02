import pandas as pd
import numpy as np
import datetime as dt
from csv import writer
from sklearn.impute import KNNImputer
import pickle



def clean_pred(dirty_df):
    # get tlv rows
    dirty_df = dirty_df[dirty_df['linqmap_city'] == 'תל אביב - יפו'].copy()

    # remove unwanted cols
    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments', 'linqmap_city']
    dirty_df.drop(columns=cols_to_remove, inplace=True)

    # fill nans
    dirty_df.fillna(method='ffill', inplace=True)
    dirty_df.fillna(method='bfill', inplace=True)

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


def get_comb_for_pred(df):
    new_cols = []
    for i in range(1, 5):
        new_cols += list(df.columns + '_S' + str(i))
    with open('to_predict.csv', 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_cols)
        for i in range(0, df.shape[0], 4):
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


def preprocess(dirty_df):
    dirty_df = clean_pred(dirty_df)
    return get_comb_for_pred(dirty_df)


def run_1(data_path):
    df = preprocess(pd.read_csv(data_path))

    # load the model from disk
    loaded_model = pickle.load(open('finalized_knn_model.sav', 'rb'))


    # read trained model
    # output results
    return
