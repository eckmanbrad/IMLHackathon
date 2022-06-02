import numpy as np
import pandas as pd
import datetime as dt


def clean_2(dirty_df):
    dirty_df.dropna(subset=['linqmap_type', 'update_date'], inplace=True)

    dirty_df['pubDate'] = dirty_df.apply(lambda row: dt.datetime.strptime(row.pubDate, "%m/%d/%Y %H:%M:%S"), axis=1)
    dirty_df['update_date'] = dirty_df.apply(lambda row: dt.datetime.fromtimestamp(row.update_date / 1000), axis=1)

    print(dirty_df.shape)
    df1 = dirty_df.loc[:, ['update_date', 'linqmap_type']].copy(deep=True)
    df2 = dirty_df.loc[:, ['pubDate', 'linqmap_type']].copy(deep=True)
    df2['update_date'] = dirty_df['pubDate'].copy(deep=True)
    df2.drop(columns='update_date', inplace=True)
    dirty_df = pd.concat([df1, df2], axis=0)

    dirty_df['day_of_week'] = dirty_df.apply(lambda row: row.update_date.weekday(), axis=1)
    dirty_df['time_slot'] = dirty_df.apply(lambda row: row.update_date.hour, axis=1)

    index = ((8 <= dirty_df['time_slot']) & (dirty_df['time_slot'] < 10)) | \
            ((12 <= dirty_df['time_slot']) & (dirty_df['time_slot'] < 14)) | \
            ((18 <= dirty_df['time_slot']) & (dirty_df['time_slot'] < 20))

    dirty_df = dirty_df[index].copy()

    slots_dict = {8: 'morning', 9: 'morning', 12: 'afternoon', 13: 'afternoon', 18: 'evening', 19: 'evening'}
    dirty_df['time_slot'].replace(slots_dict, inplace=True)

    dirty_df.to_csv('test.csv')
    print(dirty_df)
    return dirty_df


def main():
    df = pd.read_csv('waze_data.csv')
    df = clean_2(df)


if __name__ == '__main__':
    main()
