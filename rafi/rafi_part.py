import pandas as pd
import openpyxl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def clean(dirty_df):
    # remove unwanted cols
    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments']
    dirty_df.drop(columns=cols_to_remove, inplace=True)

    # remove nans from chosen cols
    dirty_df.dropna(subset=['linqmap_type', 'linqmap_subtype', 'linqmap_reportRating', 'linqmap_roadType',
                            'linqmap_magvar', 'linqmap_reliability'], inplace=True)

    # add NS & WE cols
    dirty_df['NS'] = np.where((dirty_df['linqmap_magvar'] < 90) | (dirty_df['linqmap_magvar'] > 270), 1, 0)
    dirty_df['EW'] = np.where((dirty_df['linqmap_magvar'] > 180) | (dirty_df['linqmap_magvar'] < 361), 1, 0)

    # remove magvar col
    dirty_df.drop(columns='linqmap_magvar', inplace=True)
    return dirty_df


def main():
    df = pd.read_csv('waze_data.csv')
    df = clean(df)
    print('all ok')

    print(df.columns)
    # print(set(df['linqmap_city']))
    # print(set(df['linqmap_street']))
    # print(sum(df['linqmap_subtype'].isna()))

    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments']

    subtypes = df.groupby('linqmap_type').aggregate({'linqmap_subtype': np.unique})
    subtypes.to_csv('subtypes.csv')

    # new map
    df['str_roadType'] = df.loc[:, 'linqmap_roadType'].astype(str)
    mapp = go.Figure(layout=dict(title=dict(text='map')))
    mapp.add_scatter(go.Scatter())
    # px.scatter(df, x='x', y='y', color='str_roadType')
    # mapp.write_html('map.html', auto_open=False)

    # map of roads
    # df['str_roadType'] = df.loc[:, 'linqmap_roadType'].astype(str)
    # mapp = px.scatter(df, x='x', y='y', color='str_roadType')
    # mapp.write_html('map.html', auto_open=False)


if __name__ == '__main__':
    # pd.set_option('display.max_rows', None)  # number of df rows to show
    # pd.set_option('display.max_columns', None)  # number of df columns to show
    main()
