import pandas as pd
import openpyxl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def clean(df):
    df['NS'] = 0
    df['EW'] = 0
    df['NS'][np.where(df['linqmap_magvar'] < 90 or df['linqmap_magvar'] > 270)] = 1
    df['EW'][np.where(df['linqmap_magvar'] > 90 and df['linqmap_magvar'] < 270)] = 1

    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments', 'linqmap_magvar']

    df.drop(columns=cols_to_remove)
    return df


def main():
    df = pd.read_csv('waze_data.csv')
    print(df.columns)
    print(set(df['linqmap_city']))
    print(set(df['linqmap_street']))
    # print(sum(df['linqmap_subtype'].isna()))

    cols_to_remove = ['OBJECTID', 'linqmap_reportDescription', 'linqmap_nearby', 'linqmap_reportMood',
                      'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments']

    subtypes = df.groupby('linqmap_type').aggregate({'linqmap_subtype': np.unique})
    print(subtypes)

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
