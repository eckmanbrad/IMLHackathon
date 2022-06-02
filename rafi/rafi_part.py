import pandas as pd
import openpyxl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px






def get_comb(df):
    combs = []
    for frst in range(len(df)):
        print(frst)
        for sec in range(frst, len(df)):
            for thrd in range(sec, len(df)):
                for fourth in range(thrd, len(df)):
                    combs.append([frst, sec, thrd, fourth])
    return combs


def main():
    df = pd.read_csv('waze_data.csv')
    # df.to_excel('wazeee.xlsx')

    df = clean(df)
    df.to_csv('test.csv')
    # get_comb(df)

    print('all ok')

    print(df.columns)
    print(set(df['linqmap_city']))
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
