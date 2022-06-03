import pandas as pd
import datetime as dt
from sklearn.utils import resample
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def clean_2(dirty_df):
    dirty_df.dropna(subset=['linqmap_type', 'update_date'], inplace=True)

    dirty_df['pubDate'] = dirty_df.apply(lambda row: dt.datetime.strptime(row.pubDate, "%m/%d/%Y %H:%M:%S"), axis=1)
    dirty_df['update_date'] = dirty_df.apply(lambda row: dt.datetime.fromtimestamp(row.update_date / 1000), axis=1)
    df1 = dirty_df.loc[:, ['update_date', 'linqmap_type']].copy(deep=True)
    df2 = dirty_df.loc[:, ['pubDate', 'linqmap_type']].copy(deep=True)
    df2['update_date'] = dirty_df['pubDate'].copy(deep=True)
    df2.drop(columns='pubDate', inplace=False)

    dirty_df = pd.concat([df1, df2], axis=0)
    dirty_df['day_of_week'] = dirty_df.apply(lambda row: row.update_date.weekday(), axis=1)
    dirty_df['time_slot'] = dirty_df.apply(lambda row: row.update_date.hour, axis=1)

    dums = pd.get_dummies(dirty_df["linqmap_type"])
    dirty_df = pd.concat([dirty_df, dums], axis=1)

    index = ((8 <= dirty_df['time_slot']) & (dirty_df['time_slot'] < 10)) | \
            ((12 <= dirty_df['time_slot']) & (dirty_df['time_slot'] < 14)) | \
            ((18 <= dirty_df['time_slot']) & (dirty_df['time_slot'] < 20))

    dirty_df = dirty_df[index].copy()

    slots_dict = {8: 'morning', 9: 'morning', 12: 'afternoon', 13: 'afternoon', 18: 'evening', 19: 'evening'}
    dirty_df['time_slot'].replace(slots_dict, inplace=False)

    # dirty_df.to_csv('test.csv')
    return dirty_df


def get_empirical_mean_by_days_and_time_slots(df, n_bootstrap=10000):
    resampled = resample(df)
    epirical_mean = (resampled.groupby(["day_of_week", 'time_slot']).sum()) / n_bootstrap
    for_plot_1_morning_ACCIDENT = []
    for_plot_0_afternoon_jam = []
    for i in range(1, n_bootstrap):
        resampled = resample(df)
        epirical_mean += (resampled.groupby(["day_of_week", 'time_slot']).sum()) / n_bootstrap
        for_plot_1_morning_ACCIDENT.append(epirical_mean.loc[(1, 'morning'), 'ACCIDENT'].copy() * (n_bootstrap / i))
        for_plot_0_afternoon_jam.append(epirical_mean.loc[(0, 'afternoon'), 'JAM'].copy() * (n_bootstrap / i))

    epirical_mean.loc[(0, 'evening'), :] = (pd.NA, pd.NA, pd.NA, pd.NA)
    epirical_mean.loc[(4, 'evening'), :] = (pd.NA, pd.NA, pd.NA, pd.NA)
    epirical_mean.loc[(4, 'morning'), :] = (pd.NA, pd.NA, pd.NA, pd.NA)
    epirical_mean.loc[(5, 'afternoon'), :] = (pd.NA, pd.NA, pd.NA, pd.NA)
    epirical_mean.loc[(5, 'morning'), :] = (pd.NA, pd.NA, pd.NA, pd.NA)
    epirical_mean.fillna(epirical_mean.mean().round(1), inplace=False)
    return epirical_mean, for_plot_0_afternoon_jam, for_plot_1_morning_ACCIDENT


def main():
    df = pd.read_csv('waze_data.csv')
    df = clean_2(df)
    n_bootstrap = 150
    estimation, afternoon_0_jam, morning_1_ACCIDENT = get_empirical_mean_by_days_and_time_slots(df, n_bootstrap)
    estimation.to_csv('slots_estimation_test.csv')
    fig = make_subplots(rows=1, cols=1, subplot_titles=[''])
    fig.add_trace(go.Scatter(y=afternoon_0_jam, x=list(range(1, n_bootstrap - 1)), mode='lines'), row=1, col=1)
    fig.write_html('afternoon_0_jam.html', auto_open=False)

    fig = make_subplots(rows=1, cols=1, subplot_titles='')
    fig.add_trace(go.Scatter(y=morning_1_ACCIDENT, x=list(range(1, n_bootstrap - 1)), mode='lines'), row=1, col=1)
    fig.write_html('morning_1_ACCIDENT.html', auto_open=False)


if __name__ == '__main__':
    main()
