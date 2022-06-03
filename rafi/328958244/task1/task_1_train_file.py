import time
import pickle
import numpy as np
import datetime as dt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, mean_squared_error
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso, Ridge, LinearRegression
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
    dirty_df['linqmap_street'] = np.where(dirty_df['linqmap_street'] == 'שלום עליכם', 'סירקין',
                                          dirty_df['linqmap_street'])

    # add dummies for street
    dums = pd.get_dummies(dirty_df['linqmap_street'])
    dirty_df = pd.concat([dirty_df, dums], axis=1)

    # create saved df file of types and subtype
    subtypes = dirty_df.groupby('linqmap_type').aggregate({'linqmap_subtype': np.unique})
    subtypes.to_csv('subtypes.csv')

    dirty_df.sort_values(by='update_date', inplace=True)
    # add dummies for subtype
    dums = pd.get_dummies(dirty_df['linqmap_subtype'])
    dirty_df = pd.concat([dirty_df, dums], axis=1)

    x_cord, y_cord = dirty_df['x'], dirty_df['y']

    # remove magvar & pubdate & street colb
    dirty_df.drop(columns=['linqmap_magvar', 'pubDate', 'linqmap_street', 'update_date', 'linqmap_type'],
                  inplace=True)

    dirty_df.to_csv('test2.csv')
    return dirty_df


def get_comb_reg(df):
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

    temp.drop(columns=['linqmap_subtype_S1', 'linqmap_subtype_S2', 'linqmap_subtype_S3',
                       'linqmap_subtype_S4'] + cols_to_remove, inplace=True)
    temp = temp.reindex(sorted(temp.columns), axis=1)
    return temp


def get_comb_knn(df):
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


    temp.drop(columns=['linqmap_subtype_S1', 'linqmap_subtype_S2', 'linqmap_subtype_S3',
                       'linqmap_subtype_S4'] + cols_to_remove, inplace=True)
    temp = temp.reindex(sorted(temp.columns), axis=1)
    return temp


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
    temp = temp.reindex(sorted(temp.columns), axis=1)
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
    return X, y.loc[:, ['x_S5', 'y_S5']].copy()


def cross_validate(estimator, X: np.ndarray, y: np.ndarray, scoring, cv: int = 5):
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_loss_lst = []
    test_loss_lst = []
    m = len(y)
    for i in range(cv):
        train_x = np.concatenate((X[:int(m * (i / cv))], X[int(m * ((i + 1) / cv)):]), axis=0)
        train_y = np.concatenate((y[:int(m * (i / cv))], y[int(m * ((i + 1) / cv)):]), axis=0)
        test_x = X[int(m * (i / cv)):int(m * ((i + 1) / cv))]
        test_y = y[int(m * (i / cv)):int(m * ((i + 1) / cv))]

        estimator.fit(train_x, train_y)
        train_loss_lst.append(scoring(train_y, estimator.predict(train_x)))
        test_loss_lst.append(scoring(test_y, estimator.predict(test_x)))
    return np.average(train_loss_lst), np.average(test_loss_lst)


def our_knn(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=24)

    loss_lst = []
    for i in range(2, 15):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(train_x, train_y)

        y_predict = neigh.predict(test_x)
        loss_lst.append(f1_score(test_y.to_numpy(), y_predict, average='weighted'))
    k = np.argmax(loss_lst) + 2
    elbow = px.line(x=list(range(2, 15)), y=loss_lst,
                    title='k =' + str(k) + ', F1: ' + str(round(loss_lst[k - 1], 3)),
                    labels={'index': 'k', 'value': 'F1'})
    elbow.write_html('knn_elbow.html', auto_open=False)

    print('\nk: ', k)
    print('knn F1: ', round(loss_lst[k - 1], 3))
    kneigh = KNeighborsClassifier(n_neighbors=k)
    kneigh.fit(X, y)

    pickle.dump(kneigh, open('finalized_knn_model.sav', 'wb'))
    return


def select_regularization_parameter(X, y_x, y_y, y_x_lam_range, y_y_lam_range, n_evaluations: int = 50):
    r_train_score_lst, r_test_score_lst, l_train_score_lst, l_test_score_lst = [], [], [], []
    loss_func = lambda y_true, y_pred: mean_squared_error(y_true, y_pred)

    fig = make_subplots(rows=2, cols=1, subplot_titles=['x - lasso', 'y - lasso'],
                        horizontal_spacing=0.01, vertical_spacing=0.08)

    train_x, test_x, train_y, test_y = train_test_split(X, y_x, test_size=0.25, random_state=24)

    for i in y_x_lam_range:
        print('\r' + str(round(100 * i / y_x_lam_range[-1])) + ' %', end='')
        r_train_loss, r_test_loss = cross_validate(Lasso(alpha=i), train_x, train_y, loss_func)
        r_train_score_lst.append(r_train_loss)
        r_test_score_lst.append(r_test_loss)
    print('\r100 %')
    train_x, test_x, train_y, test_y = train_test_split(X, y_y, test_size=0.25, random_state=24)
    for i in y_y_lam_range:
        print('\r' + str(round(100 * i / y_y_lam_range[-1])) + ' %', end='')
        l_train_loss, l_test_loss = cross_validate(Lasso(alpha=i, max_iter=10000), train_x, train_y, loss_func)
        l_train_score_lst.append(l_train_loss)
        l_test_score_lst.append(l_test_loss)

    fig.add_trace(go.Scatter(x=y_x_lam_range, y=r_train_score_lst, mode='lines', name='ridge avrg train_score'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=y_x_lam_range, y=r_test_score_lst, mode='lines', name='ridge avrg test score'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=y_y_lam_range, y=l_train_score_lst, mode='lines', name='lasso avrg train_score'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=y_y_lam_range, y=l_test_score_lst, mode='lines', name='lasso avrg test score'), row=2,
                  col=1)
    fig.write_html(r'Ridge_lasso_regularization.html', auto_open=False)

    rid_lam = np.argmin(r_test_score_lst)
    lasso_lam = np.argmin(l_test_score_lst)

    with open('MODEL_SUMMARY.txt', 'w') as f:
        f.write(str(dt.datetime.now().date()))
        print('\nridge lambda:', y_x_lam_range[rid_lam], '\nlasso lambda:', y_y_lam_range[lasso_lam])
        f.write('\nridge lambda: ' + str(y_x_lam_range[rid_lam]))
        f.write('\nlasso lambda: ' + str(y_y_lam_range[lasso_lam]))
        print('cv ridge loss:', r_test_score_lst[rid_lam], '\ncv lasso loss:', l_test_score_lst[lasso_lam])
        f.write('\ncv ridge loss: ' + str(r_test_score_lst[rid_lam]))
        f.write('\ncv lasso loss: ' + str(l_test_score_lst[lasso_lam]))
        my_ridge_loss = loss_func(test_y, Ridge(y_x_lam_range[rid_lam]).fit(train_x, train_y).predict(test_x))
        my_lasso = Lasso(y_y_lam_range[lasso_lam], max_iter=10000).fit(train_x, train_y).predict(test_x)
        my_lasso_loss = loss_func(test_y, my_lasso)
        my_reg_loss = loss_func(test_y, LinearRegression().fit(train_x, train_y).predict(test_x))
        print('ridge loss:', my_ridge_loss, '\nlasso loss:', my_lasso_loss, '\nreg loss:', my_reg_loss)
        f.write('\nridge loss: ' + str(my_ridge_loss))
        f.write('\nlasso loss: ' + str(my_lasso_loss))
        f.write('\nregular regression loss: ' + str(my_reg_loss))
        f.close()
    return


def main():
    df = pd.read_csv('waze_data.csv')
    clean_df = clean(df)
    learn_df_reg = get_comb_reg(clean_df)
    learn_df_reg.to_csv('learn_df.csv')
    time.sleep(0.2)
    df = pd.read_csv('learn_df.csv', index_col='Unnamed: 0')

    X, y = split_X_y_cord_pred(df)
    y_x = y.drop(columns='y_S5')
    y_y = y.drop(columns='x_S5')

    # select_regularization_parameter(X.to_numpy(), y_x.to_numpy(), y_y.to_numpy(),
    #                                 np.linspace(1, 10, 300), np.linspace(1, 10, 300))

    x_model = Lasso(alpha=3.2864321608040203).fit(X, y_x)
    y_model = Lasso(alpha=3.7386934673366836).fit(X, y_y)
    pickle.dump(x_model, open('finalized_reg_x_model.sav', 'wb'))
    pickle.dump(y_model, open('finalized_reg_y_model.sav', 'wb'))

    learn_df_knn = get_comb_knn(clean_df)
    learn_df_knn.to_csv('learn_df_knn.csv')
    time.sleep(0.2)
    df = pd.read_csv('learn_df_knn.csv', index_col='Unnamed: 0')
    X, y = split_X_y_type_pred(df)
    our_knn(X, y)


if __name__ == '__main__':
    np.random.seed(0)
    # pd.set_option('display.max_rows', None)  # number of df rows to show
    # pd.set_option('display.max_columns', None)  # number of df columns to show
    main()
