import pandas as pd
from sklearn import tree
from final_functions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, mean_squared_error
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso, Ridge, LinearRegression


def cross_validate(estimator, X: np.ndarray, y: np.ndarray,scoring, cv: int = 5):
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
    k = np.argmax(loss_lst) + 1
    elbow = px.line(loss_lst, title='k =' + str(k) + ', F1: ' + str(round(loss_lst[k - 1], 3)),
                    labels={'index': 'k', 'value': 'F1'})
    elbow.write_html('knn_elbow.html', auto_open=True)

    print('\nk: ', k)
    print('knn missclasification: ', round(loss_lst[k - 1], 3))
    kneigh = KNeighborsClassifier(n_neighbors=k)
    kneigh.fit(X, y)
    return


def our_rf(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=24)

    loss_lst = []
    rf = RandomForestClassifier(max_depth=40).fit(train_x, train_y)
    test_y_predict = rf.predict(test_x)
    miss = f1_score(test_y.to_numpy(), test_y_predict, average='weighted')

    elbow = px.scatter(loss_lst, title='depth = 6' + ', F1: ' + str(round(miss, 3)),
                       labels={'index': 'depth', 'value': 'F1'})
    elbow.write_html('rf.html', auto_open=False)

    res = pd.DataFrame({'predicted': test_y_predict, 'real': test_y})
    figres = px.scatter(res, title='F1: ' + str(round(miss, 2)),
                        labels={'index': 'trade', 'value': 'label'})
    figres.write_html('random_forest.html', auto_open=True)


# def our_tree_part(X, y):
#     train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=24)
#
#     loss_lst = []
#     rf = RandomForestClassifier(max_depth=40).fit(train_x, train_y)
#     test_y_predict = rf.predict(test_x)
#     miss = f1_score(test_y.to_numpy(), test_y_predict, average='weighted')
#
#     elbow = px.scatter(loss_lst, title='depth = 6' + ', F1: ' + str(round(miss, 3)),
#                        labels={'index': 'depth', 'value': 'F1'})
#     elbow.write_html('rf.html', auto_open=False)
#
#     res = pd.DataFrame({'predicted': test_y_predict, 'real': test_y})
#     figres = px.scatter(res, title='misclassification_error: ' + str(round(miss, 2)),
#                         labels={'index': 'trade', 'value': 'label'})
#     figres.write_html('final.html', auto_open=True)


def select_regularization_parameter(X, y, r_lam_range, l_lam_range, n_evaluations: int = 50):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms

    r_lam_range, l_lam_range are numpy linspace objects with range for lambdas
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=24)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    r_train_score_lst, r_test_score_lst, l_train_score_lst, l_test_score_lst = [], [], [], []
    loss_func = lambda y_true, y_pred: mean_squared_error(y_true, y_pred)

    fig = make_subplots(rows=2, cols=1, subplot_titles=['ridge', 'lasso'],
                        horizontal_spacing=0.01, vertical_spacing=0.08)
    for i in r_lam_range:
        print('\r' + str(round(100 * i / r_lam_range[-1])) + ' %', end='')
        r_train_loss, r_test_loss = cross_validate(Lasso(alpha=i), train_x, train_y, loss_func)
        r_train_score_lst.append(r_train_loss)
        r_test_score_lst.append(r_test_loss)
    print('\r100 %')
    for i in l_lam_range:
        print('\r' + str(round(100 * i / l_lam_range[-1])) + ' %', end='')
        l_train_loss, l_test_loss = cross_validate(Lasso(alpha=i, max_iter=10000), train_x, train_y, loss_func)
        l_train_score_lst.append(l_train_loss)
        l_test_score_lst.append(l_test_loss)

    fig.add_trace(go.Scatter(x=r_lam_range, y=r_train_score_lst, mode='lines', name='ridge avrg train_score'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=r_lam_range, y=r_test_score_lst, mode='lines', name='ridge avrg test score'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=l_lam_range, y=l_train_score_lst, mode='lines', name='lasso avrg train_score'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=l_lam_range, y=l_test_score_lst, mode='lines', name='lasso avrg test score'), row=2,
                  col=1)
    fig.write_html(r'Ridge_lasso_regularization.html', auto_open=True)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    rid_lam = np.argmin(r_test_score_lst)
    lasso_lam = np.argmin(l_test_score_lst)
    with open('MODEL_SUMMARY.txt', 'w') as f:
        f.write(str(dt.datetime.now().date()))
        print('\nridge lambda:', r_lam_range[rid_lam], '\nlasso lambda:', l_lam_range[lasso_lam])
        f.write('\nridge lambda: ' + str(r_lam_range[rid_lam]))
        f.write('\nlasso lambda: ' + str(l_lam_range[lasso_lam]))
        print('cv ridge loss:', r_test_score_lst[rid_lam], '\ncv lasso loss:', l_test_score_lst[lasso_lam])
        f.write('\ncv ridge loss: ' + str(r_test_score_lst[rid_lam]))
        f.write('\ncv lasso loss: ' + str(l_test_score_lst[lasso_lam]))
        my_ridge_loss = loss_func(test_y, Ridge(r_lam_range[rid_lam]).fit(train_x, train_y).predict(test_x))
        my_lasso = Lasso(l_lam_range[lasso_lam], max_iter=10000).fit(train_x, train_y).predict(test_x)
        my_lasso_loss = loss_func(test_y, my_lasso)
        my_reg_loss = loss_func(test_y, LinearRegression().fit(train_x, train_y).predict(test_x))
        print('ridge loss:', my_ridge_loss, '\nlasso loss:', my_lasso_loss, '\nreg loss:', my_reg_loss)
        f.write('\nridge loss: ' + str(my_ridge_loss))
        f.write('\nlasso loss: ' + str(my_lasso_loss))
        f.write('\nregular regression loss: ' + str(my_reg_loss))
        f.close()
    return


def main():
    # df = pd.read_csv('waze_data.csv')
    # df = clean(df)
    #
    # learn_df = get_comb(df)
    # learn_df.to_csv('learn_df.csv')
    df = pd.read_csv('learn_df.csv', index_col='Unnamed: 0')
    X, y = split_X_y_type_pred(df)
    # our_knn(X, y)
    # our_rf(X, y)
    # our_tree_part(X, y)

    X,y = split_X_y_cord_pred(df)
    y_x = y.drop(columns='y_S5')
    y_y = y.drop(columns='x_S5')

    select_regularization_parameter(X.to_numpy(), y_x.to_numpy(), np.linspace(0, 5, 50), np.linspace(0, 20, 200))
    select_regularization_parameter(X.to_numpy(), y_y.to_numpy(), np.linspace(0, 5, 50), np.linspace(0, 20, 200))

    print('all ok')

    # new map
    # df['str_roadType'] = df.loc[:, 'linqmap_roadType'].astype(str)
    # mapp = go.Figure(layout=dict(title=dict(text='map')))
    # mapp.add_scatter(go.Scatter())
    # px.scatter(df, x='x', y='y', color='str_roadType')
    # mapp.write_html('map.html', auto_open=False)


def pred():
    df = pd.read_csv('waze_take_features.csv')
    new_df = get_comb_for_pred(df)
    print(new_df.shape)


if __name__ == '__main__':
    np.random.seed(0)
    # pd.set_option('display.max_rows', None)  # number of df rows to show
    # pd.set_option('display.max_columns', None)  # number of df columns to show
    main()
    # pred()
