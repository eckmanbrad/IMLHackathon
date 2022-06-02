from final_functions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import plotly.express as px


def our_knn(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=24)

    loss_lst = []
    for i in range(2, 15):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(train_x, train_y)

        y_predict = neigh.predict(test_x)
        loss_lst.append(f1_score(test_y.to_numpy(), y_predict, average='weighted'))
    k = np.argmin(loss_lst) + 1
    elbow = px.line(loss_lst, title='k =' + str(k) + ', misclassification_error: ' + str(round(loss_lst[k - 1], 3)),
                       labels={'index': 'k', 'value': 'missclassification'})
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

    elbow = px.scatter(loss_lst, title='depth = 6' + ', misclassification_error: ' + str(round(miss, 3)),
                       labels={'index': 'depth', 'value': 'missclassification'})
    elbow.write_html('rf.html', auto_open=False)

    res = pd.DataFrame({'predicted': test_y_predict, 'real': test_y})
    figres = px.scatter(res, title='misclassification_error: ' + str(round(miss, 2)),
                        labels={'index': 'trade', 'value': 'label'})
    figres.write_html('final.html', auto_open=True)


# def our_tree_part():




def main():
    # df = pd.read_csv('waze_data.csv')
    # df = clean(df)
    #
    # learn_df = get_comb(df)
    # learn_df.to_csv('learn_df.csv')
    df = pd.read_csv('learn_df.csv', index_col='Unnamed: 0')
    X, y = split_X_y_type_pred(df)
    our_knn(X, y)
    our_rf(X, y)

    print('all ok')

    # new map
    # df['str_roadType'] = df.loc[:, 'linqmap_roadType'].astype(str)
    # mapp = go.Figure(layout=dict(title=dict(text='map')))
    # mapp.add_scatter(go.Scatter())
    # px.scatter(df, x='x', y='y', color='str_roadType')
    # mapp.write_html('map.html', auto_open=False)


if __name__ == '__main__':
    np.random.seed(0)
    # pd.set_option('display.max_rows', None)  # number of df rows to show
    # pd.set_option('display.max_columns', None)  # number of df columns to show
    # main()
    pred()
