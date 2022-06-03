import datetime as dt
from task_1 import run_1
from task_2 import run_2


def main(path_to_task_1_test, path_to_task_2_test):
    try:
        run_1(path_to_task_1_test)
    except:
        print('task_1 failed.')
    try:
        run_2(path_to_task_2_test)
    except:
        print('task_2 failed.')



if __name__ == '__main__':
    main('waze_take_features.csv', [dt.datetime.strptime('2022.06.05', '%Y.%m.%d'),
                                                 dt.datetime.strptime('2022.06.07', '%Y.%m.%d'),
                                                 dt.datetime.strptime('2022.06.09', '%Y.%m.%d')])
