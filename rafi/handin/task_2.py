import numpy as np
import pandas as pd
from csv import writer


def run_2(date_list):
    file_names = ['05.06.2022', '07.06.2022', '09.06.2022']
    date_list_weekday = [x.weekday() for x in date_list]
    model_table = pd.read_csv('model_task_2.csv')
    timeslots = ['morning', 'afternoon', 'night']

    for i, weekday in enumerate(date_list_weekday):
        with open(file_names[i] + '.csv', 'w') as f_object:
            writer_object = writer(f_object)

            temp_weekday = model_table[model_table['day_of_week'] == weekday]
            for slot in timeslots:
                temp_slot = temp_weekday[temp_weekday['time_slot'] == slot].copy()
                temp_slot.reset_index(drop=True, inplace=True)

                to_file = temp_slot.loc[0, :].values.tolist()[-4:]
                writer_object.writerow(to_file)
                print('k')
