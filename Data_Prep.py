import numpy as np
import pandas as pd


def Prepare_data(data):
    data['international_plan'].replace({'no': 0, 'yes': 1}, inplace = True)
    data['churn'].replace({'no': 0, 'yes': 1}, inplace = True)
    data['voice_mail_plan'].replace({'no':0, 'yes': 1}, inplace = True)

    data.drop(columns=['state', 'area_code'], inplace=True)

    data['total_minutes'] = data['total_day_minutes'] + data['total_eve_minutes'] + data['total_night_minutes']
    data['total_calls'] = data['total_day_calls'] + data['total_eve_calls'] + data['total_night_calls']
    data['total_charge'] = data['total_day_charge'] + data['total_eve_charge'] + data['total_night_charge']

    data.drop(columns = ['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_day_calls', 'total_eve_calls', 'total_night_calls',
                                            'total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls'], inplace=True)

    #Drop Churn result from input data
    data = data
    y = data['churn']
    x = data.drop(columns = ['churn'], axis = 1)
            
    return x, y

def Prepare_testdata(data):
    data['international_plan'].replace({'no': 0, 'yes': 1}, inplace = True)
    data['voice_mail_plan'].replace({'no':0, 'yes': 1}, inplace = True)
    data.drop(columns=['state', 'area_code'], inplace=True)

    data['total_minutes'] = data['total_day_minutes'] + data['total_eve_minutes'] + data['total_night_minutes']
    data['total_calls'] = data['total_day_calls'] + data['total_eve_calls'] + data['total_night_calls']
    data['total_charge'] = data['total_day_charge'] + data['total_eve_charge'] + data['total_night_charge']

    data.drop(columns = ['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_day_calls', 'total_eve_calls', 'total_night_calls',
                                'total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls'], inplace=True)
    
    new_data = data.drop(columns = ['id'], axis = 1)

    return new_data