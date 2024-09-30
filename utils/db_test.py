import pandas as pd
import numpy as np
import os, json

watchDir = 'C:/sleep/DATA'
SLEEP_SKIP_INDEX = 5*30 # 5분

pd.set_option('mode.chained_assignment',  None)

def analyze_sleep(run_id, hr_filename, model):
    '''
    This is Main Function for analyzing sleep.
    '''
    user_id = run_id.split('_')[0]
    result = {
        "user_id": user_id,
        "run_id": run_id,
        "sleep_start": "",
        "sleep_end": "",
        "interval_to_sleep": "",
        "total_sleep_time": "",
        "light_sleep_time": "",
        "deep_sleep_time": "",
        "toss_turn_count": 0,
        "toss_turn_time": {},
        "hr_max": 0,
        "hr_min": 0,
        "hr_mean": 0,
        "total_snoring_time": "",
        "snoring_num": 0,
        "snoring_time": {},
        "exercise_val": 0.0
    }
    result, df = analyze_sensors(run_id, hr_filename, result, model)
    if result is None:
        print('Error: 데이터가 너무 짧습니다. 5분 이상의 데이터만 분석가능합니다.')
        return None, None
    df.insert(0, 'run_id', run_id)
    df.insert(1, 'user_id', user_id)
    
    result_df = pd.DataFrame.from_dict(result, orient='index').transpose()
    print(f"result: {result}")
    return df, result_df


