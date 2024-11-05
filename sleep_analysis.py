import numpy as np
import pandas as pd
import pickle
import torch
import json

# 글로벌 변수 추가
sleep_start_id = None
sleep_end_id = None
sleep_start_time = None  # 시작 시간을 저장할 변수 추가

def convert_to_dataframe(records):
    """
    DB 레코드를 DataFrame으로 변환하는 함수
    
    Args:
        records: DB에서 조회한 RealtimeData 레코드 리스트
    
    Returns:
        pd.DataFrame: 변환된 DataFrame
    """
    data_dict = {
        'id': [],
        'sleep_state': [],
        'heart_rate': [],
        'breath_rate': [],
        'sound_value': [],
        'move_value': [],
        'time': []
    }
    
    for record in records:
        data_dict['id'].append(record.id)
        data_dict['sleep_state'].append(record.sleep_state)
        data_dict['heart_rate'].append(record.heart_rate)
        data_dict['breath_rate'].append(record.breath_rate)
        data_dict['sound_value'].append(record.sound_value)
        data_dict['move_value'].append(record.move_value)
        data_dict['time'].append(record.time)
    
    return pd.DataFrame(data_dict)
####################################################파라미터 수정
def find_sleep_boundaries(indices, df, hold_step=30):
    global sleep_start_id, sleep_end_id, sleep_start_time  # 전역 변수 사용
    
    start_index = 0
    index_ranges = []
    group_indices = []  
    current_group = []  

    # 그룹 인덱스 생성 로직
    for i in indices:
        if start_index == 0:
            start_index = i
            previous_index = i
            current_group = [i]
            continue
        
        if i - previous_index < hold_step:
            current_group.append(i)
            previous_index = i
        else:
            index_ranges.append((start_index-1, previous_index))
            group_indices.append(current_group)
            start_index = i
            previous_index = i
            current_group = [i]
        
        if i == indices[-1]:
            index_ranges.append((start_index, i))
            current_group.append(i)
            group_indices.append(current_group)

    num_groups = len(index_ranges)
    print(f"num_groups: {index_ranges}")

    # 수면 시작/종료 ID 설정
    if num_groups > 1:
        if sleep_start_id is None:
            first_group_start_idx = indices[0]
            sleep_start_id = df.iloc[first_group_start_idx]['id']
            sleep_start_time = df.iloc[first_group_start_idx]['time']
            print(f"num_groups > 1, sleep_start_id is None, sleep_start_id: {sleep_start_id}")
    
    elif num_groups == 1:
        start_idx, end_idx = index_ranges[0]
        if sleep_start_id is None:
            sleep_start_id = df.iloc[start_idx]['id']
            sleep_start_time = df.iloc[start_idx]['time']
            print(f"num_groups == 1, sleep_start_id is None, sleep_start_id: {sleep_start_id}")
        temp_end_id = df.iloc[end_idx]['id']
        print(f"start_id: {sleep_start_id}, end_id: {temp_end_id}")       
        
        # 수면 종료 ID 찾기
        current_last_id = df['id'].max()
        print(f"현재 들어온 마지막 id: {current_last_id}")
        sleep_range = df[(df['id'] >= sleep_start_id) & (df['id'] <= current_last_id)].copy()
        sleep_range = sleep_range.reset_index(drop=True)
        
        consecutive_zeros = 0
        for i in range(len(sleep_range)-1, -1, -1):
            current_state = sleep_range.iloc[i]['sleep_state']
            if current_state == 0:
                consecutive_zeros += 1
            else:
                ####################################################파라미터 수정
                if consecutive_zeros >= 90:
                    sleep_end_id = sleep_range.iloc[i]['id']
                    print(f"연속된 0이 {consecutive_zeros}개 발견되어 sleep_end_id 설정됨: {sleep_end_id}")
                break
    else:
        print("Warning: No sleep end index")
        sleep_start_id = None

    print(f"sleep_start_id: {sleep_start_id}, type: {type(sleep_start_id)}")
    print(f"sleep_end_id: {sleep_end_id}, type: {type(sleep_end_id)}")
    
    return sleep_start_id, sleep_end_id, sleep_start_time


def analyze_sleep_groups(indices, df, breath_scaler, breath_model, heartbeat_scaler, heartbeat_model):

    sleep_start_id, sleep_end_id, sleep_start_time = find_sleep_boundaries(indices, df)
    sleep_analysis = {}
    
    # 명시적으로 None이 아닌지 확인하고 정수형으로 변환
    if sleep_end_id is not None and sleep_start_id is not None:
        sleep_start_id = int(sleep_start_id)
        sleep_end_id = int(sleep_end_id)
        
        # ID가 DataFrame에 존재하는지 확인
        start_exists = len(df[df['id'] == sleep_start_id]) > 0
        end_exists = len(df[df['id'] == sleep_end_id]) > 0
        
        print(f"ID 존재 여부 - start_exists: {start_exists}, end_exists: {end_exists}")
        print(f"DataFrame의 ID 범위 - 최소: {df['id'].min()}, 최대: {df['id'].max()}")

        # start_id가 현재 DataFrame 범위를 벗어난 경우 처리
        if not start_exists:
            # DataFrame의 최소 ID를 sleep_start_id로 설정
            df.loc[df.index[0], 'id'] = sleep_start_id
            start_exists = True
            print(f"DataFrame의 최소 ID를 sleep_start_id로 조정: {sleep_start_id}")

        if start_exists and end_exists:
            # 저장된 시작 시간 사용
            start_time = sleep_start_time if sleep_start_time is not None else df[df['id'] == sleep_start_id]['time'].iloc[0]
            end_time = df[df['id'] == sleep_end_id]['time'].iloc[0]
            
            print(f"start_id: {sleep_start_id}, end_id: {sleep_end_id}")
            print(f"start_time: {start_time}, end_time: {end_time}")
            # 수면 시작/종료 시간 포맷팅 (UTC -> KST)
            sleep_start = pd.to_datetime(start_time, unit='s').tz_localize('UTC').tz_convert('Asia/Seoul').strftime('%Y-%m-%d %H:%M')
            sleep_end = pd.to_datetime(end_time, unit='s').tz_localize('UTC').tz_convert('Asia/Seoul').strftime('%Y-%m-%d %H:%M')
            
            # sleep_range_df 정의 (수면 시간 동안의 데이터만 필터링)
            sleep_range_df = df[(df['id'] >= sleep_start_id) & (df['id'] <= sleep_end_id)]
            
            # 심박수 최대/최소/평균 계산 (sleep_range_df 사용)
            hr_values = sleep_range_df['heart_rate'].values
            br_values = sleep_range_df['breath_rate'].values
            
            # 심박수 계산 (0 제외)
            hr_nonzero = hr_values[hr_values > 0]  # 0보다 큰 값만 선택
            hr_max = int(np.max(hr_nonzero)) if len(hr_nonzero) > 0 else 0
            hr_min = int(np.min(hr_nonzero)) if len(hr_nonzero) > 0 else 0
            hr_mean = round(np.mean(hr_nonzero), 2) if len(hr_nonzero) > 0 else 0
            
            # 호흡수 계산 (0 제외)
            br_nonzero = br_values[br_values > 0]  # 0보다 큰 값만 선택
            br_max = int(np.max(br_nonzero)) if len(br_nonzero) > 0 else 0
            br_min = int(np.min(br_nonzero)) if len(br_nonzero) > 0 else 0
            br_mean = round(np.mean(br_nonzero), 2) if len(br_nonzero) > 0 else 0 

            # 분석 결과 저장하는 부분
            sleep_analysis = {
                'sleep_start': sleep_start,
                'sleep_end': sleep_end,
                # 'total_sleep_time': total_sleep_str,
                # 'deep_sleep_time': deep_sleep_str,
                # 'light_sleep_time': light_sleep_str,
                'hr_max': hr_max,
                'hr_min': hr_min,
                'hr_mean': hr_mean,
                'br_max': br_max,
                'br_min': br_min,
                'br_mean': br_mean
            }

            # 코골이 모델 준비
            ###################################################여기 수정해주기
            generalization_value = 200 
            snoring_model_file = r'C:\Users\gistk\Documents\program\data_server\models\snoring_model.pkl'
            snoring_model = pickle.load(open(snoring_model_file, 'rb'))

            # 코골이 분석 추가
            if len(sleep_range_df) > 0:
                sound_values = sleep_range_df['sound_value'].reset_index(drop=True) / generalization_value
                
                # 코골이 분석 실행
                total_snoring_time, over_1min_snoring_ranges, snoring_index = analyze_snoring(
                    sound_values,
                    snoring_model)
            else:
                total_snoring_time = 0
                over_1min_snoring_ranges = []
                snoring_index = []
            
            # 코골이 구간별 최대/최소값을 담을 딕셔너리 생성
            snoring_values = {
                'xmin': [],
                'xmax': []
            }
            # 각 코골이 구간별로 sound_value가 최소/최대인 시점의 ID 저장
            for start_idx, end_idx in over_1min_snoring_ranges:
                
                snoring_values['xmin'].append(start_idx)
                snoring_values['xmax'].append(end_idx)
            
            # 코골이 분석 결과를 sleep_analysis 딕셔너리에 추가
            snoring_minutes = total_snoring_time // 60
            snoring_seconds = total_snoring_time % 60
            snoring_time_str = f"{snoring_minutes}분 {snoring_seconds}초"
            
            # 분석결과에 코골이 추가
            sleep_analysis.update({
                'total_snoring_time': snoring_time_str,
                'snoring_num': len(over_1min_snoring_ranges),
                'snoring_time': snoring_values
            })
            
            # 수면 판정 파라미터
            window_size = 20
            sliding_window = 10
            large_move_threshold = 10
            breath_stable_threshold = 0.2
            breath_stable_count_thres = 6
            heartbeat_stable_threshold = 0.2
            heartbeat_stable_count_thres = 12
            default_sleep_state = 1 # 수면 상태: 0 깨어있음, 1 얕은 수면, 2 깊은 수면

            # 모델을 이용한 호흡판정
            nonzero_breath = sleep_range_df[sleep_range_df['breath_rate'] != 0]
            if len(nonzero_breath) > 0:  # 데이터가 있는 경우에만 예측 수행
                breath_stable_predict = lstm_predict(nonzero_breath['breath_rate'], breath_stable_threshold, breath_scaler, breath_model)
            else:
                breath_stable_predict = np.array([])  # 빈 배열 반환
            
            sleep_range_df['breath_stable'] = np.nan
            if len(breath_stable_predict) > 0:  # 예측 결과가 있는 경우에만 할당
                sleep_range_df['breath_stable'].loc[nonzero_breath.index] = breath_stable_predict
            
            # 모델을 이용한 심박판정도 같은 방식으로 처리
            nonzero_heartbeat = sleep_range_df[sleep_range_df['heart_rate'] != 0]
            if len(nonzero_heartbeat) > 0:
                heartbeat_stable_predict = lstm_predict(nonzero_heartbeat['heart_rate'], heartbeat_stable_threshold, heartbeat_scaler, heartbeat_model)
            else:
                heartbeat_stable_predict = np.array([])
                
            sleep_range_df['heartbeat_stable'] = np.nan
            if len(heartbeat_stable_predict) > 0:
                sleep_range_df['heartbeat_stable'].loc[nonzero_heartbeat.index] = heartbeat_stable_predict

            # 호흡, 심박수 분석
            sleep = np.zeros(len(sleep_range_df))
            for i in range(0, len(sleep_range_df), sliding_window):
                snore = False
                for j in range(i, sliding_window):
                    # 1. 코골이가 있는 경우
                    if j in snoring_index:
                        sleep[i:i+sliding_window] = 2
                        snore = True
                        break
                if snore:
                    continue
                # 2. 움직임이 큰 경우
                if sleep_range_df['move_value'].iloc[i:i+window_size].mean() > large_move_threshold:
                    sleep[i:i+sliding_window] = default_sleep_state
                elif sleep_range_df['heart_rate'].iloc[i:i+window_size].where(sleep_range_df['heart_rate'] != 0).count() == 0:
                    sleep[i:i+sliding_window] = sleep[i-1] if i != 0 else default_sleep_state
                # 3. 심박이 안정한 경우
                elif sleep_range_df['heartbeat_stable'].iloc[i:i+window_size].where(sleep_range_df['heartbeat_stable'] == True).count() > heartbeat_stable_count_thres:
                    sleep[i:i+sliding_window] = 2
                # 4. 호흡이 안정한 경우
                elif sleep_range_df['breath_stable'].iloc[i:i+window_size].where(sleep_range_df['breath_stable'] == True).count() > breath_stable_count_thres:
                    sleep[i:i+sliding_window] = 1
                else:
                    sleep[i:i+sliding_window] = default_sleep_state
            print(sleep)

            # 깊은 수면(2) 구간 찾기
            deep_sleep_range = find_continuous_ranges(sleep, 2)
            
            # 딕셔너리 형태로 변환
            deep_sleep_periods = {
                'xmin': [],  # 시작 ID들을 저장할 리스트
                'xmax': []   # 종료 ID들을 저장할 리스트
            }
            
            for start_idx, end_idx in deep_sleep_range:
                start_id = int(sleep_range_df.iloc[start_idx]['id'])
                end_id = int(sleep_range_df.iloc[end_idx]['id'])
                deep_sleep_periods['xmin'].append(max(0, start_id - sleep_start_id))
                deep_sleep_periods['xmax'].append(end_id - sleep_start_id)
            
            # sleep_analysis 업데이트
            sleep_analysis.update({
                'deep_sleep_range': deep_sleep_periods
            })

            # 시간 정보와 수면 상태를 DataFrame으로 변환
            predict_df = pd.DataFrame()
            predict_df['sleep_mode'] = sleep
            
            # # UTC 시간을 KST로 변환하고 day, hour, min 추출
            # predict_df['datetime'] = pd.to_datetime(predict_df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
            # predict_df['day'] = predict_df['datetime'].dt.day
            # predict_df['hour'] = predict_df['datetime'].dt.hour
            # predict_df['min'] = predict_df['datetime'].dt.minute
            
            # # 분 단위로 그룹화하여 최빈값 계산
            # sleep_by_minute = predict_df.groupby(['day', 'hour', 'min'])['sleep_mode'].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
            
            # # 최빈값이 여러 개인 경우 처리
            # def is_np_array(value):
            #     return isinstance(value, np.ndarray)
            
            # for index, row in sleep_by_minute.iterrows():
            #     if is_np_array(row['sleep_mode']):
            #         sleep_by_minute.loc[index, 'sleep_mode'] = sleep_by_minute.loc[index-1, 'sleep_mode'] if index > 0 else default_sleep_state
            
            # sleep_by_minute['sleep_mode'] = sleep_by_minute['sleep_mode'].astype(int)


            print(f'sleep array: {len(sleep)}')
            # print(f'predict_df 길이: {len(sleep_by_minute)}')
            print(f'예측 데이터의 길이: {len(df)}')
            print(f'원본의 길이: {len(df)}')


            # 각 수면 상태별 시간 계산 (2초 단위의 원본 데이터 사용)
            light_sleep_seconds = len(predict_df[predict_df['sleep_mode'] == 1]) * 2  # 2초 단위이므로 2를 곱함
            deep_sleep_seconds = len(predict_df[predict_df['sleep_mode'] == 2]) * 2 # 모델이 2초 단위로 예측하므로 2를 곱함
            total_sleep_seconds = light_sleep_seconds + deep_sleep_seconds
            
            # 분과 초로 변환
            light_minutes, light_seconds = divmod(light_sleep_seconds, 60)
            deep_minutes, deep_seconds = divmod(deep_sleep_seconds, 60)
            total_minutes, total_seconds = divmod(total_sleep_seconds, 60)
            
            # 시간 문자열 생성
            light_sleep_str = f"{light_minutes}분 {light_seconds}초"
            deep_sleep_str = f"{deep_minutes}분 {deep_seconds}초"
            total_sleep_str = f"{total_minutes}분 {total_seconds}초"
            
            # 분석 결과 업데이트
            sleep_analysis.update({
                'light_sleep_time': light_sleep_str,
                'deep_sleep_time': deep_sleep_str,
                'total_sleep_time': total_sleep_str
            })
        # print(sleep_analysis)
    else:
        print("sleep_start_id 또는 sleep_end_id가 None입니다.")

    return sleep_analysis, sleep_start_id, sleep_end_id

## 코골이 모델 예측 함수
def event_indices_to_ranges(indices, hold_step=30):
    start_index = 0
    index_ranges = []
    for i in indices:
        if start_index == 0:
            start_index = i
            previous_index = i
            continue
        if i - previous_index < hold_step:
            previous_index = i
        else:
            index_ranges.append((start_index-1, previous_index)) # diff를 구 때 index가 하나씩 밀리므로 -1해줌
            start_index = i
            previous_index = i
        if i == indices[-1]:
            index_ranges.append((start_index, i))
    return index_ranges

def analyze_snoring(sound, model, check_time=4, interval=2, sound_threshold=3):
    if len(sound) == 0:
        return 0, [], []  # 빈 데이터일 경우 기본값 반환
    
    # sound가 checktime 이상 조용할 때를 수면에 들기 전단계로 판단
    max_queue_size = check_time / interval 
    queue = []
    sleep_ready_index = None
    for i in range(len(sound)):
        if len(queue) != max_queue_size:
            queue.append(sound[i])
        else:
            if sum(queue)/max_queue_size < sound_threshold:
                sleep_ready_index = i
                break
            else:
                queue.pop(0)
                queue.append(sound[i])

    if sleep_ready_index is None:
        print("Warning: No sleep ready index")
        sleep_ready_index = 0

    X = []
    try:
        for i in range(sleep_ready_index, len(sound)-5, 5):
            X.append(np.array(sound[i:i+5].values))
        
        if len(X) == 0:  # X가 비어있는 경우
            return 0, [], []
            
        # 2D 배열로 변환
        X = np.array(X)
        
        # 모델을 이용한 코골이 판단
        predict = model.predict(X)
        
        # 나머지 코드는 그대로 유지
        snoring = [0]*len(sound)
        for i, pred in enumerate(predict):
            for j in range(5):
                if sleep_ready_index + i*5 + j < len(snoring):
                    snoring[sleep_ready_index + i*5 + j] = pred
                    
        snoring_index = np.where(np.array(snoring) == 1)[0]
        res = event_indices_to_ranges(snoring_index, hold_step=3)
        
        total_snoring_time = 0
        for i in res:
            total_snoring_time = total_snoring_time + 2*(i[1] - i[0])
            
        over_1min_snoring_ranges = list(filter(lambda x: x[1] - x[0] >= 30, res))
        
        return total_snoring_time, over_1min_snoring_ranges, snoring_index
        
    except Exception as e:
        print(f"Error in analyze_snoring: {e}")
        return 0, [], []

def lstm_predict(data_nonzero, threshold, scale_model, lstm_model):
    # 데이터가 비어있는지 확인
    if len(data_nonzero) == 0:
        return np.array([])  # 빈 배열 반환
        
    scaled_data = scale_model.transform(data_nonzero.values.reshape(-1, 1)).reshape(-1)
    with torch.no_grad():
        predict = lstm_model(torch.tensor(scaled_data).float().view(-1, 1, 1))
    expected = scale_model.inverse_transform(predict.view(-1).numpy().reshape(-1, 1)).reshape(-1)
    result = np.abs((expected - data_nonzero.values)/data_nonzero.values) <= threshold
    return result

# 연속된 구간을 찾기 위한 함수
def find_continuous_ranges(arr, value):
    # diff가 1이 아닌 위치 찾기
    indices = np.where(arr == value)[0]
    if len(indices) == 0:
        return []
    
    ranges = []
    range_start = indices[0]
    prev_idx = indices[0]
    
    for idx in indices[1:]:
        if idx - prev_idx > 1:  # 연속되지 않은 경우
            ranges.append((range_start, prev_idx))
            range_start = idx
        prev_idx = idx
    
    # 마지막 구간 처리
    ranges.append((range_start, prev_idx))
    return ranges
