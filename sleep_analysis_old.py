import pandas as pd
import numpy as np

def analyze_sleep_groups(indices, df, hold_step=10, previous_end_id=None):
    start_index = 0
    index_ranges = []
    is_continuous = 0
    group_indices = []  # 각 그룹의 모든 인덱스를 저장할 리스트
    continuous_indices = []  # 연속성이 있을 때 저장할 인덱스
    
    # 그룹화 로직
    current_group = []  # 현재 그룹의 인덱스를 저장할 임시 리스트
    
    for i in indices:
        if start_index == 0:
            start_index = i
            previous_index = i
            current_group = [i]  # 첫 번째 인덱스 저장
            # 첫 번째 그룹의 시작점에서 연속성 체크
            if previous_end_id is not None:
                first_group_id = df.iloc[i]['id']
                id_difference = abs(first_group_id - previous_end_id)
                is_continuous = 1 if id_difference < 30 else 0
                
                # 연속성이 있을 경우 이전 파일의 마지막 ID부터 현재 첫 그룹의 시작까지의 모든 인덱스 저장
                if is_continuous:
                    start_idx = df[df['id'] == previous_end_id].index[0]
                    end_idx = df[df['id'] == first_group_id].index[0]
                    continuous_indices = list(range(start_idx, end_idx + 1))
            continue
            
        if i - previous_index < hold_step:
            current_group.append(i)  # 현재 그룹에 인덱스 추가
            previous_index = i
        else:
            index_ranges.append((start_index, previous_index))
            group_indices.append(current_group)  # 완성된 그룹 저장
            start_index = i
            previous_index = i
            current_group = [i]  # 새 그룹 시작
            
        if i == indices[-1]:
            index_ranges.append((start_index, i))
            current_group.append(i)
            group_indices.append(current_group)  # 마지막 그룹 저장
    
    # 결과 정리
    num_groups = len(index_ranges)
    last_group_info = None
    first_group_info = None
    sleep_status = 0 if num_groups < 2 else 1
    group_ranges = []  # 그룹별 범위를 저장할 리스트
    time_ranges = []
    
    if num_groups == 1:
        start_id = df.iloc[index_ranges[0][0]]['id']
        end_id = df.iloc[index_ranges[0][1]]['id']
        start_time = df.iloc[index_ranges[0][0]]['time']
        end_time = df.iloc[index_ranges[0][1]]['time']
        duration = int(end_time - start_time)
        
        # heart rate와 breath rate 통계 계산
        group_indices = list(range(index_ranges[0][0], index_ranges[0][1] + 1))
        hr_values = [df.iloc[i]['heart_rate'] for i in group_indices]
        br_values = [df.iloc[i]['breath_rate'] for i in group_indices]
        
        hr_max = max(hr_values)
        hr_min = min(hr_values)
        hr_mean = round(sum(hr_values) / len(hr_values), 2)
        
        br_max = max(br_values)
        br_min = min(br_values)
        br_mean = round(sum(br_values) / len(br_values), 2)
        
        # 전체 시간 계산
        minutes = duration // 60
        seconds = duration % 60
        duration_str = f"{minutes}분 {seconds}초"
        
        # deep sleep 계산
        # group_indices = list(range(index_ranges[0][0], index_ranges[0][1] + 1))
        deep_sleep_indices = [i for i in group_indices if df.iloc[i]['sleep_state'] == 2]
        
        deep_sleep_str = "0분 0초"  # 기본값 설정
        light_sleep_str = f"{minutes}분 {seconds}초"  # 기본적으로 전체 시간을 light sleep으로
        
        deep_sleep_str = "0분 0초"
        light_sleep_str = f"{minutes}분 {seconds}초"  # 기본적으로 전체 시간을 light sleep으로

        if len(deep_sleep_indices) > 1:  # deep sleep이 2개 이상일 때만 계산
            deep_start_time = df.iloc[deep_sleep_indices[0]]['time']
            deep_end_time = df.iloc[deep_sleep_indices[-1]]['time']
            deep_duration = int(deep_end_time - deep_start_time)
            light_duration = duration - deep_duration
            
            deep_minutes = deep_duration // 60
            deep_seconds = deep_duration % 60
            deep_sleep_str = f"{deep_minutes}분 {deep_seconds}초"
            
            light_minutes = light_duration // 60
            light_seconds = light_duration % 60
            light_sleep_str = f"{light_minutes}분 {light_seconds}초"
        
        formatted_start = pd.to_datetime(start_time, unit='s').strftime('%Y-%m-%d %H:%M')
        formatted_end = pd.to_datetime(end_time, unit='s').strftime('%Y-%m-%d %H:%M')
        minutes = duration // 60
        seconds = duration % 60
        duration_str = f"{minutes}분 {seconds}초"
        
        time_ranges.append([formatted_start, formatted_end, duration_str, 
                          light_sleep_str, deep_sleep_str,
                          hr_max, hr_min, hr_mean,
                          br_max, br_min, br_mean])
        group_ranges.append([start_id, end_id])
    
    elif num_groups >= 2:
        for start_idx, end_idx in index_ranges[:-1]:
            start_id = df.iloc[start_idx]['id']
            end_id = df.iloc[end_idx]['id']
            start_time = df.iloc[start_idx]['time']
            end_time = df.iloc[end_idx]['time']
            duration = int(end_time - start_time)
            
            # heart rate와 breath rate 통계 계산
            group_indices = list(range(start_idx, end_idx + 1))
            hr_values = [df.iloc[i]['heart_rate'] for i in group_indices]
            br_values = [df.iloc[i]['breath_rate'] for i in group_indices]
            
            hr_max = max(hr_values)
            hr_min = min(hr_values)
            hr_mean = round(sum(hr_values) / len(hr_values), 2)
            
            br_max = max(br_values)
            br_min = min(br_values)
            br_mean = round(sum(br_values) / len(br_values), 2)
            
            # deep sleep 계산
            deep_sleep_indices = [i for i in group_indices if df.iloc[i]['sleep_state'] == 2]
            
            deep_sleep_str = ""
            light_sleep_str = ""
            if len(deep_sleep_indices) > 1:  # deep sleep이 2개 이상일 때만 계산
                deep_start_time = df.iloc[deep_sleep_indices[0]]['time']
                deep_end_time = df.iloc[deep_sleep_indices[-1]]['time']
                deep_duration = int(deep_end_time - deep_start_time)
                light_duration = duration - deep_duration
                
                deep_minutes = deep_duration // 60
                deep_seconds = deep_duration % 60
                deep_sleep_str = f"{deep_minutes}분 {deep_seconds}초"
                
                light_minutes = light_duration // 60
                light_seconds = light_duration % 60
                light_sleep_str = f"{light_minutes}분 {light_seconds}초"
            
            formatted_start = pd.to_datetime(start_time, unit='s').strftime('%Y-%m-%d %H:%M')
            formatted_end = pd.to_datetime(end_time, unit='s').strftime('%Y-%m-%d %H:%M')
            minutes = duration // 60
            seconds = duration % 60
            duration_str = f"{minutes}분 {seconds}초"
            
            time_ranges.append([formatted_start, formatted_end, duration_str, 
                              light_sleep_str, deep_sleep_str,
                              hr_max, hr_min, hr_mean,
                              br_max, br_min, br_mean])
            group_ranges.append([start_id, end_id])
    
    return {
        'ranges': index_ranges,
        'num_groups': num_groups,
        'is_continuous': is_continuous,
        'first_group_info': first_group_info,
        'last_group_info': last_group_info,
        'group_ranges': group_ranges,
        'time_ranges': time_ranges,
        'continuous_indices': continuous_indices,
        'sleep_status': sleep_status
    }

def analyze_snoring(sound, model, check_time=4, interval=2, sound_threshold=3):
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
    X = []
    if sleep_ready_index == None:
        print("Warning: No sleep ready index")
        sleep_ready_index = 0
    for i in range(sleep_ready_index, len(sound)-5, 5):
        X.append(np.array(sound[i:i+5].values))
    # 모델을 이용한 코골이 판단
    predict = model.predict(X)
    # 모델 index는 5개씩 묶음이고 코콜이 유무므로 다시 1개씩으로 변환
    snoring = [0]*len(sound)
    for i, pred in enumerate(predict):
        for j in range(5):
            snoring[sleep_ready_index + i*5 + j] = pred

    # 코골이한 구간만 추출
    snoring_index = np.where(np.array(snoring) == 1)[0]
    res = event_indices_to_ranges(snoring_index, hold_step=3)
    print('snoring_index', res)
    
    # 전체 코골이 시간 추출
    total_snoring_time = 0
    for i in res:
        total_snoring_time = total_snoring_time + 2*(i[1] - i[0])
        
    # 1분 이상 코골이 구간 추출
    over_1min_snoring_ranges = list(filter(lambda x: x[1] - x[0] >= 30, res))
    # print(f"snoring: {total_snoring_time}, {over_1min_snoring_ranges}")
    return total_snoring_time, over_1min_snoring_ranges, snoring_index

## 코골이 모델 예측 함수
def event_indices_to_ranges(indices, hold_step=10):
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
            index_ranges.append((start_index-1, previous_index)) # diff를 구할 때 index가 하나씩 밀리므로 -1해줌
            start_index = i
            previous_index = i
        if i == indices[-1]:
            index_ranges.append((start_index, i))
    return index_ranges