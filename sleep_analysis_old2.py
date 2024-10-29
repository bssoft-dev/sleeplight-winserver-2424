import numpy as np
import pandas as pd

# # 글로벌 변수 추가
# sleep_start_id = None
# sleep_end_id = None

def analyze_sleep_groups(indices, df, hold_step=30, sleep_start_id=None, sleep_end_id=None):
    # global sleep_start_id, sleep_end_id

    start_index = 0
    index_ranges = []
    group_indices = []  
    current_group = []  

    for i in indices:
        if start_index == 0:
            start_index = i
            previous_index = i
            current_group = [i]  # 현재 그룹 시작
            continue
        
        if i - previous_index < hold_step:  # 들여쓰기 수정
            current_group.append(i)  # 현재 그룹에 인덱스 추가
            previous_index = i
        else:
            index_ranges.append((start_index-1, previous_index))  # diff를 구할 때 index가 하나씩 밀리므로 -1해줌
            group_indices.append(current_group)  # 완성된 그룹을 저장
            start_index = i
            previous_index = i
            current_group = [i]  # 새로운 그룹 시작
        
        if i == indices[-1]:  # 마지막 인덱스 처리
            index_ranges.append((start_index, i))
            current_group.append(i)
            group_indices.append(current_group)

    num_groups = len(index_ranges)
    # print(f"num_groups: {index_ranges}")
    if num_groups > 1:
        if sleep_start_id is None:
            first_group_start_idx = indices[0]
            sleep_start_id = df.iloc[first_group_start_idx]['id']
    elif num_groups == 1:
        start_idx, end_idx = index_ranges[0]
        start_id = df.iloc[start_idx]['id']
        end_id = df.iloc[end_idx]['id']
        
        if sleep_start_id is None:
            sleep_start_id = start_id
        sleep_end_id = end_id

    print(f"sleep_start_id: {sleep_start_id}, type: {type(sleep_start_id)}")
    print(f"sleep_end_id: {sleep_end_id}, type: {type(sleep_end_id)}")
    # sleep_end_id가 설정되었을 때 수면 시간 분석
    sleep_analysis = {}
    
    # 명시적으로 None이 아닌지 확인하고 정수형으로 변환
    if sleep_end_id is not None and sleep_start_id is not None:
        sleep_start_id = int(sleep_start_id)
        sleep_end_id = int(sleep_end_id)
        
        # ID가 DataFrame에 존재하는지 확인
        start_exists = len(df[df['id'] == sleep_start_id]) > 0
        end_exists = len(df[df['id'] == sleep_end_id]) > 0
        
        print(f"start_exists: {start_exists}, end_exists: {end_exists}")  # 디버깅용
        
        if start_exists and end_exists:
            # 시작과 끝 시간 가져오기
            start_time = df[df['id'] == sleep_start_id]['time'].iloc[0]
            end_time = df[df['id'] == sleep_end_id]['time'].iloc[0]
            
            # 수면 시작/종료 시간 포맷팅
            sleep_start = pd.to_datetime(start_time, unit='s').strftime('%Y-%m-%d %H:%M')
            sleep_end = pd.to_datetime(end_time, unit='s').strftime('%Y-%m-%d %H:%M')
            
            # 총 수면 시간 계산 (초 단위)
            total_sleep_time = int(end_time - start_time)
            minutes = total_sleep_time // 60
            seconds = total_sleep_time % 60
            total_sleep_str = f"{minutes}분 {seconds}초"
            
            # deep_sleep_time 계산
            sleep_range_df = df[(df['id'] >= sleep_start_id) & (df['id'] <= sleep_end_id)]
            deep_sleep_count = sleep_range_df[sleep_range_df['sleep_state'] == 2].shape[0]
            deep_sleep_time = deep_sleep_count * 2  # 2초 단위로 가정
            
            # light_sleep_time 계산
            light_sleep_time = total_sleep_time - deep_sleep_time
            
            deep_minutes = deep_sleep_time // 60
            deep_seconds = deep_sleep_time % 60
            deep_sleep_str = f"{deep_minutes}분 {deep_seconds}초"
            
            light_minutes = light_sleep_time // 60
            light_seconds = light_sleep_time % 60
            light_sleep_str = f"{light_minutes}분 {light_seconds}초"
            
            hr_values = sleep_range_df['heart_rate'].values
            br_values = sleep_range_df['breath_rate'].values
            
            hr_max = int(np.max(hr_values))
            hr_min = int(np.min(hr_values))
            hr_mean = round(np.mean(hr_values), 2)
            
            br_max = int(np.max(br_values))
            br_min = int(np.min(br_values))
            br_mean = round(np.mean(br_values), 2)
            
            sleep_analysis = {
                'sleep_start': sleep_start,
                'sleep_end': sleep_end,
                'total_sleep_time': total_sleep_str,
                'deep_sleep_time': deep_sleep_str,
                'light_sleep_time': light_sleep_str,
                'hr_max': hr_max,
                'hr_min': hr_min,
                'hr_mean': hr_mean,
                'br_max': br_max,
                'br_min': br_min,
                'br_mean': br_mean
            }
        print(sleep_analysis)

    # return {
    #         'ranges': index_ranges,
    #         'num_groups': num_groups,
    #         'sleep_start_id': sleep_start_id,
    #         'sleep_end_id': sleep_end_id,
    #         'sleep_analysis': sleep_analysis
    #     }
    return sleep_analysis, sleep_start_id, sleep_end_id