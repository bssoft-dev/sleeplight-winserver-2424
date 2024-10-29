import numpy as np
import pandas as pd
import pickle

# # 글로벌 변수 추가
# sleep_start_id = None
# sleep_end_id = None

def analyze_sleep_groups(indices, df, sleep_start_id, sleep_end_id, hold_step=30):
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
            
            # 심박수 최대/최소/평균 계산
            hr_values = sleep_range_df['heart_rate'].values
            br_values = sleep_range_df['breath_rate'].values
            
            # 최소값은 0을 제외한 값
            hr_max = int(np.max(hr_values))
            hr_min = int(np.min(hr_values[hr_values > 0])) if np.any(hr_values > 0) else 0
            hr_mean = round(np.mean(hr_values), 2)
            
            # 호흡수 최대/최소/평균 계산
            br_max = int(np.max(br_values))
            br_min = int(np.min(br_values[br_values > 0])) if np.any(br_values > 0) else 0
            br_mean = round(np.mean(br_values), 2)
            
            # 분석 결과 저장하는 부분
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

            # 코골이 모델 준비
            generalization_value = 200
            snoring_model_file = 'models/snoring_model.pkl'
            snoring_model = pickle.load(open(snoring_model_file, 'rb'))

            # 코골이 분석 추가
            sleep_range_df = df[(df['id'] >= sleep_start_id) & (df['id'] <= sleep_end_id)].reset_index(drop=True)
            sound_values = sleep_range_df['sound_value'].reset_index(drop=True) / generalization_value
            
            # 코골이 분석 실행
            total_snoring_time, over_1min_snoring_ranges, snoring_index = analyze_snoring(
                sound_values,
                snoring_model)
            
            # 코골이 구간별 최대/최소값을 담을 딕셔너리 생성
            snoring_values = {
                'xmin': [],
                'xmax': []
            }
            
            # 각 코골이 구간별로 sound_value의 최대/최소값 구하기
            for start_idx, end_idx in over_1min_snoring_ranges:
                segment_values = sleep_range_df['sound_value'].iloc[start_idx:end_idx+1]
                snoring_values['xmin'].append(int(segment_values.min()))
                snoring_values['xmax'].append(int(segment_values.max()))
            
            # 코골이 분석 결과를 sleep_analysis 딕셔너리에 추가
            snoring_minutes = total_snoring_time // 60
            snoring_seconds = total_snoring_time % 60
            snoring_time_str = f"{snoring_minutes}분 {snoring_seconds}초"
            
            # 분석결과에 코콜이 추가
            sleep_analysis.update({
                'total_snoring_time': snoring_time_str,
                'snoring_num': len(over_1min_snoring_ranges),
                'snoring_time': snoring_values
            })
        # print(sleep_analysis)

    # return {
    #         'ranges': index_ranges,
    #         'num_groups': num_groups,
    #         'sleep_start_id': sleep_start_id,
    #         'sleep_end_id': sleep_end_id,
    #         'sleep_analysis': sleep_analysis
    #     }
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
            index_ranges.append((start_index-1, previous_index)) # diff를 구할 때 index가 하나씩 밀리므로 -1해줌
            start_index = i
            previous_index = i
        if i == indices[-1]:
            index_ranges.append((start_index, i))
    return index_ranges

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

    # if sleep_ready_index is None:
    #     # 경고 메시지를 로깅으로 변경하고, 의미있는 기본값 설정
    #     print("수면 준비 상태를 찾을 수 없습니다. 첫 번째 조용한 구간을 찾아 분석을 시작합니다.")
        
    #     # 첫 번째로 발견되는 조용한 구간을 찾아보기
    #     for i in range(0, len(sound)-int(max_queue_size)):
    #         if np.mean(sound[i:i+int(max_queue_size)]) < sound_threshold:
    #             sleep_ready_index = i
    #             break
        
    #     # 여전히 찾지 못한 경우
    #     if sleep_ready_index is None:
    #         print("분석 가능한 수면 구간을 찾을 수 없습니다.")
    #         return 0, [], []  # 의미있는 기본값 반환
                
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