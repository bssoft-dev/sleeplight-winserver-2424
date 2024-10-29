import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('data/realtime_data_test_prepared.csv')

# sleep_state 값 변경 함수
def modify_sleep_state():
    rand = np.random.random()
    if rand < 0.70:
        return 0
    elif rand < 0.95:
        return 1
    else:
        return 2

# sleep_state 열 수정
df['sleep_state'] = df['sleep_state'].apply(lambda x: modify_sleep_state())

# 수정된 데이터를 CSV 파일로 저장
df.to_csv('data/fake_data.csv', index=False)

print("파일이 성공적으로 수정되었습니다.")