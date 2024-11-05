from serial_board import is_board_ready, SleepBoard, ControlBoard
import time
from database import session, RealtimeData, SleepResultData
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select
from sleep_analysis import analyze_sleep_groups
import pickle
from models.lstm_model import LSTMModel
import torch
import json
from sleep_analysis import find_sleep_boundaries, convert_to_dataframe


target_receive_time = 2

if __name__ == "__main__":
    
    last_analysis_time = datetime.now()
    analysis_interval = timedelta(minutes=30)

    # 모델 불러오기
    # 스케일러 불러오기
    with open(r'C:\Users\gistk\Documents\program\data_server\models\breath_scaler.pkl', 'rb') as f:
        breath_scaler = pickle.load(f)

    hidden_size = 50
    num_layers = 1

    # 스케일러 불러오기
    with open(r'C:\Users\gistk\Documents\program\data_server\models\heartbeat_scaler.pkl', 'rb') as f:
        heartbeat_scaler = pickle.load(f)

    # 모델 불러오기
    breath_model_file = r'C:\Users\gistk\Documents\program\data_server\models\lstm_breath_model.pth'
    breath_model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    breath_model.load_state_dict(torch.load(breath_model_file))
    breath_model = breath_model.eval()

    # 모델 불러오기
    heartbeat_model_file = r'C:\Users\gistk\Documents\program\data_server\models\lstm_heartbeat_model.pth'
    heartbeat_model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    heartbeat_model.load_state_dict(torch.load(breath_model_file))
    heartbeat_model = heartbeat_model.eval()
    


    # 전역 변수로 sleep_start_id와 sleep_end_id 정의
    sleep_start_id = None
    sleep_end_id = None
    analysis_result = None

    # 수면 판정을 위한 전역 변수 정의
    flag_move_large = []
    breath_idx = []
    heartbeat_idx = []

    while not is_board_ready():
        print("Board is not connected")
        time.sleep(1)
    time.sleep(2)
    sleep_board = SleepBoard()
    control_board = ControlBoard()

    try:
        data = {}
        while True:
            stime = time.time()
            for coms in sleep_board.commands:
                value = sleep_board.get_value(coms)
                if value is not None:
                    print(f"{coms}: {value}")
                    data[coms] = value
                else:
                    print(f"{coms} 읽기 실패")

            for coms in control_board.commands:
                value = control_board.get_value(coms)
                if value is not None:
                    print(f"{coms}: {value}")
                    data[coms] = value
                else:
                    print(f"{coms} 읽기 실패")
            print("-" * 20)

            data['time'] = int(time.time())

            try:
                real = RealtimeData(**data)
                session.add(real)
                session.commit()            
            except Exception as e:
                print(f"Error: {e}")
                session.rollback()

            # 30분마다 분석 실행
            current_time = datetime.now()
            start_time = time.time()
            if current_time - last_analysis_time >= analysis_interval:

                if sleep_start_id is None:
                    # 최근 30분간의 데이터 조회
                    thirty_mins_ago = int((current_time - analysis_interval).timestamp())
                    query = select(RealtimeData).where(RealtimeData.time >= thirty_mins_ago)
                    sleep_data = session.execute(query).scalars().all()
                    
                    if sleep_data:
                        feature_df = convert_to_dataframe(sleep_data)
                        indices = np.where(np.array(feature_df['sleep_state']) != 0)[0]
                        temp_start_id, temp_end_id, sleep_start_time = find_sleep_boundaries(indices, feature_df)

                        if temp_end_id is None:
                            print("계속 수면 중입니다. 다음시간으로 넘어갑니다 sleep_end_id is None.")
                            sleep_start_id = temp_start_id  # 수면 시작 ID 저장
                        else:
                            print(f"sleep_start_id: {temp_start_id}, sleep_end_id: {temp_end_id}, sleep_start_time: {sleep_start_time}")
                            indices = np.where(np.array(feature_df['sleep_state']) != 0)[0]
                            analysis_result, start_id, end_id = analyze_sleep_groups(
                                indices, 
                                feature_df,
                                breath_scaler,
                                breath_model,
                                heartbeat_scaler,
                                heartbeat_model
                            )
                    else:
                        print("데이터가 없습니다. 확인해주세요.")

                elif sleep_start_id is not None:
                        print("수면 중인 데이터를 분석합니다.")
                        query = select(RealtimeData).where(
                            RealtimeData.id >= sleep_start_id,
                            RealtimeData.time <= int(current_time.timestamp())
                        )
                        sleep_data_start_id = session.execute(query).scalars().all()
                        
                        feature_df = convert_to_dataframe(sleep_data_start_id)
                        indices = np.where(np.array(feature_df['sleep_state']) != 0)[0]
                        _, sleep_end_id, _ = find_sleep_boundaries(indices, feature_df)

                        if sleep_end_id is None:
                            print("계속 수면 중입니다. 다음시간으로 넘어갑니다.")
                        else:
                            print(f"수면이 종료되었습니다. sleep_end_id: {sleep_end_id}")
                            indices = np.where(np.array(feature_df['sleep_state']) != 0)[0]
                            analysis_result, start_id, end_id = analyze_sleep_groups(
                                indices, 
                                feature_df,
                                breath_scaler,
                                breath_model,
                                heartbeat_scaler,
                                heartbeat_model
                            )

                if analysis_result:  # sleep_analysis가 있는 경우에만 처리
                    print(analysis_result)
                    try:
                        # 동일한 시간대의 수면 데이터가 이미 존재하는지 확인
                        existing_record = session.query(SleepResultData).filter(
                            SleepResultData.sleep_start == analysis_result['sleep_start'],
                            SleepResultData.sleep_end == analysis_result['sleep_end']
                        ).first()
                        
                        if existing_record is None:
                            sleep_result = SleepResultData(
                                sleep_start=analysis_result['sleep_start'],
                                sleep_end=analysis_result['sleep_end'],
                                total_sleep_time=analysis_result['total_sleep_time'],
                                light_sleep_time=analysis_result['light_sleep_time'],
                                deep_sleep_time=analysis_result['deep_sleep_time'],
                                hr_max=int(analysis_result['hr_max']),
                                hr_min=int(analysis_result['hr_min']),
                                hr_mean=float(analysis_result['hr_mean']),  # np.float64를 float로 변환
                                br_max=int(analysis_result['br_max']),
                                br_min=int(analysis_result['br_min']),
                                br_mean=float(analysis_result['br_mean']),  # np.float64를 float로 변환
                                total_snoring_time=analysis_result['total_snoring_time'],
                                snoring_num=analysis_result['snoring_num'],
                                snoring_time=json.dumps({
                                    'xmin': [int(x) for x in analysis_result['snoring_time']['xmin']],
                                    'xmax': [int(x) for x in analysis_result['snoring_time']['xmax']]
                                }),
                                deep_sleep_range=json.dumps({
                                    'xmin': [int(x) for x in analysis_result['deep_sleep_range']['xmin']],
                                    'xmax': [int(x) for x in analysis_result['deep_sleep_range']['xmax']]
                                })
                            )
                            session.add(sleep_result)
                            session.commit()
                            print("수면 분석 결과가 데이터베이스에 저장되었습니다.")

                        run_id = sleep_result.id
                        print(f'run_id: {run_id}')

                        # 수면으로 판단한 모든 realtime_data에 run_id 업데이트
                        print(f'start_id: {start_id}')
                        print(f'end_id:{end_id}')
                        try:
                            session.query(RealtimeData).filter(RealtimeData.id.between(start_id, end_id)).update({RealtimeData.run_id: run_id})
                            session.commit()
                        except Exception as e:
                            print(f"Error: {e}")

                        # 분석 완료 후 전역 변수 초기화
                        sleep_start_id = None
                        sleep_end_id = None
                        analysis_result = None
                        flag_move_large = []
                        breath_idx = []
                        heartbeat_idx = []
                        
                    except Exception as e:
                        print(f"수면 분석 결과 저장 중 오류 발생: {e}")
                        session.rollback()
                    
                    print("-" * 20)
                
                last_analysis_time = current_time
                print(f"END_Time: {time.time() - start_time}")
            remaining_time = target_receive_time - (time.time() - stime)
            if remaining_time > 0:
                time.sleep(remaining_time)
            print(f"time: {time.time()-stime}")

    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
    finally:
        sleep_board.close()
        control_board.close()
        session.close()


