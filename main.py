from serial_board import is_board_ready, SleepBoard, ControlBoard
import time
from database import session, RealtimeData, SleepResultData
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select
from sleep_analysis import analyze_sleep_groups

if __name__ == "__main__":
    
    last_analysis_time = datetime.now()
    analysis_interval = timedelta(minutes=30)

    # 전역 변수로 sleep_start_id와 sleep_end_id 정의
    sleep_start_id = None
    sleep_end_id = None

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
            if current_time - last_analysis_time >= analysis_interval:
                print("30분이 경과했습니다. 수면 데이터 분석을 시작합니다.")
                
                # 최근 30분간의 데이터 조회
                thirty_mins_ago = int((current_time - analysis_interval).timestamp())
                query = select(RealtimeData).where(RealtimeData.time >= thirty_mins_ago)
                recent_data = session.execute(query).scalars().all()
                
                if recent_data:
                    # DataFrame으로 변환하기 위한 데이터 준비
                    data_dict = {
                        'id': [],
                        'sleep_state': [],
                        'heart_rate': [],
                        'breath_rate': [],
                        'sound_value': [],
                        'time': []
                    }
                    
                    for record in recent_data:
                        data_dict['id'].append(record.id)
                        data_dict['sleep_state'].append(record.sleep_state)
                        data_dict['heart_rate'].append(record.heart_rate)
                        data_dict['breath_rate'].append(record.breath_rate)
                        data_dict['sound_value'].append(record.sound_value)
                        data_dict['time'].append(record.time)
                    
                    # 수면 상태 분석
                    indices = np.where(np.array(data_dict['sleep_state']) != 0)[0]
                    analysis_result = analyze_sleep_groups(
                        indices, 
                        pd.DataFrame(data_dict), 
                        sleep_start_id, 
                        sleep_end_id,
                        hold_step=30
                    )

                if analysis_result:  # sleep_analysis가 있는 경우에만 처리
                    print(analysis_result)
                    try:
                        sleep_result = SleepResultData(
                            sleep_start=analysis_result[0]['sleep_start'],
                            sleep_end=analysis_result[0]['sleep_end'],
                            total_sleep_time=analysis_result[0]['total_sleep_time'],
                            light_sleep_time=analysis_result[0]['light_sleep_time'],
                            deep_sleep_time=analysis_result[0]['deep_sleep_time'],
                            hr_max=analysis_result[0]['hr_max'],
                            hr_min=analysis_result[0]['hr_min'],
                            hr_mean=analysis_result[0]['hr_mean'],
                            br_max=analysis_result[0]['br_max'],
                            br_min=analysis_result[0]['br_min'],
                            br_mean=analysis_result[0]['br_mean'],
                            total_snoring_time=analysis_result[0]['total_snoring_time'],
                            snoring_num=analysis_result[0]['snoring_num'],
                            snoring_time=analysis_result[0]['snoring_time']
                        )
                        session.add(sleep_result)
                        session.commit()
                        print("수면 분석 결과가 데이터베이스에 저장되었습니다.")
                        
                        # 분석 완료 후 전역 변수 초기화
                        sleep_start_id = None
                        sleep_end_id = None
                        
                    except Exception as e:
                        print(f"수면 분석 결과 저장 중 오류 발생: {e}")
                        session.rollback()
                    
                    print("-" * 20)
                
                last_analysis_time = current_time
            else:
                time.sleep(1.5)

            time.sleep(0.4)
            print(f"time: {time.time()-stime}")

    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
    finally:
        sleep_board.close()
        control_board.close()
        session.close()