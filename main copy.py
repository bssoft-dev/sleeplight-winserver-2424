from serial_board import is_board_ready, SleepBoard, ControlBoard
import time
from database import session, RealtimeData



if __name__ == "__main__":
    while not is_board_ready():
        print("Board is not connected")
        time.sleep(1)
    time.sleep(2) # Stay until the board initialized
    sleep_board = SleepBoard()
    control_board = ControlBoard()

    try:
        # Turn on auto monitoring for all features
        #for feature in ['move', 'heart', 'breath']:
        #    sleep_board.set_auto_monitoring(feature, False)

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

            time.sleep(1)  # 1초 대기
            print(f"time: {time.time()-stime}")

    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
    finally:
        sleep_board.close()
        control_board.close()
        session.close()