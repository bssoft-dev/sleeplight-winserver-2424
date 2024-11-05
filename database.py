from sqlalchemy import create_engine, Column, Integer, SMALLINT, String, Double
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from env import DATABASE_URL

# SQLite 데이터베이스 엔진 생성
engine = create_engine(DATABASE_URL)

# 기본 클래스 생성
Base = declarative_base()

# 사용자 모델 정의
class RealtimeData(Base):
    __tablename__ = 'realtime_data'

    id = Column(Integer, primary_key=True)
    move_state = Column(SMALLINT, nullable=True)
    move_value = Column(SMALLINT, nullable=True)
    heart_rate = Column(SMALLINT, nullable=True)
    breath_rate = Column(SMALLINT, nullable=True)
    sleep_state = Column(SMALLINT, nullable=True)
    sound_value = Column(SMALLINT, nullable=True)
    temp_value = Column(SMALLINT, nullable=True)
    run_id = Column(Integer, nullable=True)
    time = Column(Integer)

    def __repr__(self):
        return f"User(name='{self.name}', email='{self.email}', age={self.age})"
    
class SleepResultData(Base):
    __tablename__ = 'sleep_results'
    
    id = Column(Integer, primary_key=True)
    sleep_start = Column(String, nullable=True)
    sleep_end = Column(String, nullable=True)
    total_sleep_time = Column(String, nullable=True)
    light_sleep_time = Column(String, nullable=True)
    deep_sleep_time = Column(String, nullable=True)
    deep_sleep_range = Column(String, nullable=True)
    hr_max = Column(Double, nullable=True)
    hr_min = Column(Double, nullable=True)
    hr_mean = Column(Double, nullable=True)
    br_max = Column(Double, nullable=True)
    br_min = Column(Double, nullable=True)
    br_mean = Column(Double, nullable=True)
    total_snoring_time = Column(String, nullable=True)
    snoring_num = Column(Integer, nullable=True)
    snoring_time = Column(String, nullable=True)
    deep_sleep_range = Column(String, nullable=True)
    
    def __repr__(self):
        return f"SleepResult(id={self.id}, sleep_start='{self.sleep_start}', sleep_end='{self.sleep_end}')"

# 데이터베이스 테이블 생성
Base.metadata.create_all(engine)

# 세션 생성
Session = sessionmaker(bind=engine)
session = Session()