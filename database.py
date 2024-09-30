from sqlalchemy import create_engine, Column, Integer, SMALLINT
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
    time = Column(Integer)

    def __repr__(self):
        return f"User(name='{self.name}', email='{self.email}', age={self.age})"

# 데이터베이스 테이블 생성
Base.metadata.create_all(engine)

# 세션 생성
Session = sessionmaker(bind=engine)
session = Session()