from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String
import fire

resume_engine = create_engine('sqlite:///resume.db', echo=True, future=True)

resume_metadata = MetaData()

resume_table = Table(
    'resume_data',
    resume_metadata,
    Column('id', Integer, primary_key = True),
    Column('first_name', String),
    Column('last_name', String),
    Column('clearance', String, nullable=True),
    Column('certification', String, nullable=True),
    Column('skills', String),
    Column('education', String),
    Column('work_history', Integer)
)

def construct_database(
    engine=resume_engine,
    metadata=resume_metadata,
    table_data=resume_table,
):
    metadata.create_all(engine)

if __name__ == "__main__":
    fire.Fire(construct_database)
