from sqlalchemy.exc import SQLAlchemyError

def evaluate_sql(engine, metadata, table ,pred_sql):
    try: 
        connection = engine.connect()
        sql_stmt = text(pred_sql)
        results = connection.execute(sql_stmt)
        ids = [res[0] for res in results]
        query = resume_data.select().where(resume_data.columns.id.in_(ids))
        output = connection.execute(query)
        entities = output.fetchall()
        entities = [entity._asdict() for entity in entities]
    except SQLAlchemyError as error:
        print(error)
    finally:
        if connection:
            connection.close()
    return entities
