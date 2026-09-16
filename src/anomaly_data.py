import awswrangler as wr


def get_tmax_data():

    query = """
        SELECT date, value AS tmax
        FROM mart_weather_signals
        WHERE datatype = 'TMAX'
        ORDER BY date
    """

    df = wr.athena.read_sql_query(sql=query, database="flagship_weather_db")
    return df