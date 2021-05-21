import sys
import datetime
import psycopg2
import pandas as pd 
import yfinance as yf #yahoo finance stock data
from io import StringIO

def Update_databases():
    STOCKS=yf.Tickers("SPY GME JPM T MSFT GOOG AAPL")
    TODAY=datetime.date.today()
    START=datetime.date(2018,1,4)  

    connection = psycopg2.connect(user="postgres",
        password="adminpw",
        host="127.0.0.1",
        port="5432",
        database="Thesis")

    sio = StringIO()

    try:
        curr = connection.cursor()
        curr.execute('SELECT max(date) FROM stockprices')
        maxdate=curr.fetchall()[0]
        startdate=(START if maxdate[0] is None else (maxdate[0]+datetime.timedelta(days=1)) )

        if startdate==TODAY:
            return('stockprices SQL table is up-to-date')

        df=STOCKS.history(start=startdate,end=TODAY)['Close']
        df=df[startdate:]
        df.reset_index(inplace=True)
        df.columns= df.columns.str.lower()
        
        if df.shape[0]==0:
            return('No new data is available to update table')

        sio.write(df.to_csv(index=None, header=None))
        sio.seek(0)
        
        with curr as c:
            c.copy_from(sio, "stockprices", columns=df.columns, sep=',')
            connection.commit()

        curr.close()
    except (Exception, psycopg2.Error) as error:
        print("Error", error)

if __name__=='__main__':
    Update_databases()