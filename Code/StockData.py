if __name__== "__main__":
    import sys
    import datetime
    
    if not len(sys.argv) > 1:
        sys.exit("Required parameter is a stock symbol")
    
    if sys.argv[1]=="":
        sys.exit("No stock was given")
        
    if sys.argv!="":  
        end=datetime.datetime.strptime(sys.argv[2], '%Y-%m-%d')
    else:
        end=datetime.date.today()
        
    stock=sys.argv[1]
    
    import pandas as pd 
    from pandas_datareader import data as pdr
    import yfinance as yf #yahoo finance stock data
    
    start=end-datetime.timedelta(days=7)
    min=yf.Ticker(stock).history(period="7d",interval="1m")
    #yf.download(tickers=stock,start=start,end=end,interval="60m")
    
    start=end-datetime.timedelta(days=59)
    thirt_min=yf.Ticker(stock).history(period="60d",interval="30m")
    #yf.download(tickers=stock,start=start,end=end,interval="30m")
    
    start=end-datetime.timedelta(days=200)
    hour=yf.Ticker(stock).history(period="730d",interval="60m")
    #yf.download(tickers=stock,start=start,end=end,interval="60m")
    
    start=end-datetime.timedelta(days=500)
    daily=yf.Ticker(stock).history(period="max",interval="1d")
    #yf.download(tickers=stock,start=start,end=end,interval="1d")
    
    start=end-datetime.timedelta(weeks=700)
    weekly=yf.Ticker(stock).history(period="max",interval="1wk")
    #yf.download(tickers=stock,start=start,end=end,interval="1wk")
    
    min.to_csv(f"..\Data\Prices\{stock}_min.csv",sep=",",index=True)
    thirt_min.to_csv(f"..\Data\Prices\{stock}_thirt_min.csv",sep=",",index=True)
    hour.to_csv(f"..\Data\Prices\{stock}_hour.csv",sep=",",index=True)   
    daily.to_csv(f"..\Data\Prices\{stock}_daily.csv",sep=",",index=True)
    weekly.to_csv(f"..\Data\Prices\{stock}_weekly.csv",sep=",",index=True)
    
    
    
    
    