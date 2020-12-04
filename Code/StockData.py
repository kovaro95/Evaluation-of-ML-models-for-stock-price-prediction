if __name__== "__main__":
    import sys
    import datetime
    
    if sys.argv[1]=="":
        sys.exit("No stock was given")
        
    if sys.argv[2]=="":
        end=datetime.datetime(2020,1,1,0,0) 
    else:
        end=datetime.datetime.strptime(end, '%Y %m %e %H:%M')

    stock=sys.argv[1]
    end=sys.argv[2]
 
    import pandas as pd 
    from pandas_datareader import data as pdr
    import yfinance as yf #yahoo finance stock data
    
    start=end-datetime.timedelta(minutes=1000)
    min=yf.download(tickers=stock,start=start,end=end,interval="1m")
    
    thirt_min=yf.download(tickers=stock,start=start,end=end,interval="30m")
    
    hour=yf.download(tickers=stock,start=start,end=end,interval="1h")
    
    daily=yf.download(tickers=stock,start=start,end=end,interval="1d")
 
    weekly=yf.download(tickers=stock,start=start,end=end,interval="1wk")
    
    min.to_csv(f"{stock}_min.csv",sep=",",index=True)
    thirt_min.to_csv(f"{stock}_thirt_min.csv",sep=",",index=True)
    hour.to_csv(f"{stock}_hour.csv",sep=",",index=True)   
    daily.to_csv(f"{stock}_daily.csv",sep=",",index=True)
    weekly.to_csv(f"{stock}_weekly.csv",sep=",",index=True)
    
    
    
    
    