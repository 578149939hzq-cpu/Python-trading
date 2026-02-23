import requests
import pandas as pd
import time
from datetime import datetime

def get_binance_data(symbol="ETHUSDT", interval="1h", start_str="2020-01-01"):
    """
    直接从币安 API 抓取历史数据 (分段抓取，因为一次只能抓1000根)
    """
    print(f"🚀 开始从币安下载 {symbol} ({interval}) ...")
    
    # 1. 转换时间格式
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000) # 现在
    
    data_list = []
    current_ts = start_ts
    
    # 2. 循环抓取 (因为币安限制一次只能给1000条)
    while current_ts < end_ts:
        print(f"   ⏳ 正在下载: {datetime.fromtimestamp(current_ts/1000)} ...")
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000, # 币安最大允许1000
            "startTime": current_ts
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            
            if not data or len(data) == 0:
                break
                
            # 存入列表
            for row in data:
                # 币安格式: [Open Time, Open, High, Low, Close, Volume, ...]
                data_list.append({
                    "timestamp": datetime.fromtimestamp(row[0]/1000), # 转成可读时间
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })
            
            # 更新下一次抓取的起点 (最后一条数据的时间 + 1个周期)
            # 1h = 3600秒 = 3600000毫秒
            last_time = data[-1][0]
            current_ts = last_time + 3600000 
            
            # 稍微休息一下，别把币安惹毛了
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ 下载出错: {e}")
            break

    # 3. 转成 DataFrame 并保存
    if len(data_list) > 0:
        df = pd.DataFrame(data_list)
        filename = f"Binance_{symbol}_{interval}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ 下载完成! 共 {len(df)} 行数据。")
        print(f"💾 已保存为: {filename}")
    else:
        print("❌ 未获取到任何数据。")

if __name__ == "__main__":
    # 下载 ETH
    get_binance_data("ETHUSDT", "1h", "2020-01-01")
    
    # 以后你想下 SOL 也可以这样：
    # get_binance_data("SOLUSDT", "1h", "2021-01-01")