import requests
import pandas as pd
import time
from datetime import datetime
def download_binance_data(symbol, start_date, filename):
    print(f"🚀 开始从币安下载清洗版 {symbol} 数据...")
    
    # 1. 转换时间为毫秒时间戳
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    data_list = []
    current_ts = start_ts
    
    # 2. 循环抓取
    while current_ts < end_ts:
        print(f"   ⏳ 下载进度: {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')}")
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1h",
            "limit": 1000,
            "startTime": current_ts
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            
            if not data or len(data) == 0:
                break
                
            for row in data:
                data_list.append({
                    "timestamp": row[0], # 币安原生就是毫秒，非常标准
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })
            
            # 更新下一次起点
            last_time = data[-1][0]
            current_ts = last_time + 3600000 
            time.sleep(0.1) # 防止被封IP
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            break

    # 3. 保存
    if len(data_list) > 0:
        df = pd.DataFrame(data_list)
        # 币安的时间戳是 unix 毫秒，我们直接保存，你的 Jarvis 现在能识别它
        df.rename(columns={"timestamp": "unix"}, inplace=True) 
        df.to_csv(filename, index=False)
        print(f"\n✅ 成功！纯净数据已保存为: {filename}")
        print(f"📊 总行数: {len(df)}")
    else:
        print("❌ 下载失败，没有数据。")

if __name__ == "__main__":
    # 下载 2018 年至今的 BTC 数据
    download_binance_data("BTCUSDT", "2018-01-01", "Binance_BTCUSDT_1h.csv")