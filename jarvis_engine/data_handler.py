"""
数据层：负责行情数据的加载与清洗。
通过抽象基类 BaseDataLoader 定义数据源接口，CSVDataLoader 为具体实现。
所有 DataFrame 保留时间索引与列名标准化；无 for 循环。
"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseDataLoader(ABC):
    """数据加载器抽象基类：统一数据源接口。"""

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        从数据源获取并清洗 OHLC 数据。

        Returns:
            以时间为索引的 DataFrame，至少包含 open, high, low, close。
            若失败或缺少时间列，返回空 DataFrame。
        """
        pass


class CSVDataLoader(BaseDataLoader):
    """从 CSV 文件加载行情数据并标准化列名与时间索引。"""

    def __init__(self, csv_path: str) -> None:
        """
        Args:
            csv_path: CSV 文件路径（通常来自 Config.DATA_PATH）。
        """
        self._csv_path: str = csv_path

    def fetch_data(self) -> pd.DataFrame:
        """
        读取 CSV 并清洗：统一列名小写、解析时间列、设置时间索引、补全 OHLC。
        无 for 循环，保持原有向量化与逻辑。
        """
        try:
            df: pd.DataFrame = pd.read_csv(self._csv_path, low_memory=False)
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return pd.DataFrame()

        if len(df) > 0 and (
            "http" in str(df.columns[0]) or "www" in str(df.columns[0])
        ):
            df = pd.read_csv(self._csv_path, skiprows=1, low_memory=False)

        df.columns = [c.strip().lower() for c in df.columns]

        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"])
        elif "unix" in df.columns:
            df["unix"] = pd.to_numeric(df["unix"], errors="coerce")
            unit: str = "ms"
            max_ts: float = float(df["unix"].max())
            if max_ts > 1e14:
                unit = "us"
            elif max_ts < 1e11:
                unit = "s"
            df["time"] = pd.to_datetime(df["unix"], unit=unit)
        elif "date" in df.columns:
            df["time"] = pd.to_datetime(df["date"])
        else:
            return pd.DataFrame()

        df = df.set_index("time").sort_index()

        if "close" not in df.columns and "open" in df.columns:
            df["close"] = df["open"]
        if "open" not in df.columns:
            df["open"] = df["close"]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]

        return df


def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    兼容函数：通过 CSVDataLoader 获取数据。
    供未迁移到 OOP 的调用方使用。

    Args:
        csv_path: CSV 文件路径。

    Returns:
        以时间为索引的 DataFrame，至少包含 open, high, low, close。
    """
    loader: CSVDataLoader = CSVDataLoader(csv_path)
    return loader.fetch_data()
