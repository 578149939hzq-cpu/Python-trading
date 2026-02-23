# Stock Jarvis 量化交易系统 — 系统架构逻辑报告

> 基于工作区全局 Python 文件扫描，不涉及任何代码修改。  
> 扫描范围：`main.py`、`config.py`、`jarvis_engine/alpha.py`、`jarvis_engine/day19_forecast.py`、`jarvis_engine/day20_position.py`、`jarvis_engine/day12_ma_backtest_pro.py`、`experiments/`、`download_data.py`、`download_btc_clean.py` 等。

---

## 1. 模块分工（Module Map）

| 文件路径 | 职责简述 |
|----------|----------|
| **`config.py`** | **配置中心**：全局常量与策略参数。包含基础设施（`BASE_DIR`、`DATA_PATH`）、Alpha 策略参数（`STRATEGY_PARAMS`、`WEIGHTS`、`VOL_LOOKBACK`、RSI/趋势权重）、风险管理（`TARGET_VOLATILITY`、`MAX_LEVERAGE`、Regime/生存阈值）、回测仿真（`POSITION_BUFFER`、`FEE_RATE`、`INITIAL_CAPITAL`）。 |
| **`main.py`** | **入口与展示层**：流程编排 + 可视化 + 绩效统计。调用 `config`、`alpha` 完成「加载 → 信号 → 仓位 → 回测」流水线；实现全景报告（`plot_full_report`）、灾难快照（`plot_crash_snapshots`）、Sortino/回撤/交易统计等；不实现策略与回测算法本身。 |
| **`jarvis_engine/alpha.py`** | **策略与回测核心**：数据加载、信号生成、仓位目标、向量化回测一体化。`load_price_data` 负责数据清洗与标准化；`calculate_scaled_forecast` 做趋势+RSI 混合预测（Vol Scaling 在此用于信号归一化）；`calculate_position_target` 做 Regime Filter + Vol Scaling + Survival Stop + Buffer；`run_vectorized_backtest` 做收益/费用/资金曲线计算。生产主链路唯一使用的引擎。 |
| **`jarvis_engine/day19_forecast.py`** | **历史/教学用信号模块**：简化版 Carver 风格 EWMA 预测（单 span 列表、无 RSI、无配置中心），可独立运行验证；主流程未引用。 |
| **`jarvis_engine/day20_position.py`** | **历史/教学用仓位模块**：简化版「forecast/10 → clip → buffer → shift」仓位逻辑，无 Regime/Vol Scaling/Survival；依赖 `day19_forecast`，主流程未引用。 |
| **`jarvis_engine/day12_ma_backtest_pro.py`** | **独立回测/实验脚本**：内置 `load_price_data`、MA 策略、回测与绘图，用于均线策略完整演示；与主链路无依赖。 |
| **`experiments/Day 21_backtest.py`** | **回测实验/基准**：纯向量化回测（对数收益 + 手续费 + 资金曲线），带简单性能测试；逻辑与 `alpha.run_vectorized_backtest` 类似但独立实现，主流程未引用。 |
| **`experiments/day18_bollinger.py`** | **策略实验**：布林带均值回归策略 + 自带数据加载与回测；与主链路无依赖。 |
| **`experiments/day12_ma_backtest_advanced.py`** | **策略实验**：MA 策略 + 回测工具函数（资金曲线、最大回撤等）；与主链路无依赖。 |
| **`download_data.py`** | **数据获取**：从 Binance API 拉取 K 线（如 ETHUSDT 1h），保存为 CSV；不参与策略与回测。 |
| **`download_btc_clean.py`** | **数据获取**：Binance 历史数据下载并保存为「unix 毫秒」格式 CSV，供 Jarvis 读取；不参与策略与回测。 |

**小结**：生产主链路 = `config` + `main` + `jarvis_engine/alpha`；其余为实验、教学或独立工具，与主链路解耦。

---

## 2. 数据流向（Data Flow）— 以 `alpha.py` 为例

### 2.1 输入

- **原始 DataFrame**：来自 `load_price_data(Config.DATA_PATH)`。  
- 必备列：时间索引（`time`）、`open` / `high` / `low` / `close`（缺失时用 `close` 填充）。  
- 经过列名标准化、时间解析与排序，输出为「以时间为索引的 OHLC DataFrame」。

### 2.2 核心特征与信号计算（在 `alpha.py` 内）

1. **Vol Scaling（波动率归一化）**  
   - 在 `calculate_scaled_forecast` 中：`volatility = close.ewm(span=VOL_LOOKBACK).std() + 1e-8`。  
   - 趋势子预测：`(fast_ema - slow_ema) * scalar / volatility`，再按权重合成 `trend_forecast`。  
   - 作用：信号按波动率缩放，避免高波时信号过大。

2. **趋势信号（Trend Component）**  
   - 多组 (fast_span, slow_span, scalar) 得到多列 `fc_*`，加权求和 → `trend_forecast`，再 `clip(-20, 20)`。

3. **RSI 反转信号（含平滑与软死区）**  
   - 标准 RSI → 12 周期平滑 → `(50 - smooth_rsi)` → 软死区（\|diff\|≤10 输出 0，否则线性）→ 再 EMA(24) 平滑 → `rsi_forecast`，`clip(-20, 20)`。

4. **信号融合**  
   - `forecast = trend_forecast * TREND_WEIGHT + rsi_forecast * RSI_WEIGHT`（如 0.9/0.1）。  
   - 输出维度：**单列 `forecast`**，标量、按 bar 的连续信号（约 -20～+20）。

5. **Regime Filter（环境过滤）**  
   - 在 `calculate_position_target` 中：`regime_ma = close.rolling(REGIME_MA_WINDOW).mean()`，`is_bull_regime = close > regime_ma`。  
   - 熊市下杠杆上限降为 `BEAR_MODE_MAX_LEVERAGE`（如 1.0），牛市为 `MAX_LEVERAGE`（如 3）。

6. **Vol Scaling（仓位侧）**  
   - `ann_vol_pct = long_term_vol * sqrt(365*24)`，`raw_leverage_ratio = TARGET_VOLATILITY / ann_vol_pct`。  
   - `ideal_position = (forecast / 2.0) * raw_leverage_ratio`，再按 `dynamic_max_cap` clip。

7. **Survival Hard Stop（灾难熔断）**  
   - ATR(24) × multiplier 得到 `crash_threshold`，若 `hourly_ret < -crash_threshold` 则仓位归零，并记 `sigma_event` / `sl_threshold`。

8. **Buffer（缓冲器）**  
   - 仅当 \|目标仓位 - 当前仓位\| > buffer 时才调仓，得到 `buffered_pos`；再 `position = buffered_pos.shift(1)` 防未来函数。

### 2.3 最终输出维度（由 `alpha.py` 暴露给 `main`）

- **信号/风控相关**：`forecast`、`trend_forecast`、`rsi_forecast`、`position`、`raw_target`、`buffered_pos`、`leverage_ratio`、`ann_vol_pct`、`regime_ma`、`dynamic_max_cap`、`sl_threshold`、`sigma_event`。  
- **回测结果**（`run_vectorized_backtest` 产出）：`market_ret`、`net_ret`、`equity`、`buy_hold_equity`、`net_log_ret`、`market_log_ret` 等。  

即：**原始 DataFrame → 特征与 Vol Scaling/Regime Filter → 单维连续信号 `forecast` → 仓位 `position` → 资金曲线与收益序列**。

---

## 3. 架构师审查（CTO 视角）

### 3.1 模块解耦程度

- **优点**  
  - **配置与逻辑分离**：`config.py` 集中管理参数，`alpha.py` 通过 `getattr(Config, ...)` 读取，策略与运行参数不硬编码在引擎内。  
  - **主链路清晰**：`main` 只做编排与展示，不包含策略公式或回测实现；数据加载、信号、仓位、回测均收敛在 `alpha`，便于单点升级。  
  - **实验代码隔离**：`experiments/` 与 `day19`/`day20`/`day12_ma_backtest_pro` 与主流程无交叉依赖，不影响生产链路。

- **不足**  
  - **策略与回测同文件**：`alpha.py` 同时承担「数据加载、信号、仓位、回测」四类职责，未按「数据 / 策略 / 执行」拆成独立模块。  
  - **数据层未抽象**：`load_price_data` 与 CSV 路径强绑定，若未来接入数据库或实时行情，需改 `alpha` 或复制逻辑。  
  - **执行层缺失**：当前仅有回测仿真（`run_vectorized_backtest`），无实盘/模拟盘订单执行、风控执行等模块，无法评估「数据、策略、执行」中执行侧的解耦。

### 3.2 「数据、策略、执行」分离情况

| 层次 | 现状 | 评价 |
|------|------|------|
| **数据** | 由 `config` 指定路径，`alpha.load_price_data` 读 CSV 并做基础清洗；无统一 DataAdapter/Repository。 | **部分做到**：数据来源集中、格式统一，但未抽象为独立数据层接口。 |
| **策略** | 信号（`calculate_scaled_forecast`）与仓位/风控（`calculate_position_target`）均在 `alpha` 内，参数来自 `config`；策略逻辑与数据 I/O、回测写在同一文件。 | **部分做到**：策略可配置、可替换权重与参数，但未拆成独立「策略模块」与接口（如只接受 DataFrame 输入并输出 `forecast`/`position`）。 |
| **执行** | 仅有「模拟执行」：`run_vectorized_backtest` 用 `position` 与价格序列算收益与资金曲线；无订单、撮合、实盘网关。 | **未分离**：执行仅等于回测中的收益计算，无独立执行层或执行接口。 |

### 3.3 总结与建议

- **结论**：当前架构在「配置与主流程分离」「实验与生产隔离」上做得较好；**尚未做到清晰的「数据、策略、执行」三层分离**。策略与回测集中在 `alpha.py`，数据与执行未抽象成独立层次。  
- **建议（仅方向，不要求改代码）**：  
  - **数据层**：抽象出「数据源接口」（如 `load_ohlc(symbol, start, end)`），由 CSV/DB/API 分别实现，`alpha` 只消费标准化 DataFrame。  
  - **策略层**：将 `calculate_scaled_forecast` 与 `calculate_position_target` 抽到独立模块（或保留在 `alpha` 但通过清晰接口调用），输入/输出约定为 DataFrame 列名与语义，便于多策略切换与单元测试。  
  - **执行层**：将「给定 position 序列 + 行情 → 收益与资金曲线」抽象为回测执行器；若未来有实盘，可增加「订单生成 → 撮合/模拟撮合 → PnL」的执行层，与策略层通过「目标仓位」或「信号」接口对接。

以上为基于现有代码的静态架构逻辑报告，未对任何实现进行修改。
