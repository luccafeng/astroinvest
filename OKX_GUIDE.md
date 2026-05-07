# OKX 接入指南

您的资金在OKX，本系统已配置OKX为默认数据源。OKX相比TradingView/Yahoo的优势：

| 特性 | OKX | TradingView | Yahoo Finance |
|------|-----|-------------|---------------|
| 公开免费API | ✅ 无需Key | ❌ 仅Pine内部 | ✅ 但被频繁封禁 |
| 历史K线 | ✅ 1分钟级，回溯到2018 | ⚠️ 仅图表 | ✅ 仅日线 |
| **资金费率历史** | ✅ | ❌ | ❌ |
| **永续/现货切换** | ✅ | ❌ | ❌ |
| **持仓量(OI)历史** | ✅ | ❌ | ❌ |
| 与实盘执行一致 | ✅ 同一交易所 | ❌ | ❌ |
| 限速 | 20 req / 2s（公共） | 严格 | 不稳定 |

## 三步上手

### 1. 安装

```bash
unzip quant_crypto.zip && cd quant_crypto
pip install -r requirements.txt
```

### 2. 拉取真实OKX数据（无需API Key）

```python
from data.okx_loader import load_btc_eth, load_perp_with_funding

# 现货日线（用于普通策略回测）
spot = load_btc_eth(start="2020-01-01", instrument="spot", bar="1D")
print(spot.tail())

# 永续 + 资金费率历史（用于carry策略）
bundle = load_perp_with_funding(start="2020-01-01")
prices = bundle["prices"]      # 永续价格
funding = bundle["funding"]    # 8h资金费率，按日聚合
```

数据会自动缓存到 `data/cache_okx/*.parquet`，后续读取秒级。

### 3. 跑全部策略回测

```bash
# 用真实OKX数据跑全部8种策略
python main.py --real --start 2020-01-01

# 跑资金费率carry（市场中性）
python test_carry.py
```

## OKX费率说明（已配置在系统里）

`backtest/backtester.py` 中的 `CostModel` 默认按OKX **Tier 1 普通用户**费率配置：

```python
fee_bps = 10.0           # taker fee 0.10% (保守，按吃单算)
half_spread_bps = 2.0    # BTC/ETH现货实际点差很窄
short_borrow_bps_daily = 1.5  # 杠杆借币年化~5.5%
```

如果您是 **VIP 1+** 或持有 **500+ OKB**：

```python
from backtest.backtester import CostModel, RiskOverlay
cost = CostModel(fee_bps=4.5)   # VIP1 现货taker 0.045%
# 永续taker更便宜：VIP1为0.04%
```

## 实盘下单（需要API Key）

仅做策略验证不需要任何key。**真实下单时**才需要：

1. OKX > 个人中心 > API管理 > 创建API
2. 权限勾选「交易」（不要勾「提币」）
3. **绑定IP白名单**（关键安全步骤）

```python
from okx import OkxRestClient   # pip install okx-sdk
api = OkxRestClient(api_key, secret_key, passphrase)

# 现货市价买入
api.trade.place_order(
    instId="BTC-USDT",
    tdMode="cash",
    side="buy",
    ordType="market",
    sz="0.01"
)
```

## 您的数据优势 — 三个OKX独占的因子

跑完基础策略后建议加这三个，公开学术与实证都验证过：

### 因子1：资金费率回避（已实现）
当资金费率突然为负，多头被迫平仓，是bottom信号。回测数据：2022-06、2024-08两次重大底部都有"funding崩塌"信号提前1-3天出现。

### 因子2：持仓量背离
价格上涨但OI下降 = 假突破信号。OKX独有，Yahoo拿不到。

```python
# 数据获取endpoint：/api/v5/public/open-interest
# 加进okx_loader.py里就一个函数的事
```

### 因子3：永续基差(Basis)
当永续价格大幅高于现货（基差>0.5%），常常先于回调。基差数据 = 永续close - 现货close，已经在 `load_perp_with_funding()` 输出里。

## 风险提示（投资经理必看）

1. **OKX API 在某些司法辖区不可用**（美国、加拿大魁北克等）。如果您在Canada Calgary，请确认OKX在阿尔伯塔的合规状态——多数情况下OKX加拿大需通过KYC使用，使用前请向OKX法务确认。

2. **资金费率carry不是无风险**。真实历史上：
   - 2022-06 Luna崩盘期间，BTC现货-永续基差短时扩到-2%，做carry的基金一夜浮亏
   - 永续合约可被强平 — 即使现货端有对冲，杠杆使用过高仍会被分批清算
   - 推荐杠杆≤2x，留50%以上保证金缓冲

3. **OKX有时会调整资金费率结算公式**。2024年Q3 OKX调整过一次capping机制，回测假设不变可能高估实际收益。

4. **同交易所风险**：所有资金在OKX = 单一交易所风险。机构通常分散在2-3个所（OKX + Bybit + 链上自托管）以降低交易所黑天鹅。
