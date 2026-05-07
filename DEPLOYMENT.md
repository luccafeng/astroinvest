# 部署指南 (Deployment Guide)

完整的从零到出回测结果的部署流程。**预计用时10-15分钟**。

---

## 第零步：合规预检（30秒）

**OKX在加拿大的状态（2026年）**：
- ❌ 不接受加拿大**新用户**注册（自2023年3月起）
- ✅ **公共行情API依然可访问**（无需登录、无需KYC）—— 您可以放心做回测
- ⚠️ 实盘下单/出入金可能受限 —— 实盘前需确认账户状态

**所以：**
- **回测/研究阶段** → 直接用OKX，本指南直接走
- **实盘阶段** → 切换到 **Kraken（加拿大持牌，最合规）** 或 **Bybit**，本框架已预留接口

---

## 第一步：环境准备（5分钟）

### 1.1 检查Python版本

需要 **Python 3.10 或更高**。打开终端：

**macOS / Linux：**
```bash
python3 --version
```

**Windows（PowerShell）：**
```powershell
python --version
```

如果版本低于3.10或没装：
- **macOS**：`brew install python@3.12`（如果没brew，先装[Homebrew](https://brew.sh)）
- **Windows**：从 https://www.python.org/downloads/ 下载，**安装时务必勾选 "Add Python to PATH"**
- **Linux (Ubuntu/Debian)**：`sudo apt update && sudo apt install python3.12 python3.12-venv`

### 1.2 解压项目

把 `quant_crypto.zip` 放到一个您喜欢的目录，比如：
- macOS：`~/projects/quant_crypto/`
- Windows：`C:\Users\您的用户名\projects\quant_crypto\`

```bash
# macOS / Linux
cd ~/projects
unzip quant_crypto.zip
cd quant_crypto

# Windows PowerShell
cd C:\Users\$env:USERNAME\projects
Expand-Archive quant_crypto.zip -DestinationPath .
cd quant_crypto
```

### 1.3 创建虚拟环境（强烈推荐，避免污染系统Python）

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

⚠️ Windows如果遇到"无法加载脚本"错误，先以管理员身份运行：
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

激活成功后，命令行前会出现 `(venv)`。**之后每次打开新终端都要先 activate**。

### 1.4 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

预计需要 **2-3分钟** 下载安装。如果下载慢（中国大陆网络）：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.5 验证安装

```bash
python -c "import pandas, numpy, matplotlib, requests, yfinance; print('所有依赖OK')"
```

应输出 `所有依赖OK`。

---

## 第二步：验证数据接入（2分钟）

### 2.1 测试OKX公共API连通性

```bash
python -c "
import requests
r = requests.get('https://www.okx.com/api/v5/market/candles',
                 params={'instId':'BTC-USDT','bar':'1D','limit':3}, timeout=10)
print(r.json())
"
```

**预期输出**：JSON数据，包含3根最近的BTC日K线。

如果这一步失败：
- **报错"connection timeout"** → 网络无法访问OKX。中国大陆需要代理，加拿大可能因区域限制需要VPN（但**仅用于读取行情数据本身不违反加拿大法律**，这是公开数据）
- **报错"403 Forbidden"** → IP被拦截，换网络或用VPN
- **方案B：跳过OKX，用Binance**（公共API在加拿大可访问，无限制）

### 2.2 拉取真实历史数据

```bash
python data/okx_loader.py
```

**预期输出**（约30秒，取决于网络）：
```
=== Testing OKX loader ===
[okx_loader] fetching BTC-USDT from OKX...
  ...page 5, oldest fetched = 2024-08-XX
  -> fetched 365 bars [2024-01-XX -> 2024-12-XX]
[okx_loader] fetching ETH-USDT from OKX...
  ...
Days: 365  BTC ann.vol: 47.2%  ETH ann.vol: 60.1%  Corr: 0.812
```

数据自动缓存在 `data/cache_okx/` 目录下，下次秒级读取。

### 2.3 如果OKX访问失败 — 切换Binance

编辑 `data/data_loader.py`，找到这一行：
```python
def load_real_prices(..., source: str = "okx", ...):
```
改成：
```python
def load_real_prices(..., source: str = "binance", ...):
```

或在调用时显式指定：
```python
prices = get_prices(use_real=True, source="binance", start="2020-01-01")
```

---

## 第三步：跑第一次真实回测（3分钟）

### 3.1 全策略回测

```bash
python main.py --real --start 2020-01-01
```

**预期输出**（首次约2分钟，包含数据下载；后续约15秒）：
```
Loaded 2200 days  (2020-01-01 -> 2026-01-XX)
BTC ann.vol: 47.2%  ETH ann.vol: 60.1%  BTC-ETH corr: 0.812

========================================================================
  FULL SAMPLE: 2020-01-01 -> 2026-01-XX  (2200 days)
========================================================================

--- FULL-SAMPLE PERFORMANCE ---
                              cagr  ann_vol  sharpe  ...
Buy & Hold 60/40             0.X    0.X      X.XX
MA Crossover (50/200)        0.X    0.X      X.XX
Donchian Breakout (20/55)    0.X    0.X      X.XX
...
```

**真实数据上的Sharpe大概率会比模拟数据低**（这是正常的、健康的）。模拟数据上Donchian Sharpe 2.5是上限，真实数据期望 0.7-1.2。

### 3.2 生成全套图表

```bash
python run_full.py
```

完成后查看 `reports/` 目录，会有7张PNG图：
- `01_prices.png` - 价格总览
- `02_equity_curves.png` - 各策略净值曲线
- `03_drawdowns.png` - 回撤曲线
- `04_rolling_sharpe.png` - 滚动Sharpe
- `05_return_distribution.png` - 收益分布
- `06_strategy_correlation.png` - 策略相关性热图

### 3.3 跑参数鲁棒性检验

```bash
python robustness.py
```

输出参数Sharpe矩阵 + 保存 `reports/07_parameter_heatmap.png`。**这一步是您评估策略可信度的核心** —— 如果Sharpe在邻域内剧烈变化（最高3.0最低0.2这种），就是过拟合，不能用。

### 3.4 跑资金费率carry策略

```bash
python test_carry.py
```

注意：这一步**模拟了资金费率**（因为完整历史carry回测需要单独拉取funding-rate-history数据）。要用真实funding：

```python
from data.okx_loader import load_perp_with_funding
bundle = load_perp_with_funding(start="2022-01-01")  # 注意：约5分钟拉取
# 然后调用 backtest_perp_carry(bundle['prices'], bundle['funding'])
```

---

## 第四步：日常使用（您回测调研的工作流）

### 4.1 测试新策略想法

打开 `strategies/strategies.py`，复制任何一个策略类，改逻辑：

```python
class MyNewStrategy(Strategy):
    name = "我的策略"
    def generate_weights(self, prices):
        # 返回一个 (天数 × 资产数) 的目标权重 DataFrame
        # 取值在 [-1, +1] 之间
        # +1 = 满仓多, -1 = 满仓空, 0 = 空仓
        ...
        return weights
```

加到 `ALL_STRATEGIES` 列表里：
```python
ALL_STRATEGIES = [..., MyNewStrategy()]
```

`python main.py --real` 一跑，结果立刻出来。

### 4.2 调整成本模型

如果您是**OKX VIP1或持有500+OKB**：
```python
# main.py 或 run_full.py 里
cost = CostModel(fee_bps=4.5)   # VIP1现货taker
```

### 4.3 改风控参数

`backtest/backtester.py` 里的 `RiskOverlay`：
```python
risk = RiskOverlay(
    target_vol=0.15,         # 把目标波动从25%降到15% (更保守)
    max_leverage=1.0,        # 不加杠杆
    drawdown_kill_pct=0.25,  # 25%回撤就强制空仓 (更敏感)
)
```

### 4.4 切换时间窗口做样本外测试

```bash
python main.py --real --start 2020-01-01 --end 2023-12-31  # 训练期
python main.py --real --start 2024-01-01                    # 样本外
```

---

## 第五步（可选）：实盘部署

**警告：在做了至少3个月模拟+小资金验证之前，不要上实盘。**

### 5.1 切换到加拿大合规交易所

如果您要正式实盘交易，建议**Kraken**（FINTRAC注册，加拿大合规）：

```bash
pip install krakenex
```

然后我可以帮您写 `data/kraken_loader.py`（结构和OKX完全一样，几小时就能搞定）。

### 5.2 创建API Key（以OKX为例，其它所类似）

1. 登录OKX → 个人中心 → API管理 → 创建API
2. 权限：**只勾「交易」，绝对不要勾「提币」**
3. **绑定IP白名单**（您VPS或本机的固定IP）
4. 获得 `api_key`, `secret_key`, `passphrase`

### 5.3 安全存储Key（关键）

**绝对不要**把Key写在代码里。用环境变量：

```bash
# macOS/Linux: 加到 ~/.zshrc 或 ~/.bashrc
export OKX_API_KEY="您的key"
export OKX_SECRET="您的secret"
export OKX_PASSPHRASE="您的passphrase"

# Windows PowerShell (永久):
[System.Environment]::SetEnvironmentVariable("OKX_API_KEY", "您的key", "User")
```

代码里读取：
```python
import os
api_key = os.environ["OKX_API_KEY"]
```

### 5.4 上云24/7运行（实盘必需）

最便宜可靠：**Vultr / DigitalOcean / Hetzner**，$6/月起。

新加坡或东京机房延迟到OKX最低（~5ms）。**纽约/伦敦机房延迟>200ms，不要用**。

---

## 故障排查 (Troubleshooting)

### 问题1：`pip install` 报错 "error: Microsoft Visual C++ 14.0 is required"
- **Windows专属**。装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 即可

### 问题2：`No module named 'data'`
- 确保您在项目根目录 `quant_crypto/` 下运行命令，不是在子目录里

### 问题3：图表中文显示成方框
- macOS：默认字体支持中文
- Linux：`sudo apt install fonts-noto-cjk`
- Windows：默认字体支持中文
- 已生成的图表全部用英文标注，避免此问题

### 问题4：OKX返回 `"code":"50101","msg":"APIKey does not match current environment"`
- 您的Key是Demo Trading的，不是Production的，反之亦然。重新创建对应环境的Key

### 问题5：回测结果与模拟数据差异很大
- **正常的**。模拟数据偏理想化。真实数据下Sharpe下降30-50%是预期内的。如果差超过70%，说明策略对数据生成过程过拟合，需要重新设计

### 问题6：在加拿大无法访问OKX
- **公共行情API在加拿大通常可访问**（不涉及登录）
- 如果连接被阻断，临时方案是用VPN连香港/新加坡节点拉取行情数据（注意：仅做数据获取，不要用VPN账号操作交易）
- 或直接切换到 Binance/Kraken 数据源

---

## 部署完成检查清单

- [ ] Python 3.10+ 已安装并验证
- [ ] 项目已解压到工作目录
- [ ] 虚拟环境已创建并激活
- [ ] `requirements.txt` 全部依赖安装成功
- [ ] OKX/Binance公共行情API测试成功
- [ ] `python main.py --real` 跑出真实回测结果
- [ ] `reports/` 目录有图表生成
- [ ] 已理解**不要用任何API Key**直到充分验证策略
- [ ] 已确认OKX在加拿大的实盘合规状态（如果计划实盘）

---

## 您下一步该做什么

按优先级：

1. **本周**：跑通本指南，让真实数据上的8种策略全部出结果。看看哪些策略真实数据下还活着（很多模拟里好看的策略真实数据下会死掉，这是研究的开始）

2. **下周**：用 `robustness.py` 做参数稳健性扫描。**剔除任何参数邻域内Sharpe方差>50%的策略** —— 这些是过拟合的，留着只会害您

3. **第三周**：拉取真实funding rate数据，跑carry策略。这是我个人认为最靠谱的"alpha"，因为有清晰的经济驱动

4. **第四周开始**：选 1-2 个表现最稳的策略，开始 paper trading（纸面交易，记录但不真下单），至少 3 个月

5. **3个月后**：用<5%账户资金做小规模实盘，重点观察滑点是不是和回测假设一致

**任何阶段卡住都直接告诉我，我帮您一起debug。**
