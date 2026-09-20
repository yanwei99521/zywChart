# 比特币幂律与黄金相对强弱

这是一张基于真实周度数据的独立复算图，不是对原 PNG 的图像编辑，也不是 Fidelity/FMRC 的官方数据产品。

## 数据

- 比特币：Coin Metrics Community API，`PriceUSD` 日线，按周日取最后一个有效值。
- 黄金：Yahoo Finance 的 COMEX Gold Futures 连续合约 `GC=F`，周线收盘。它是可公开复算的黄金代理，不等同于需要许可的 LBMA Gold Price。
- 截止日：2026-09-20；比特币最新有效日为 2026-09-19，统一映射到周末。

## 计算

令 `t` 为距比特币创世区块（2009-01-03）的天数，`P` 为 BTC/USD 周价：

1. 幂律中枢：`ln(P) = a + b × ln(t)`，用全部可用周度数据做 OLS。
2. 幂律支撑：`support = trend / e`。
3. BTC 相对幂律偏离：`100 × ln(P / trend)`。因此支撑线对应 `-100%`。
4. BTC/黄金 52 周 Z-score：先算 `BTC价格 / 黄金每盎司价格`，再以 52 周滚动均值和总体标准差标准化，最后乘 100。

## 复算

```bash
MPLCONFIGDIR=/private/tmp/mplconfig OPENBLAS_NUM_THREADS=1 python3 build_chart.py \
  --btc-json /path/to/coinmetrics-btc.json \
  --gold-csv /path/to/yfinance-gold.csv \
  --output-dir . \
  --as-of 2026-09-20
```

输出包括 PNG、SVG、`model-summary.json` 和 `data/weekly_model_data.csv`。

## 公开 API 与自动更新

项目包含一个无需 API Key 的 FastAPI 服务。服务按北京时间每天 `07:00` 和
`18:00` 自动下载最新数据并重绘；如果刷新失败，上一版图表会继续对外提供，
不会被半成品覆盖。首次启动且没有任何 PNG 时，会额外执行一次初始化刷新。

公开接口：

| 接口 | 内容 |
| --- | --- |
| `GET /api/v1/charts/bitcoin-power-law/latest` | 最新 PNG，固定地址 |
| `GET /api/v1/charts/bitcoin-power-law/latest.png` | 最新 PNG |
| `GET /api/v1/charts/bitcoin-power-law/latest.svg` | 最新 SVG |
| `GET /api/v1/charts/bitcoin-power-law/summary` | 最新模型摘要 JSON |
| `GET /api/v1/charts/bitcoin-power-law/data.csv` | 完整周度数据下载 |
| `GET /api/v1/health` | 图表与调度状态 |
| `GET /docs` | OpenAPI 交互文档 |

本地启动：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/uvicorn chart_service.app:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

浏览器或其他网站可以直接引用：

```html
<img src="https://你的域名/api/v1/charts/bitcoin-power-law/latest.png"
     alt="比特币幂律与黄金相对强弱">
```

也可以使用 Docker Compose：

```bash
docker compose up -d --build
```

服务监听 `8000` 端口。公网使用时应在前方配置 Nginx、Caddy 或 Cloudflare
Tunnel，绑定域名并启用 HTTPS。调度器在 API 进程内运行，因此必须保持
`--workers 1`；若需要水平扩容，应把定时刷新拆为单独任务。

接口不要求鉴权，CORS 允许所有来源读取。公网反向代理仍建议配置基础限流，
避免恶意请求消耗带宽。图表代码与数据源的许可是两件事；尤其 yfinance 项目
明确提示 Yahoo 数据面向个人使用，公开或商业分发前应自行确认上游条款。

## 边界

幂律是历史拟合，不是“价格地板”。改变起始日期、采样频率、价格源或拟合窗口，都会改变中枢、支撑和偏离读数。黄金连续期货还包含换月影响，因此该图适合做长期相对强弱观察，不应被解释为精确交易信号。
