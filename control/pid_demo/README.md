# PID 车辆定速巡航 — 交互演示

> 用车控场景学 PID：油门控制车速，3 个 Demo，每个都是 A/B 对比，新手也能看懂。

## 环境准备

### 1. 安装 Python 3.10+

[python.org](https://www.python.org/downloads/) 下载，安装时勾选 **"Add Python to PATH"**。

```bash
python --version
```

### 2. 进入目录 & 安装依赖

```bash
cd pid_demo
pip install -r requirements.txt
```

### 3. 运行

```bash
python main.py
```

输入 1-3 选择 Demo，或 `a` 全部依次运行。

---

## 场景与内容

| # | 场景 | 对比什么 |
|---|------|---------|
| 1 | **定速巡航 (三种工况)** | 位置式 vs 增量式 — 正常/油门到底/变速 |
| 2 | **爬陡坡** | 无保护 vs Clamping — 积分饱和如何导致超速 |

每个 Demo 会：
- 在终端讲清楚"你在开什么车、出了什么事"
- 打印实验条件、对比指标、结论
- 弹出图表（关闭图表窗口后继续）

---

## 项目结构

```
pid_demo/
├── main.py                 # 主入口
├── requirements.txt        # numpy, scipy, matplotlib
├── README.md
├── utils/
│   ├── pid_controllers.py  # Positional / Incremental / Clamping / Deadzone
│   ├── simulation.py       # 车辆巡航仿真引擎
│   └── plot_setup.py       # 中文字体
└── demos/
    ├── demo_01_positional_vs_incremental.py
    ├── demo_02_anti_windup.py
    └── demo_03_deadzone.py
```
