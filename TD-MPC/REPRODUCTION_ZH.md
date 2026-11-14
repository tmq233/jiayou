# TD-MPC 复现工程

论文：Temporal Difference Learning for Model Predictive Control，ICML 2022；用户提供 arXiv:2203.04955v2（2022-07-19）。
官方源码：https://github.com/nicklashansen/tdmpc
固定提交：f4d85eca7419039b71bab2234ffc8aca378dd313

本目录包含该提交的源码与许可证，未改动算法。新增启动脚本仅用于分阶段验证。当前尚未连接 GPU 服务器，尚未训练或验证依赖安装，不代表论文结果已经复现。

## 首轮范围

先选 DMControl 的 state / walker-walk，正式运行 5 个种子（1–5），每次评估 10 个 episode。默认 500,000 环境步，action_repeat=2；配置 train_steps=250,000 表示智能体决策次数。官方循环包含最后一整个 episode，实际交互次数略超预算；保留原始行为。

论文表 4 的普通 state 任务参数与源码默认值核对：discount=0.99，horizon=5，population=512，elites=64，iterations=6，latent_dim=50，batch_size=512，lr=0.001；reward/value/consistency 系数为 0.5/0.1/2。源码 tau=0.01 是新参数权重，对应论文旧参数权重 0.99。源码 buffer 上限为 1,000,000；论文列为不限，在此任务预算内不构成截断。

## 服务器执行

需要 Linux、NVIDIA GPU、可用驱动以及 MuJoCo。官方 environment.yaml 使用 Python 3.8、PyTorch 1.9、CUDA 11.1、Gym 0.21 和旧版 dm-control；现代服务器上的安装兼容性需实测，不能直接假定能安装。先检查服务器与已有环境，再决定保留旧栈或使用独立的兼容性修补环境。

原始安装入口：

```bash
conda env create -f environment.yaml
conda activate tdmpc
bash run_reproduction.sh smoke
bash run_reproduction.sh full
```

smoke 保持网络与规划规模，缩短训练和随机探索阶段，只验证环境交互、梯度更新、评估和模型保存，不能用于论文性能对比。full 保留官方任务参数。不要在同一实验目录重复运行，以免覆盖日志与模型。

结果位于 logs/walker-walk/state/reproduction/<seed>/eval.log 与 models/model.pt；终端输出和环境信息在 run_records/。本地日志默认关闭 W&B，不需要 API key。官方代码的视频记录依赖 W&B，本脚本未启用视频。

## 验收

1. smoke 无异常完成，损失有限，生成评估日志与模型。
2. 5 个种子均达到完整训练预算。
3. 根据 env_step 对齐各种子的评估曲线，汇总均值与 95% 置信区间；明确区间算法。
4. 对照 results/ 中官方 walker-walk 数据，并注明环境版本、源码版本与任何修改。
5. 此工程只覆盖首轮单任务复现；全论文 DMControl、pixels、Meta-World 与消融尚未执行。
