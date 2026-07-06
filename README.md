# V9 Tensor-Memetic: LLM-Enhanced Memetic Search via Tensorized Population

基于 LLM 全局语义注入 + 张量化种群搜索 + PPO 联合调度的组合优化框架，支持 **最小支配集 (MDS)** 和 **最小顶点覆盖 (MVC)** 两大 NP-hard 问题。

---

## 目录

- [核心思想](#核心思想)
- [架构总览](#架构总览)
- [V9 模块详解](#v9-模块详解)
  - [1. 算子层 (operators)](#1-算子层-operators)
  - [2. 环境层 (env)](#2-环境层-env)
  - [3. 编码器 (encoder)](#3-编码器-encoder)
  - [4. 策略网络 (policy)](#4-策略网络-policy)
  - [5. 智能体 (agent)](#5-智能体-agent)
- [V9MVC 领域扩展](#v9mvc-领域扩展)
- [训练流程伪代码](#训练流程伪代码)
- [推理流程伪代码](#推理流程伪代码)
- [后处理流程](#后处理流程)
- [基线算法](#基线算法)
- [实验结果](#实验结果)
- [项目目录结构](#项目目录结构)
- [快速开始](#快速开始)

---

## 核心思想

V9 将 **元启发式 Memetic 搜索** 与 **深度强化学习** 统一为端到端可微框架：

1. **张量化种群**：维护 B 个并行解（种群矩阵），每步由 GNN 编码器统一处理
2. **LLM 全局语义注入**：通过 SentenceTransformer 编码图描述文本，经 `feature_proj` 层注入每个节点的 GNN 输入特征
3. **双头策略调度**：GNN 编码器输出图级上下文 → 两个独立 MLP 头分别选择 **破坏算子** 和 **修复算子**
4. **模拟退火接受**：环境内嵌 SA 机制决定是否接受劣解，平衡探索与利用
5. **PPO-Clip 联合更新**：将 destroy/repair 两个决策的对数概率相加作为联合策略梯度

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                    V9 Tensor-Memetic 架构                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  输入: PyG Data (x, edge_index, global_emb)                  │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────┐             │
│  │         MemeticGNNEncoder                    │             │
│  │  [x, global_emb] → feature_proj → LayerNorm │             │
│  │       → GCN×3 → global_max_pool             │             │
│  │       → global_mean_pool → concat            │             │
│  │              → Q_context (2×hidden)          │             │
│  └──────────────────┬──────────────────────────┘             │
│                     ↓                                        │
│       ┌─────────────┼─────────────┐                          │
│       ↓             ↓             ↓                          │
│  mlp_destroy   mlp_repair   mlp_critic                       │
│  (4-class)     (2-class)     (1-value)                       │
│       ↓             ↓             ↓                          │
│  prob_d        prob_r         V(s)                           │
│       ↓             ↓                                        │
│  ┌─────────────────────────────────────┐                     │
│  │       TensorMemeticEnv              │                     │
│  │  Pop[D₁..D_B] → destroy → repair   │                     │
│  │  → SA accept/reject → elite update │                     │
│  │  → T ← T × α (cooling)            │                     │
│  └─────────────────────────────────────┘                     │
│         ↓                                                    │
│  reward = max(0, -Δ|D|)  (仅改善时奖励)                      │
│         ↓                                                    │
│  PPO-Clip 联合梯度更新                                       │
│  L = L_actor + 0.5·L_critic - 0.01·H(π)                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## V9 模块详解

### 1. 算子层 (operators)

[src/v9/operators.py](src/v9/operators.py) 定义了 Memetic 搜索的核心算子：

**破坏算子 (Destroy Operators)** — 从当前解 D 中移除部分节点：

| 算子 | 策略 | 说明 |
|------|------|------|
| `random_remove` | 随机 | 移除 ⌈\|D\|/5⌉ 个随机节点 |
| `worst_remove` | 贪心 | 移除 Private Neighbor (PN) 最大的 ⌈\|D\|/5⌉ 个节点（贡献最小） |
| `crossover_destroy` | 精英交叉 | 保留 D ∩ D_elite，不足 50% 时用高度数节点补充 |
| `community_bombing_destroy` | 社区轰炸 | 随机选 1-2 个 Louvain 社区，移除社区内所有 D 中节点 |

**修复算子 (Repair Operators)** — 将部分解扩展为可行解：

| 算子 | 策略 | 说明 |
|------|------|------|
| `greedy_repair` | 贪心增益 | 每步选覆盖未覆盖节点最多的节点加入 D |
| `hub_repair` | 度中心性 | 每步选度最大的未覆盖节点加入 D |

**Private Neighbor (PN)** 定义：节点 u 的 PN 集合 = N[u] \ ∪_{v∈D,v≠u} N[v]，即仅被 u 覆盖的节点集合。PN 大 = u 的覆盖贡献大 = 不应移除。

### 2. 环境层 (env)

[src/v9/env.py](src/v9/env.py) — `TensorMemeticEnv` 实现 MDP 环境：

**状态空间**：
- `cover`: [B, N] 张量，每个种群成员的节点覆盖状态
- `in_D`: [B, N] 张量，每个种群成员的节点选择状态
- `D_elite_size`: 标量，当前精英解大小

**动作空间**：
- `action_d`: 离散 {0,1,2,3}，选择破坏算子
- `action_r`: 离散 {0,1}，选择修复算子

**奖励**：
```
Δ|D| = |D_candidate| - |D_current|
if Δ|D| < 0:  accepted, reward = -Δ|D|
else:         accepted with prob exp(-Δ|D|/T), reward = 0
```

**终止条件**：步数 ≥ max_steps 或温度 T < T_min

**初始化**：使用带噪声的贪心（同等增益随机选择）生成 B 个初始解，增加种群多样性。

**Louvain 社区检测**：reset 时调用 `community_louvain.best_partition()` 获取社区结构，供 `community_bombing_destroy` 使用。

### 3. 编码器 (encoder)

[src/v9/encoder.py](src/v9/encoder.py) — `MemeticGNNEncoder`：

```
输入:
  x:           [B×N, d_in]       节点特征（度归一化 + 覆盖状态）
  edge_index:  [2, B×E]          批量边索引
  global_emb:  [B, 384]          LLM 全局嵌入（SentenceTransformer）
  batch:       [B×N]             批量归属向量

流程:
  1. global_emb 扩展到每个节点: [B×N, 384]
  2. 拼接: h = [x, global_emb]  → [B×N, d_in+384]
  3. feature_proj: h → [B×N, hidden_dim]
  4. LayerNorm + ReLU
  5. GCN × 3 层 (中间层 ReLU + Dropout)
  6. global_max_pool → [B, hidden_dim]
  7. global_mean_pool → [B, hidden_dim]
  8. concat → Q_context: [B, 2×hidden_dim]
```

**LLM 注入机制**：`global_emb` 由 SentenceTransformer (all-MiniLM-L6-v2) 对图的文本描述编码得到，384 维向量通过 `feature_proj` 线性层映射到 hidden_dim 并与节点特征融合，实现全局语义的逐节点广播。

### 4. 策略网络 (policy)

[src/v9/policy.py](src/v9/policy.py) — `MemeticSchedulerNet`：

```
输入:
  data: PyG Data (x, edge_index, global_emb)
  cover_states_batch: [B, N] 种群覆盖状态

流程:
  1. 扩展节点特征: x → [B×N, d_in], 拼接 cover → [B×N, d_in+1]
  2. 构建批量边索引: edge_index + offsets
  3. GNN 编码: Q_context = encoder(x_node, edge_index_batch, global_emb, batch)
  4. 三头输出:
     prob_d  = softmax(mlp_destroy(Q_context))  → [B, 4]
     prob_r  = softmax(mlp_repair(Q_context))    → [B, 2]
     values  = mlp_critic(Q_context)             → [B, 1]
```

**批量前向**：单张图 × B 个种群成员在一次 GNN 前向中处理，通过偏移边索引实现高效批量化。

### 5. 智能体 (agent)

[src/v9/agent.py](src/v9/agent.py) — `PPOAgentV9`：

**训练模式 (`run_episode`)**：
- 采样 destroy/repair 动作：`dist.sample()`
- 收集轨迹：log_prob_d, log_prob_r, value, reward, entropy

**推理模式 (`run_episode`)**：
- 贪心选择：`prob.argmax()`

**PPO 更新 (`ppo_update`)**：
1. 对每条轨迹计算 GAE 优势估计
2. 联合对数概率：`old_log_prob = log_prob_d + log_prob_r`
3. PPO-Clip 比率裁剪：`ratio = exp(new - old), clip(ratio, 1-ε, 1+ε)`
4. 总损失：`L = L_actor + 0.5·L_critic - 0.01·H(π)`

---

## V9MVC 领域扩展

[src/v9_mvc/](src/v9_mvc/) 将 V9 从 MDS 扩展到 MVC 问题：

| 组件 | MDS (V9) | MVC (V9MVC) |
|------|----------|-------------|
| 覆盖定义 | 节点 u ∈ D → N[u] 全覆盖 | 节点 u ∈ D → 所有 (u,v) 边覆盖 |
| worst_remove | 移除 PN 最大的节点 | 移除 PE (Private Edge) 最小的节点 |
| greedy_repair | 选覆盖最多未覆盖节点的节点 | 选覆盖最多未覆盖边的节点 |
| hub_repair | 选度最大的未覆盖节点 | 选度最大的边端点 |
| 后处理 | `hybrid_optimize_dominating_set` | `mvc_hybrid_optimize` |
| 环境状态 | 节点覆盖向量 | 边覆盖比率向量 |

**Private Edge (PE)** 定义：节点 u 的 PE 数 = |{v ∈ N(u) : v ∉ D}|，即 u 独占覆盖的边数。PE 小 = u 的边覆盖贡献小 = 优先移除。

V9MVC 复用 V9 的 `MemeticGNNEncoder` 和 `MemeticSchedulerNet`，仅替换环境和算子。

---

## 训练流程伪代码

```
Algorithm: V9 Training (PPO)
─────────────────────────────
Input: Training graphs G_train, config
Output: Trained policy θ

Initialize MemeticSchedulerNet θ
Initialize PPOAgent with optimizer Adam(θ, lr=3e-4)

FOR epoch = 1 TO max_epochs:
    Shuffle G_train
    all_trajectories = []

    FOR each graph G in G_train:
        env = TensorMemeticEnv(G, batch_size=pop_size)
        state = env.reset()          // B greedy+noise 初始解
        B_trajectories = [{log_probs, values, rewards, ...}] × B

        WHILE NOT env.is_done():
            // ── 策略前向 ──
            prob_d, prob_r, V = policy_θ(G, state.cover)

            // ── 采样动作 ──
            a_d ~ Categorical(prob_d)    // destroy 算子
            a_r ~ Categorical(prob_r)    // repair 算子

            // ── 环境交互 ──
            state', reward, done, info = env.step(a_d, a_r)
            // reward = max(0, -Δ|D|) if accepted, else 0

            // ── 记录轨迹 ──
            FOR i = 1 TO B:
                trajectories[i].append(
                    log_prob_d[i], log_prob_r[i],
                    V[i], reward[i], entropy[i]
                )
        END WHILE

        elite_size = info['elite_size']
        all_trajectories.extend(B_trajectories)
    END FOR

    // ── PPO 更新 ──
    FOR each trajectory:
        advantages, returns = GAE(rewards, values, γ=0.99, λ=0.95)
    Normalize advantages across all trajectories

    FOR ppo_epoch = 1 TO 2:
        old_log_prob = log_prob_d + log_prob_r

        // 重新前向计算 new_log_prob
        new_log_prob_d, new_log_prob_r, new_V = policy_θ(data, covers)
        new_log_prob = new_log_prob_d + new_log_prob_r

        ratio = exp(new_log_prob - old_log_prob)
        L_actor = -min(ratio·A, clip(ratio, 0.8, 1.2)·A).mean()
        L_critic = 0.5 · MSE(new_V, returns)
        H = mean(entropy_d + entropy_r)

        L_total = L_actor + 0.5·L_critic - 0.01·H
        θ ← θ - ∇_θ L_total    (with grad clip = 0.5)
    END FOR

    // ── 周期性评估 ──
    IF epoch % 5 == 0:
        eval_ds = agent.evaluate(G_test)  // greedy decode
        IF eval_ds < best_eval_ds:
            Save checkpoint
END FOR
```

---

## 推理流程伪代码

```
Algorithm: V9 Inference (Greedy Decode)
───────────────────────────────────────
Input: Graph G, trained policy θ, pop_size B
Output: Dominating set D_elite

env = TensorMemeticEnv(G, batch_size=B)
state = env.reset()              // B greedy+noise 初始解

WHILE NOT env.is_done():
    prob_d, prob_r, _ = policy_θ(G, state.cover)

    // ── 贪心选择 ──
    a_d = argmax(prob_d)         // 最优 destroy 算子
    a_r = argmax(prob_r)         // 最优 repair 算子

    state, _, done, info = env.step(a_d, a_r)
END WHILE

D_elite = env.D_elite            // 种群中最优解

// ── 可选: Sample-k-Best ──
IF sk_k > 0:
    best_D = D_elite
    FOR seed = 0 TO sk_k-1:
        env.reset(seed)
        WHILE NOT done:
            a_d ~ Categorical(prob_d)   // 采样
            a_r ~ Categorical(prob_r)
            state, _, _, _ = env.step(a_d, a_r)
        IF |env.D_elite| < |best_D|:
            best_D = env.D_elite
    D_elite = best_D

// ── 可选: Hybrid 后处理 ──
D_final = hybrid_optimize_dominating_set(D_elite, G.edge_index, G.num_nodes)

RETURN D_final
```

---

## 后处理流程

[src/common/postprocess.py](src/common/postprocess.py) 实现三阶段混合后处理：

**MDS 后处理** (`hybrid_optimize_dominating_set`)：

```
Stage 1: Leaf Subsumption
  FOR each leaf node u in D (degree=1):
    IF neighbor v ∉ D: replace u with v

Stage 2: PN-Directed Hub Promotion
  WHILE exists u ∈ D with PN_size(u) == 1:
    w = PN_target(u)                    // w 仅被 u 覆盖
    Find z ∈ N[w]\D with max degree     // 寻找 w 邻域中的 hub
    IF deg(z) > deg(u):
      D ← D \ {u} ∪ {z}
      IF NOT full_coverage(D): rollback

Stage 3: Strict Pruning
  WHILE exists u ∈ D with PN_size(u) == 0:
    D ← D \ {u}
    IF NOT full_coverage(D): D ← D ∪ {u}, break
```

**MVC 后处理** (`mvc_hybrid_optimize`)：结构与 MDS 相同，但将 PN 替换为 PE (Private Edge)。

---

## 基线算法

| 基线 | 文件 | 说明 |
|------|------|------|
| Greedy | [baselines/greedy.py](src/common/baselines/greedy.py) | 贪心 MDS（每步选覆盖增益最大节点） |
| GA | [baselines/genetic_algorithm.py](src/common/baselines/genetic_algorithm.py) | 遗传算法（锦标赛选择 + 均匀交叉 + 变异） |
| ALNS | [baselines/alns.py](src/common/baselines/alns.py) | 自适应大邻域搜索（自适应算子权重） |
| RLD | [baselines/rl_domination.py](src/common/baselines/rl_domination.py) | RL-Domination 基线（Chen et al. 2024） |
| Greedy_MVC | [baselines/greedy_mvc.py](src/common/baselines/greedy_mvc.py) | 贪心 MVC |
| GA_MVC | [baselines/genetic_algorithm_mvc.py](src/common/baselines/genetic_algorithm_mvc.py) | 遗传算法 MVC |
| MVCApprox | [baselines/mvc_approx.py](src/common/baselines/mvc_approx.py) | MVC 2-近似算法（MVCApprox + MVCApprox-Greedy） |
| S2V-DQN | [baselines/s2v_dqn_mvc.py](src/common/baselines/s2v_dqn_mvc.py) | Structure2Vec DQN (Dai et al. 2017) PyTorch 复现 |

---

## 实验结果

> **训练配置**：epoch=25, pop_size=8, batch_size=8, lr=3e-4, device=cuda（N=40 MDS 模型使用 epoch=150 加长训练）
> 
> **测试集**：每尺度 80 张图，V9 使用 Greedy argmax 解码 + Hybrid 后处理
> 
> **推理时间定义**：单张图从输入到输出完整解的端到端墙钟时间（秒/图），含模型前向/算法迭代/后处理。各算法配置：
> - **V9**：pop_size=8, Greedy decode + Hybrid 后处理（含 GNN 前向 × N_steps + 3 阶段后处理）
> - **Greedy**：贪心构造（每步选覆盖增益最大节点）
> - **GA**：pop_size=50, generations=200, mutation_rate=0.1（含 50×200 次适应度评估）
> - **ALNS**：max_iterations=1000（含自适应算子选择 + 1000 次 destroy-repair 循环）
> - **RLD**：RL-Domination 独立进程推理（含模型加载 + 500 episodes）

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
pip install python-louvain   # Louvain 社区检测（可选，缺失时 community_bombing 退化为恒等）
```

### 训练

```bash
# V9 MDS 训练（ER + BA, N=20,40,80）
python scripts/train/train_v9.py --n-sizes 20,40,80 --graph-types ER,BA --max-epochs 50 --pop-size 8 --batch-size 8 --device cuda

# V9MVC 训练（ER, N=20,40,80）
python scripts/train/train_v9_mvc.py --n-sizes 20,40,80 --max-epochs 50 --pop-size 8 --batch-size 8 --device cuda

# V9 无 LLM 消融训练
python scripts/train/train_v9_ablation_no_llm.py --n-sizes 20,40,80 --max-epochs 25 --pop-size 8 --batch-size 8 --device cuda
```

### 评估

```bash
# V9 MDS 对比评估
python scripts/eval/eval_v9_comparison.py --n-sizes 20,40,80 --graph-types ER --device cuda

# V9MVC 对比评估
python scripts/eval/eval_v9_mvc_comparison.py --n-sizes 20,40,80 --device cuda
```

### 推理

```bash
# MDS 推理（Greedy 解码 + 后处理）
python inference.py --input data/ER/N20/test --checkpoint checkpoints/v9/ER/N20/best_model.pth --task mds --postprocess --device cuda

# MVC 推理（Sample-10-Best + 后处理）
python inference.py --input data/ER/N20/test --checkpoint checkpoints/v9_mvc/ER/N20/best_model.pth --task mvc --sk-k 10 --postprocess --device cuda
```

### 统一调度

```bash
# 通过 runner.py 训练
python src/runner.py train --arch v9 --n-sizes 20,40,80 --graph-types ER,BA --epochs 50 --device cuda

# 通过 runner.py 评估
python src/runner.py eval --arch v9 --n-sizes 20,40,80 --graph-types ER --device cuda
```
