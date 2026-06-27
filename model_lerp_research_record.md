# model_lerp 研究档案

**类脑非 Transformer 线性注意力语言模型 — 架构、数学、实验、验证全记录**

> 谱系:PhotonLM → CUV → CortexS/CortexSLM → **model_lerp**(本档案对象)
> 目标:以"人类智能终极架构"为远期目标的去-KV 线性注意力 LM。
> 硬件:RTX A5000 24GB / Windows / conda `myenv` / torch 2.6.0+cu124 / 项目目录 `C:\Users\Administrator\Desktop\Train\lerp`

---

## 0. 摘要

model_lerp 用**单一通信原语**(`I`=Query, `T`=Key, `C`=Value,对线性矩阵账本做因果读取)贯穿一个四级类脑层级(cell → cortex → lobe → whole brain)。每个 token 通过三组 lerp-注意力对"大脑"读写;大脑通过跨皮层聚合"思考";输出经 RMSNorm+FFN 进入复数振幅 FeynmanHead。

**核心工程性质:`lin_read` 是分块(chunked)线性注意力,跨块只携带 `(dk,dv)` 状态、不含序列维 `L`,因此状态是 `O(B·H·dk·dv)` 而非 `O(B·H·L·dk·dv)` —— 永不爆炸。** 这是本会话纠正的关键误判(见 §2、§11)。

本架构 = **四级骨架 + 八项升级**(八项见 §5;#7 已把初始六投影精简为**单读四投影**,§10.13)。

---

## 1. 设计哲学与谱系

| 原则 | 落地 |
|---|---|
| 去-KV(de-KV) | 不存每 token 的 K/V,只累加线性账本 `Σ φ(T)⊗C`,读取 = 一次矩阵-向量积 |
| 完全并行 + 因果 | 线性注意力的结合律 → 分块并行;块内因果 mask,块间状态传递 |
| 层级 = 注意力头的分组 | cortex=一个线性注意力头;lobe=一组 cortex;whole=全脑;cell=头内一个 d 通道 |
| 容量优先 | 先把 `d_cell` 填到任务够用,再加 cortex 数(见容量律 §10.1) |
| 梯度健康是一等架构指标 | 每个组件都要过梯度有限/无悬崖/无消失爆炸检验 |

谱系要点(历史):PhotonLM(3 通道门控线性注意力 + PathIntegralHead)→ CUV(ITC 注意力 + 复数路径积分传播子 + 可学习 value bank)→ CortexS/CortexSLM(生产级训练/评估设施)→ model_lerp(本档案,回到四级骨架并叠加全部升级)。

---

## 2. 核心原语:`lin_read`(分块因果线性注意力,去-KV)

### 2.1 数学定义

对 token `t`,以 `φ = elu + 1` 为特征映射:

```
out_t = φ(I_t) · ( Σ_{s≤t} φ(T_s) ⊗ C_s )  /  ( φ(I_t) · Σ_{s≤t} φ(T_s) )
```

- 分子:查询 `φ(I_t)` 读取因果累加的外积账本 `Σ φ(T_s)⊗C_s ∈ ℝ^{dk×dv}`
- 分母:归一化项 `φ(I_t)·Σ φ(T_s) ∈ ℝ`(clamp 下限 `1e-6` 防除零)
- `I,T ∈ ℝ^{B×H×L×dk}`,`C ∈ ℝ^{B×H×L×dv}`,输出 `∈ ℝ^{B×H×L×dv}`

### 2.2 分块实现(工程最优,数学等价)

把 `L` 切成 `nc` 个长 `chunk` 的块,携带跨块状态 `S∈ℝ^{dk×dv}`、`z∈ℝ^{dk}`:

```
for each chunk c:
    q,k,v = φ(I_c), φ(T_c), C_c                       # (B,H,chunk,*)
    att   = (q @ kᵀ) ⊙ tril_mask                      # 块内因果分数
    num   = q @ S + att @ v                            # 块间(状态) + 块内
    den   = (q ⊙ z).sum(-1) + att.sum(-1)  (clamp 1e-6)
    out_c = num / den
    S    += kᵀ @ v ;  z += k.sum(0)                    # 状态传到下一块
```

参考实现 `_lin_read_ref`(cumsum,显存重)仅用于验证等价性。

### 2.3 为什么永不爆炸(本会话核心纠正)

| | 携带量 | 量级 |
|---|---|---|
| 分块 `lin_read` | 状态 `S=(dk,dv)`、`z=(dk)`,**无 L 维** | `O(B·H·dk·dv)` |
| naive cumsum 账本 | 每 token 一个外积,`(L,dk,dv)` | `O(B·H·L·dk·dv)` |

**de-KV 把序列维从携带状态中彻底移除。** 实测(本会话 T4,`B=8, L=512, H=256, dk=128, dc=64`):

```
分块状态 0.074 MB   vs   naive 账本 37.7 MB   →   512× 节省
```

> 注:此前"四级/256-cortex 会 OOM 60GB"的说法是**张冠李戴** —— 那 60GB 属于另一个旧模型(360-area Cortex `model.py`,它按 token 存每区状态 `B·T·360·d`)。model_lerp 不存在此问题。

---

## 3. 四级层级结构

```
cell    : 一个头内的一个 d_cell 维通道(线性注意力的最小单元)
cortex  : 一个线性注意力头(I/T/C 账本)。数 = H = n_lobe × n_cortex
lobe    : 一组 cortex(n_cortex 个),共 n_lobe 个
whole   : 全脑 = 所有 cortex 的可学习聚合(HierAgg,见 §7)
```

聚合方向:`cell → cortex → lobe → whole` 逐级 reduce 出"全脑摘要",再广播回各 cortex(残差)。用户设定的满规模:`d_model=512, d_cell=64, n_lobe=4, n_cortex=64 → H=256`。

---

## 4. 单层架构(完整前向数学)

记 `dc = d_cell`,`dk = phi_expand·dc`(Q/K 维翻倍),`H = n_lobe·n_cortex`。

```
h = RMSNorm(X)                                                  # 层输入归一化

# 四投影（去 lerp 门 + hop2 后:to_Tx/to_Cx 删,它们只喂已移除的门）
Ix = heads(W_Ix h, dk)                                                     # token 查询 _x
Ts = heads(W_Ts h, dk)   Cs = heads(W_Cs h, dc)                            # 全脑 _s 账本(键/值)
Is = heads(W_Is h, dc)                                                     # 自思考投影(conv 模式 dc)

# [#2] Q/K 短因果卷积(局部选择性)
Ix = DWConv_causal_q(Ix) ;  Ts = DWConv_causal_k(Ts)

# 唯一一次读:token 读大脑(原 hop1)
Xl = lin_read(Ix, Ts, Cs)                                       # (B,H,L,dc)

# [#8] 自思考:卷积(默认)或线性自读(消融)
Ta = DWConv_causal(Is)              # think_mode='conv'(局部重组)
   | lin_read(Is, Ts, Cs)           # think_mode='lin'(全局,消融对照)

# [#4] 可学习跨皮层聚合(self-think 路;agg_r 随 hop2 删）
Ta = HierAgg_t(Ta)

# 组装
feat = concat[ Xl ; Ta ]                                        # (B,L, 2·H·dc)
O    = X + W_o · feat                                           # 残差
O    = O + FFN(RMSNorm(O))                                      # FFN 残差
```

**两路语义:**
- `Xl`:token 的查询读全脑账本 = "token 读大脑记忆"(**唯一一次扫描**)
- `Ta`(self-think):大脑对自身投影做局部时序卷积 = "自思考/局部重组"

**[本轮移除] lerp 门 + hop2(原 2-hop 迭代检索):** 门精炼 Q/K/V 后**唯一消费者就是 hop2**;hop2 读同一共享账本、信息已被 `Xl` 覆盖(结构性冗余,`−hop2` 消融中性/更低 §10[2]/§10.13)。删二者 → **每层扫描 2→1,输入投影 6→4**(`to_Tx/to_Cx` 死)。门的 `−query_refine +0.032` 收益**与 hop2 绑死**("精炼"本质需两次读:读→精炼→再读),脱离 hop2 无独立证据 → 一并删。

---

## 5. 八项升级(数学 + 动机 + 支撑实验)

| # | 升级 | 数学/实现 | 动机与实验结论 |
|---|---|---|---|
| 1 | **φ-expand** | `dk = phi_expand·dc`(默认 2×);Q/K 翻倍,value 留 `dc` | 线性注意力选择性受限 → 可分离的键提升 recall。φ-sweep 检验键密集(`m ~ key-dim`)时的增益 |
| 2 | **qk_conv** | 对 `Ix,Ts` 做 kernel=4 因果深度卷积(`DWConv`) | 局部选择性。历史实测在 recall 变体上 **+0.142**,故默认开启 |
| 3 | **FeynmanHead** | 复数 Born + P×P 路径积分 + `head_eps` 防悬崖 + `head_cap` + **线性 base**(见 §6/§6.1) | `head_cap=5` 修复:copy L=64 loss **0.30→0.052**。**新发现**:纯 Born 在 neo 有 CE 硬地板 ~3.70(§6.1),加线性 base 后 neo CE **→0**(双验) |
| 4 | **可学习 HierAgg** | TT-RNN 式跨皮层聚合核(见 §7),init 严格等于 mean-pool | tt ≫ 固定 mean-pool;结构是二阶因素,容量主导 recall |
| 5 | **lin_read chunked** | 携带 `(dk,dv)` 状态,`O(B·H·dk·dv)` | 永不爆炸(§2.3),L=512 省 512× 显存 |
| 6 | **删除 forget-gate** | 不加回 | 近因偏置**摧毁**联想 recall(实测 0.96 → 0.06);GLA/Mamba2 式遗忘门在本任务有害 |
| 7 | **单读(删 lerp 门 + hop2)** | 历经"去翻译器 → lerp 2-hop"两版后,本轮**直接删整条第二读**:门 + hop2 + `to_Tx/to_Cx` + `agg_r` 全删,每层只剩 `Xl=lin_read(Ix;Ts,Cs)` 一次扫描,`feat=[Xl;Ta]` | 第二读结构性冗余(读同一共享账本,信息已被 Xl 覆盖):翻译器版有害(only-R_a translator-ON `0.743±0.319` vs OFF `0.862±0.106`,Δ−0.118);lerp 版 `−hop2` 消融中性/更低(§10[2]);门 `+0.032` 收益与 hop2 绑死,脱离 hop2 无独立证据。详见 §10.13 |
| 8 | **T_a → conv** | `Ta = DWConv_causal(Is)` 替代全局 `lin_read(Is;Ts,Cs)` | 自思考真实存在但在已测任务上冗余(见 §10.4) |

---

## 6. FeynmanHead(数学)+ 大词表 loss 地板(本会话新发现 + 修复)

```
a   = W_re·h + i·W_im·h                  ∈ ℂ^P           # h → P 个复数路径振幅
a'  = U · a ,  U = U_re + i·U_im         ∈ ℂ^{P×P}       # 路径积分传播子
      pr = ar@U_reᵀ − ai@U_imᵀ ;  pi = ar@U_imᵀ + ai@U_reᵀ
v   = W_out · a' , W_out = out_re + i·out_im ∈ ℂ^{V×P}   # 路径 → 词表振幅(复数)
      vr = out_re(pr) − out_im(pi) ;  vi = out_re(pi) + out_im(pr)
born   = head_cap · tanh( log(|v|² + head_eps) / head_cap ) ,   |v|² = vr² + vi²
logits = base(h) + born            # base = Linear(d,V)(可 tie 到词嵌入);head_linear_base=False 时退化为纯 born
```

- **Born 规则**:概率 ∝ `|amp|²`,提供**有界**规则抑制。
- **`head_eps`(默认 0.1)**:对数内地板。梯度放大子 `∂born/∂v ∝ 2v/(|v|²+ε)`,**上界 `1/√ε`**。`ε=1e-6 → 1000×` 悬崖;`ε=0.1 → ≈3.2×` 有界(T6/s4 实测一致)。
- **`head_cap`(默认 5.0)**:tanh 把 born 限制在 `±cap`。`cap=5` 稳定;`cap=30` 在小词表饱和 softmax → 发散。
- **`n_paths`(默认 64)**:路径数 P;`U` 是 P×P 复数矩阵,init ≈ 恒等 + 小噪声。
- 传播子 `U` 与 `out_re/out_im` 训练时**不衰减**(否则抹掉 Born 抑制,见 §12)。

### 6.1 致命发现:纯 Born 头的大词表 CE 地板(本会话数学证明 + 数值双验)

纯 born(无 base)的 logit 被**双向钉死**:`logit ∈ [cap·tanh(log(eps)/cap), cap]`。`cap=5,eps=0.1` 时区间 `[−2.153, 5.0]` → 赢家对输家的**最大可能 logit 差 = cap − floor = 7.153**(与权重无关,是 tanh 饱和 + eps 地板的硬上界)。确定性 next-token 的最优 CE 因此有**硬下界**:

```
CE_floor = log( 1 + (V−1)·exp(−(cap − floor)) )
```

| 词表 | 绝对下界(赢家=cap=5) | 现实(赢家≈3.63,\|v\|²~100) |
|---|---|---|
| byte V=256 | **0.182** | 0.579 |
| **neo V=50257** | **3.697** | **5.046** |

**→ 纯 Born 头 + neo 时,训练 loss 在数学上不可能低于 ~3.7 nats**(与容量/recall/架构无关)。`head_eps` 的两难被词表放大:eps 小→可锐利但梯度悬崖;eps 大→梯度安全但 loss 封顶。

**修复(已落地,默认开):线性 base** `logits = base(h) + born`。base 提供**无界锐度**消地板,born 保留有界抑制。`base` 在 `cfg.tie` 下与词嵌入共享权重(neo 零额外参数)。

**数值双验(本会话,CPU):**

| | byte V=256 | neo V=50257 |
|---|---|---|
| 解析地板(乐观/现实) | 0.182 / 0.579 | 3.697 / 5.046 |
| 经验最优化(直接压 CE) | 0.29 | **4.02(纯 born 卡死)** |
| **加 base 后** | — | **CE → 0.000(地板破除)** |
| 对照:无界线性头 | 0.0001 | 0.0002 |

`head_eps=1e-3` 的备选(若坚持纯 Born):logit 地板 `5·tanh(log(1e-3)/5)=−4.40`(注:`log(eps)=−6.9` 是 tanh 前的值,非 logit 地板),neo 地板降到 `~1.63` → 可收敛。

**[本轮重要纠正] 地板只对"纯 Born"成立,不是生产配置的硬墙:** 默认 `head_linear_base=True` 时**无硬地板**——历史 TinyStories 运行用同款模型 + 头到达 val **1.35 < 3.70**,直接反证线性 base 已破除地板(硬地板下 1.35 不可能)。曾观察到的 FineWeb val 卡 ~4.0 **不是头地板,而是数据难度**:TinyStories 是低熵玩具集 → 1.35;FineWeb 广域网页 → 4.0 对 ~0.33B token / 253M 是正常水平(GPT-2 124M 需 ~300B token 才到 ~3.0)。BPE-loss 与 byte-loss 不可比(BPE 一 token ≈ 3.5 字符)。**结论**:本轮把 `head_born` 默认翻为关(纯线性头无地板风险、更省、真实 LM 上 Born 零收益 §10.5);Born 头仅作可选消融保留。降 FineWeb loss 的正解是**更多 token + 同步延长 lr 余弦周期**,与头无关。

### 6.2 Delta Rule:记忆力(recall)升级(本会话实现 + CPU 验证)

recall(0.358)是质量的根本瓶颈。加性账本 `S += φ(k)ᵀv` 让每个 token 都写入 → 键碰撞累加干扰。**修法 = DeltaNet 先擦后写**(`use_delta` toggle,默认关):

```
v_old = φ(k) @ S                       # 当前键 k 处存的值
S    += β · φ(k)ᵀ (v − v_old)          # 写新值前先擦掉旧值 -> 无累加干扰
out_t = φ(q) @ S_t / (φ(q) @ z_t)      # 写后读(因果)
```

- **稳定性硬约束(踩坑记录)**:delta 分支内 **q/k 必须 L2 单位归一化**。否则擦除算子 `(I − β·φ(k)ᵀφ(k))` 的特征值 `1 − β‖φ(k)‖²` 在 ‖φ(k)‖² 大时 ≪ −1 → **S 爆炸到 e27,loss=NaN**(已实测)。归一化后特征值 `1 − β ∈ [0,1)` 收缩稳定;且单位键下 query=key 精确检索存值。
- **β**:每头可学写入门 `β=sigmoid(delta_beta)`,init 2.0(≈0.88)。`delta_beta` 仅在 `use_delta=True` 时创建(关时无死参)。
- **CPU 验证**:稳定(max|o|=2.09 不爆)、因果、**overwrite 语义**(同键二写:delta 读出纯 latest `[0,0.5,0]` 零旧值干扰 vs additive 混淆 `[0.5,0.5,0]`)、**gradcheck PASS**、use_delta 模型可训(loss 0.051 记住固定 batch)、默认关时自检 T0–T8 全过(无回归)。
- **速度(本次升级)**:已从逐 token 递归 O(L) 串行 **升级为分块并行 DeltaNet(WY 形式,Yang et al. 2024)**:块内解单位下三角系统 `(I+tril(diag(β)KKᵀ,−1))U=diag(β)(V−KS₀)`(`torch.linalg.solve_triangular`),O(L/chunk) 串行、块内全并行。**CPU 验证 == 串行 ref 到机器精度(4.4e-16)+ gradcheck 双 β 形状全过**;dtype 自适应(仅 bf16/fp16 升 fp32 保稳,fp32/fp64 原样)。`use_delta=True` 端到端可训(loss 4.29→2.71)。m=64(L=131,chunk=32)从 131 步串行降到 5 块 → ~26× 更少串行步。
- **GPU recall 验证:REFUTED(本任务)**。A5000 5-seed:`m=16 平手(0.898/0.899)→ m=48 delta 0.030 vs additive 0.656 → m=64 delta 0.013 vs additive 0.598`。delta 在密键区**崩到 ~chance**,additive 守住 0.6。成因二选一(实用结论相同):①**机制**——recall 任务是不同键(无覆写),DeltaNet 的擦除为覆写而设;密集非正交键下 `v_old=k_tᵀS` 混入重叠键的值,擦除污染、连带擦坏别的键 → 雪崩。②**数值**——密键下 WY 三角系统病态,fp32 解 U 发散(分块显式求逆可能比逐 token 迭代更不稳;等价验证用的是近正交随机键,没探到病态区)。分辨:串行 `_lin_read_delta_ref` 在 m=64 若也崩=机制,若稳=分块 conditioning。**结论:`use_delta` 默认关;distinct-key recall 的杠杆是键可分性(phi-expand),非先擦后写;delta 的用武之地是覆写/状态追踪(本任务未测)。**

---

## 7. HierAgg(TT-RNN 跨皮层聚合,数学 + init=mean 证明)

逐级(`ℓ = 1..nest_levels`,分支 `b = nest_branch`)聚合:

```
ctx ← z                                                 # 从各 cortex 输出开始
for ℓ in 1..levels:
    把 ctx 的头维分成 (H/b, b) 组兄弟
    w_ℓ = softmax(mix_ℓ)              # mix 初始化为 0 → w 均匀 = 1/b
    ctx = W_tf,ℓ ( Σ_j  w_ℓ[j] · sibling_j )            # 加权聚合 + 线性变换
                                                        # W_tf 初始化 = I(恒等)
广播:把顶层全脑摘要 ctx 复制回所有 H 个 cortex
out = z + broadcast(ctx)                                # 残差
```

**init=mean 证明**:`mix=0 → softmax → 均匀权 1/b`,`W_tf = I` → 每级聚合恰为兄弟均值,逐级即组均值 → `out = z + mean_pool(z)`,与固定 mean-pool **逐元素相等**。
本会话 T5 实测:`max|tt − mean| = 4.8e-7`(初始确等于均值池),之后可学习偏离。

**约束**:`H` 必须被 `nest_branch^nest_levels`(默认 `2²=4`)整除。`H=256 → 256/4=64 ✓`。

---

## 8. 配置 `LerpConfig`

| 字段 | 默认 | 含义 |
|---|---|---|
| `vocab_size` | 256 | 词表(byte);neo BPE 时 50257 |
| `d_model` | 512 | 残差宽度 |
| `d_cell` | 64 | `dc`:每 cortex 的 value 维 |
| `n_lobe` | 4 | lobe 数 |
| `n_cortex` | 64 | cortex/lobe → `H=256`(满规模;小实验调小) |
| `n_layers` | 4 | 层数 |
| `chunk_len` | 256 | 分块长度(纯工程旋钮,数学不变 §9.1[T0];256 比 128 更少串行块/kernel launch,大词表/长上下文下提速) |
| `phi_expand` | 2 | `dk = 2·dc` |
| `qk_conv` / `qk_kernel` | True / 4 | Q/K 短卷积 |
| `think_mode` / `think_kernel` | conv / 4 | 自思考实现 |
| `nest_mode` / `nest_branch` / `nest_levels` | tt / 2 / 2 | 聚合(`H` 须整除 `branch^levels`) |
| `head_eps` / `head_cap` / `n_paths` | 0.1 / 5.0 / 64 | FeynmanHead |
| `head_linear_base` | True | 头加无界线性 base,消除大词表 CE 地板(见 §6.1);`tie` 下与词嵌入共享权重。False=纯 Born 消融 |
| `head_born` | **False** | **本轮默认翻为关**:纯线性头(=标准 tied LM 头)。Born 抑制项仅玩具任务有 ~16% forbidden-mass 微利、真实 LM 零收益(§10.5),且省 born 参数 + 2 个 (B,L,V) GEMM + 显存。`--head_born` 重开 |
| `use_delta` | False | [记忆力] lin_read 用 DeltaNet 先擦后写 `S+=β·φ(k)ᵀ(v−φ(k)S)`,消加性干扰(SOTA 联想记忆)。**分块并行实现**(WY 形式,§6.2),需 MQAR/recall 验证后再默认开;delta 分支内 q/k 必须单位归一化(否则爆炸,见 §6.2) |

**满规模参数代价警告**:`H=256, dk=128, d_model=512` 时投影主导,约 **94M/层**;小实验把 `n_cortex` 调小。

---

## 9. 本会话验证(CPU,沙盒)

> 全部在无 GPU 的沙盒上跑(tiny 配置),验证**正确性/因果/不爆炸/可学习**,非规模化性能。

### 9.1 `model_lerp.py` 自检电池(tiny:vocab 65, d=48, dc=12, H=8, dk=24, 3 层)

| 测试 | 结果 | 判定 |
|---|---|---|
| T0 分块 == cumsum(dk≠dv) | `max|diff| = 2.38e-7` | 数学等价 ✓ |
| T1 初始 loss | `4.1732 ≈ ln65 = 4.174` | ✓ |
| T2 梯度有限 | True | ✓ |
| **T3 因果**(扰动未来 token) | 过去 `0.00e0`,之后 `4.84e-2` | **严格因果** ✓ |
| **T4 不爆炸** | L=512 状态 `0.074MB` vs naive `37.7MB`(**512×**) | ✓ |
| T5 HierAgg(tt) init==mean | `4.77e-7` | ✓ |
| T6 FeynmanHead 防悬崖 | `ε=1e-6→grad 0.4`(界 1000),`ε=0.1→0.2` | 有界 ✓ |
| T7 φ-expand 维度 | `to_Ix=192=H·dk`,`to_Cx=96=H·dc`,`Wil dc→dk` | ✓ |
| T8 能学(byte LM) | `5.54 → 3.11` | ✓ |

### 9.2 `test.py` 综合诊断台:**72 测试,PASS 66 / WARN 5 / FAIL 0 / SKIP 1**(升级后;升级前为 65/6)

- **[2] 组件消融**(100 步 tiny,越低越好):full `3.262` | −pool `3.318(+0.056)` | **−Ta `3.380(+0.118,最影响)`** | −hop2 `3.158(−0.104)` | −query_refine `3.294(+0.032)` | −FFN `3.225(−0.037)`
  > −hop2/−FFN 在小尺度短训反而更低 = **早期收敛假象**(参数少在字节 LM 上收敛更快),非组件无用;真实结论看任务驱动的 `ablation.py`。
- **[3] chunk_len 不变性(fp32)**:`max_diff = 4.8e-7`(TF32 下 ~1e-3 是舍入,非 bug)。
- **[4] 头悬崖**:`ε=1e-6` 放大 1000×,`ε=0.1` 有界 3.2×;feynman final `3.049`、maxgrad `4.1`(稳定,0 spike)。
- **[7b] 深度梯度健康**:first/last 比 `1.85(4层)→2.44→4.73→5.18→8.31→10.93(64层)`(随深度失衡增长)。
- **[7c] 多种子方差**(3 seed×60 步 tiny):std `0.054`。
- **[7d] 表征健康**:eff_rank `23/32`、mean cos `0.170`(未塌缩)。
- **[8] 仪表盘**:final `3.095`,act RMS `6.09→13.53`,578K 参数。

### 9.3 `train.py` / `ablation.py` 烟测

- `train.py`:全管线(arrow→memmap→建模 H=8/dk=24/think=conv/nest=tt→训练→eval→ckpt→曲线)通,loss `5.35 → 4.22`(tiny,byte,假 cache)。
- `ablation.py`:四 probe(phi-sweep / hard-tasks / conv-vs-lin / contribution)+ 出图全通。contribution(12 步 tiny,multihop):L0 `||Xl1||=0.256, ||Xl2||=0.127, ||Ta||=0.295`;L1 `0.460 / 0.242 / 0.342`。

### 9.4 loss 地板 + 头修复 + 提速升级(本会话验证)

- **loss 地板双验**:解析(neo 3.697/5.046、byte 0.182/0.579)+ 经验(纯 born neo 卡 4.0、加 base →0.000、无界线性头 →0.0002)。详见 §6.1。
- **头线性 base 落地**:`tie base==embed = True`(neo 零额外参数);自检 T0–T8 全过;test.py 升级后 **66 PASS / 0 FAIL**,dashboard loss `3.095 → 2.235`(base 已助降)。
- **chunk_len 256(默认)/ lin_read 预分配**:数学不变(§9.1[T0]),自检通过。
- **`lin_read_triton.py`**:torch 分块前向+手写反向 `gradcheck` **PASS**、`lin_read_fast` vs `_lin_read_ref` 等价 `<1e-4`、因果通过(均 CPU 验证);Triton 前向核需 A5000 跑校验器确认。

---

## 10. 实证发现(历史 GPU 运行,带核验数字)

> 以下数字来自此前会话的 GPU 实验(已对 transcript 核验);本会话未重跑这些规模化实验。

### 10.1 容量律(recall 的支配因素)

```
recall_acc = logistic( α·log2(H·dc²) + β ) ,  R² = 0.99
loss       ∝ (H·dc²)^(−0.55)
```

**状态容量 `H·dc²` 支配联想 recall。** 结构(tt 嵌套)是二阶因素。已对照 Stanford Zoology/Based 与 Llama 验证。

### 10.2 最重要的架构发现:**recall 是短板**

```
copy(位置)    = 0.931     ✓ 强
majority(聚合)= 0.966     ✓ 强
recall(键值)  = 0.358 ± 0.160   ✗ 弱(方差大)
```

这是**线性注意力选择性的本质局限**(非 bug):`φ=elu+1` 无法像 softmax 那样在多键中锐利挑出一个。这正是 vanilla 线性 Transformer 在 associative-recall 输给 softmax、也是 Mamba/GLA/RWKV 引入选择性的动机。**→ recall 选择性是下一个该攻的点。**

### 10.3 R_a 翻译器假说**被证伪**(支撑升级 #7)

```
only-R_a copy:  translator-ON = 0.743 ± 0.319    OFF = 0.862 ± 0.106    Δ = −0.118
```

翻译器不但没帮助还略差(且方差爆炸)。R_a 冗余是**结构性的**(读同一共享账本,信息已被 `Xl` 覆盖)。→ 当时改为 lerp 门精炼的 2-hop 检索;**本轮进一步证实 2-hop 整体冗余,已整条删除回归单读(见 §5#7 / §10.13)**。

### 10.4 T_a 自思考:**真实但冗余**(支撑升级 #8)

```
||Ta|| ≈ ||Xl||   (比值 1.0~1.14,各层)        → T_a 输出非零,在"算东西"
only-Ta majority = 0.904  >  only-Xl = 0.876   → T_a 单独能做聚合,甚至略强于读
消融 Ta:reverse-copy 0.827 → 0.977(反而变好) → 整模型里删它不疼;对"逆序检索"是干扰
```

判决:自思考真实发生,但读取通路 + FFN 已覆盖同样功能 → 多通路重复。

### 10.5 抑制头:feynman 最优但优势小

```
forbidden_mass:  feynman 0.0184 < amp 0.0189 < pathU 0.0203 < linear 0.0218
```

`cap=5` 的 feynman 最低(比 linear 低 ~16%),全部 100% argmax-in-allowed。Born 抑制真实,但玩具任务太简单(linear 自己也能学约束),优势小。

### 10.6 `head_cap=5` 修复被验证(支撑升级 #3)

| | 旧头 | cap=5 |
|---|---|---|
| copy L=64 loss | 0.30 | **0.052**(6× 更低) |
| copy L=64 exact | 0% | **29.7%** |
| copy L=64 per-token | — | **98.1%** |

`0.981^64 = 29.3% ≈ 实测 29.7%` → "L=64=0%"只是 exact-match 的指数压缩假象,模型**逐 token 几乎全对**。

### 10.7 forget-gate 摧毁 recall(支撑升级 #6)

加 GLA/Mamba2 式遗忘门 → 近因偏置 → 联想 recall **0.96 → 0.06**。故**删除,不加回**。

### 10.8 深度研究

L=1 不足;**L=2 必需**(带干扰物的 k-hop 链)。L=2 与 L=4 在噪声内。早期"深度有害"结论**已撤回**(被污染的 CSV 假象)。n_layers=12 对推理可能过量但未坐实。

### 10.9 嵌套(tt vs mean)

可学习 HierAgg(tt)≫ 固定 mean-pool(玩具 recall +约 36%);init 等于 mean(§7 已证)。

### 10.10 其他工程发现

- **chunk_len 不变性**:fp32 `7.2e-7`(数学不变);TF32 全模型 `1.4e-4` = 各算子按形状舍入的精度伪影,**非 bug**(测试时关 TF32 即恢复)。
- **深度梯度失衡**:first/last 比随深度增长(GPU 期 ~17× @64 层;本会话 tiny ~10.9× @64 层)→ 极深网需 depth-scaled init 或 LayerScale。
- **多种子方差**:小尺度短训 std `0.207`(init 敏感);规模化后 val 单调下降,不严重。
- **表征健康**:eff_rank ≈ `19/32`(GPU 期分析,约 56% 容量;本会话 tiny CPU 测 23/32),未塌缩。

### 10.11 v2 功能消融实测(用户 A5000;seeds=3 steps=2000;tiny d=64/dc=16/H=8/L=3,chance=0.025)

- **phi-sweep**:delta(phi2−phi1) 随键密度 **单调增** `+0.007(m=8) → +0.021(m=16) → +0.113(m=32)`(tiny 3-seed)。**结论性复跑(A5000,5-seed,d=128/dc=32/H=16)**:`m=16 +0.008(t=0.5) → m=48 +0.026(t=0.6) → m=64 **+0.336(t=3.84,p<0.01)**`,phi2=0.598 vs phi1=**0.262**(2.3×)且方差小得多(0.033 vs 0.193)。→ **phi-expand 确证有用**(已下结论):键超过 key-dim 时 phi1 碰撞坍缩,phi2 把碰撞边界推远。注:首版判据用 `2·max(σ)` 误判为 tie —— 正确量是差值标准误 `√(σ₁²/n+σ₂²/n)`(变体崩溃时的高方差是信号,非门槛),已修。**这是部分 recall 修法(抬高键分离上限),但线性注意力的根本干扰仍需 Delta Rule(§13)。**
- **phi 容量扫描(sep,ADDITIVE,A5000 3-seed,早停)** —— phi2→phi3→phi4 能否把召回推向 1.0?**否,饱和**:`m=48 phi2 0.688±0.022 / phi3 0.703±0.006 / phi4 0.703±0.025`(平,phi2→phi3 t≈1.15 不显著);`m=64 phi2 0.423±0.293 / phi3 0.022±0.005 / phi4 0.438±0.290`(**纯噪声**:±0.29 双稳 + **非单调**(phi3 反最低=噪声铁证)+ phi3 是 3 种子碰巧全落崩溃盆地;3 种子估不动双稳过程)。**结论**:phi 一旦 dk 舒适超过 m(phi2 够 m≤48)即**饱和**,再加只增算力、~0 召回收益。m=48 的 ~0.70 天花板非 dk 限——最可能 additive 特征图串扰(elu+1 全正 → 键无法正交;**假设,非实测证明**)。**真召回杠杆 = H·dc²(值容量)+ 深度 L,非 phi**(容量律 §10.5)。生产 phi2(dk=128, H·dc²=65536, L=12)远强于本探针(dc=32, L=3, H·dc²=16384),探针上限不限生产。**phi 作用边界厘清:清碰撞悬崖(phi1→phi2 大),之后饱和(phi2→phi4 平)。**
- **delta-vs-additive(DeltaNet 先擦后写,A5000 5-seed)**:固定 2000 步:`m=16 平手 → m=48/64 delta 崩(0.030/0.013)、additive 守 0.656/0.598`。**早停复跑(ablation.py 早停版,~1860 步)确认 + 加方差/t**:`m=16 delta 0.559±0.442 / add 0.898±0.015 → m=48 delta 0.034±0.016 / add 0.688±0.021(t=−54.6)→ m=64 delta 0.013 / add 0.598`。**REFUTED**。新洞察:delta 从 m=16 **双稳(宽方差,部分种子能学)** → m=48 **一致崩溃(窄方差)**,崩溃阈值**陡**、落在 m16–48 间;additive 平滑退化(0.898→0.688→0.598)。成因:擦除为覆写设计,本任务不同键(无覆写),密键下 `v_old=φ(k)ᵀS` 混入重叠键值 → 擦除污染、连带擦坏(机制),或 WY 解病态(数值);实用结论同。→ `use_delta` 默认关;recall 杠杆是 H·dc²/深度(非 phi、非 erase)。详见 §6.2。
- **hop2 / T_a**(induction/multihop/reverse):消融 drop 全在噪声内(multihop full 仅 **0.204**,模型太小做不动 → 无法区分通路贡献)。与历史"冗余"结论一致(§10.3/10.4),**不重复跑**。
- **通路幅度**(multihop 训练):layer0 `||Xl1||=3.44, ||Xl2||=5.83(Xl2/Xl1=1.69!), ||Ta||=0.75`;layer1/2 幅度**坍缩 ~10×**。→ hop2 在 layer0 幅度大(**被用**)却消融中性(**可替代**)= 冗余但非闲置;深层通路幅度极小(残差流主导,层只做小修正)。
- **conv vs lin**:reverse `conv 0.988 > lin 0.965`(+0.023,噪声内),conv 不劣 → 保持 conv 默认。**已结论,不重复跑**。
- **速度教训**:旧 ablation 跑 ~58 模型 + 每步 CPU 造数(B 次 randperm launch)+ H2D + 无 bf16 → 慢得离谱。已修:只跑未结论的 phi、GPU-native 向量化造数、bf16、模型数 →30。

---

### 10.12 lin_read 融合核:负结果 + TF32 发现(A5000 实测)

尝试用 Triton 融合 `lin_read` 提速,**未成功**,但定位清晰:

- **TF32 对未归一化线性注意力是灾难**:`tl.dot` 默认 TF32 在 fail-case(dk=128,dv=64,L=512)误差 **255**(放大 3.2 万倍);ieee 全 fp32 → **7.8e-3** 正确。原因:未归一化 N 累加大数值,TF32 ~1e-3 相对误差 × 大数 = 大绝对误差。`lin_read_fast` 经 N/D 归一化后 Triton 等价到 **4.77e-7**(误差相消)。→ **ieee 强制**。
- **benchmark(fwd+bwd,A5000,两轮)**:原 lin_read **7.12–7.25ms**(baseline,稳);Triton ieee **8.6–8.7(x0.82–0.84,两轮都稳定最慢)**;lin_read_fast torch **6.69–7.83(x0.93↔x1.06,横跨 1.0)**。→ **Triton 真慢**(ieee fp32-dot 比 TF32 慢 4–8×吃掉融合收益 + 手写 torch 反向慢于 autograd);**torch-fast 与 baseline 平手**(抖动全在其自身测量,~6% 差距被噪声淹没,非可靠加速)。
- **结论**:手写核**正确但不划算**,默认关(`_USE_TRITON=False`)。`lin_read` 提速正解 = `torch.compile` 包原版(已预分配/编译友好)。要 Triton 赢需再写 Triton 反向,但 ieee 约束下难超 torch。`tl.dot` 另有 M/N/K≥16 限制(prod 128/64 不受影响,已加 <16 护栏回退)。

---

### 10.13 本轮架构精简 + 推理快路(单读 / head_born 关 / chunk256 / Born 地板纠正 / CUDA-graph)

本会话基于既有消融数据做了三处确定性精简,并补齐 O(1) 解码的内核快路。**298M → 253.22M**(释放 44.7M:born ~6.5M + `to_Tx/to_Cx` ~38M + `agg_r`/门),全部经 T0–T9 自检(T9 增量==整段 forward EXACT 2.68e-7,T3 因果保持)。

| 改动 | 依据 | 代价归类 |
|---|---|---|
| **删 lerp 门 + hop2(单读)** | hop2 读同一共享账本、信息已被 Xl 覆盖(结构性冗余);翻译器版有害 Δ−0.118,lerp 版 `−hop2` 中性/更低(§10[2])。门的唯一消费者是 hop2,"精炼"本质需两次读 → 门收益与 hop2 绑死,无独立证据 | 数据支持:每层扫描 2→1,投影 6→4,质量预期持平/略好 |
| **`head_born` 默认关(纯线性头)** | Born 抑制仅玩具任务 ~16% forbidden-mass 微利、真实 LM 零收益(§10.5);线性 base 已无地板 | 玩具级代价,生产无损;省 born 参数 + 2 个 (B,L,V) GEMM + 显存 |
| **`chunk_len` 128 → 256** | 纯工程旋钮,数学不变(§9.1[T0] fp32 7.2e-7) | 证明零代价:更少串行块/launch |

**Born 地板纠正(撤回前一版误判):** 前文 §6.1 的 ~3.7 nats 地板**只对纯 Born 成立**。曾把 FineWeb val 卡 ~4.0 归因于头地板,**已撤回**——同款模型 + 头在 TinyStories 到达 **1.35 < 3.70**,硬地板下不可能;4.0 是**数据难度**(广域网页对 0.33B token/253M 的正常水平),非头限制。正解=更多 token + 延长 lr 余弦。

**训练路零改动(已核验):** 与上一会话逐字对比,`train.py` 的优化器/权重衰减策略/lr 调度/loss/训练循环**字节级一致**;本会话 `train.py` 仅改 `make_batch` 随机索引 dtype→int64(无害)+ 动态 topup(默认 20k 跑不触发);`model_lerp.py` 改动**仅在推理解码**(generate/step/快路)+ 上述架构精简。4.0 与本会话改动无因果。

**CUDA-graph / torch.compile 推理快路(GPU 待验):** 为消 O(1) 解码的逐 token kernel-launch 开销,新增固定缓冲 `step` 循环:cuda 上**录制-重放**(`fast='graph'`,录现有 eager 内核、**不走 Inductor codegen → 规避 Windows ptxas 崩**)为主;`torch.compile(reduce-overhead)`(`fast='compile'`,走 Inductor → 可能崩 → 自动回退)为可选。**架构红利**:model_lerp 状态定长 → step 静态形状 → 可 CUDA-graph;Transformer KV-cache 增长 → 不可干净 graph(诚实的非对称优势)。CPU 已验证固定缓冲 step==eager==整段 forward(贪心逐 token 一致);graph 录制/重放为 GPU-only,A5000 待验,eager 回退保正确。

### 10.14 头对头受控对比(已训 253M 单读 vs 配平 GPT,comparison.png)

控制变量:同 neo V50257 / 同 FineWeb `.bin` / 同 20000 步 / 同 bs2-block2048-accum4 / 配平 253M / bf16。anchor best_val 3.9617。

| 维度 | model_lerp | anchor(GPT) | 判定 |
|---|---|---|---|
| 参数 | 253M | 253M | 配平 |
| val loss / ppl | 4.14 / 62.69 | 3.98 / 53.66 | **质量略输(+17% ppl)** |
| 训练 tok/s | 12021 | 18118 | **慢 ~34%**(顺序 scan 低 MFU) |
| 训练显存 | 12GB | 10GB | 略高 |
| 推理 tok/s(基准长度) | 353 | 330 | 持平/略高 |
| 推理吞吐 vs 长度 | ~370 平到 16384 | 衰减 + **ctx=2048 硬封顶** | **碾压** |
| 推理显存 vs 长度 | 定长 ~3.7GB | KV-cache 线性涨 | **碾压** |
| 推理延迟 vs 长度 | ~O(N) 线性 | ~O(N²) + 过不了 2048 | **碾压** |
| 样本质量 | 流畅但循环("black/white American") | 流畅但循环("University of X") | 同档,均 253M/0.33B 失败模式 |

**结论:** 推理 scaling 三图(吞吐/显存/延迟)是线性注意力论点的实打实兑现,用配平 Transformer 基线证明;质量"近而略逊"(+17% ppl),训练更慢(~66% MFU)。与 Mamba/RWKV/GLA 等次二次架构在小预算下的典型表现一致(通常对 Transformer 近而略逊 ppl,赢在推理 scaling)。**作为"架构成立 + 推理优势为真"的验证 = 成功;作为"质量胜过 Transformer" = 未达**。gap 是本质还是欠训,需长训(max_steps→~100k + 延长 lr 周期)看收敛/稳定来区分。

### 10.15 涌现行为探针框架(emergent_probe.py,从结构推可证伪预测)

那 6 个吞吐/质量指标不是全部——架构差异理应有**可观测的行为签名**。下表从 model_lerp 的具体结构推出 6 个探针(每个发出可证伪预测 + 实测 + 判定),用同两个已训模型严格控制变量。**注:这些是待验预测、不是已确认结论;数字由脚本跑出填入,不臆造。**

| 探针 | 结构依据 | 可证伪预测 | 结果 |
|---|---|---|---|
| P1 长度外推 | 无位置嵌入,O(L) 递归 | GPT per-pos NLL 在 ctx=2048 处**硬悬崖**;model_lerp 继续(平=真外推,升=状态饱和) | 待跑 |
| P2 in-context 归纳 + 容量 | 状态 = kᵀv 外积累加 + 稀释 | [R;R] 第二份 NLL 下降=归纳;model_lerp 下降幅度随 n 增大而**衰减**(容量天花板);GPT ctx 内持平 | 待跑 |
| **P3 状态轨迹(独有)** | **固定大小可读状态 S** | ‖S‖ 增长后饱和;写入 ‖ΔS‖ 与 token surprise **正相关**(surprise-gated memory);eot 处响应。**Transformer 无单一状态向量,根本无法做此观测** | 待跑 |
| P4 熵 + 校准 | 线性注意力无 softmax 竞争 | 积分状态预测更平滑(熵更高/更不尖);ECE 校准签名不同 | 待跑 |
| P5 重复/吸引子 | 固定状态可收敛到不动点 | 长生成 model_lerp 循环更早/更紧(distinct-n 更低、onset 更早) | 待跑 |
| P6 扰动自愈 | 污染进入状态被稀释 | 单 token 污染后 dNLL **随距离衰减**(状态遗忘坏 token);GPT 注意力内持续可见 | 待跑 |

**最具区分度的是 P3**:model_lerp 的固定状态是可逐 token 读出的充分统计量,Transformer 的"状态"是增长的 KV-cache、无对应单一向量——这是 model_lerp **结构性独有**的可解释性观测,不是头对头而是"我们的架构暴露了 Transformer 暴露不了的东西"。CPU tiny 模型已验证全 6 探针接口/出图(byte tok,无需 tiktoken/ckpt);真实数字待 A5000 跑真 ckpt 填入。

---

## 11. 工程性质

| 维度 | 结论 |
|---|---|
| 显存(状态) | `O(B·H·dk·dv)`,无 L 维 → 永不爆炸;L=512 比 naive 省 **512×** |
| 复杂度 | 前向 wall-time 对 L 近线性(log-log 斜率 ~0.57,非 ~2) |
| 解码 | 携带状态 `(B,H,dk,dv)` 与 L 无关 → 原则 O(1)/步(注:`generate()` 当前重算全 ctx,需加步缓存才实现 O(1)) |
| grad_checkpoint | loss/grad 精确等价(本会话验:`|dloss|=0, |dgrad|=0`) |
| 显存调优(24GB) | 激活/logits/head ∝ `batch×seq`;参数+优化器固定(~6GB);`grad_accum` 串行**不增峰值**。默认改 `batch 16→12, accum 2→3`(激活峰值 −25%,等效 token 18432 ≥ 原 16384,模型与 loss 动态不变) |

---

## 12. 训练配方(`train.py`)

- 数据:HF-cache arrow → tokenize 一次到 memmap `.bin` → 随机窗口采样。
- 优化:bf16 AdamW,cosine + warmup,grad-accum/clip,可选 grad-ckpt;periodic eval;ckpt save/resume;CSV + loss 曲线 PNG;每步 `loss_trace.csv` 供谱分析。
- **NOWD(不衰减)**:`('U_re','U_im','out_re','out_im')` —— FeynmanHead 传播子(近恒等 Born 相位)+ 复数词嵌入不得衰减(否则抹掉 Born 抑制);name-match 兼容 `torch.compile` 的 `_orig_mod.` 前缀。
- CLI 架构旋钮:`--phi_expand --no_qk_conv --think_mode --nest_mode --head_cap --head_eps --n_paths`,可命令行做架构消融。
- 早断言:`H = n_lobe·n_cortex` 必须整除 `nest_branch^nest_levels`(=4)。
- 默认 tokenizer = neo(GPT-2 BPE 50257):大词表稀释头饱和 → 更小 loss-spike 幅度。
- `--spike_skip` 默认 OFF(研究原始动态);仅在灾难性发散时设 ~4。

---

## 13. 未解问题与下一步

1. **~~大词表 CE 地板~~(已解,本会话)**:纯 Born 头在 neo 有 ~3.70 nats 硬地板 → 已加线性 base 消除(§6.1,neo CE 双验 →0)。剩余:若坚持纯 Born,用 `head_eps=1e-3`(neo 地板 ~1.63)。
2. **recall = 记忆力(分块并行 Delta Rule 已实现 + GPU 验证:对 distinct-key recall REFUTED)**:线性注意力加性账本干扰是瓶颈(0.358)。**已实现 DeltaNet 先擦后写**(`use_delta`,§6.2):`S += β·φ(k)ᵀ(v − φ(k)S)`,CPU 验证正确(稳定/因果/overwrite/gradcheck)。**GPU 结果**:delta 在密键区崩溃、additive 守 0.6 → delta 非本任务解药(§6.2/§10.11),`use_delta` 默认关。**下一步**:① recall 杠杆 = **H·dc²(值容量)+ 深度 L**,**非 phi**(phi 已扫,phi2 后饱和,§10.11)、**非 erase**(delta REFUTED);phi2 够用(清碰撞悬崖即饱和)。若探容量轴:`ablation.py --exp sep --scan cortex`(H)或 `--scan dcell`(dc),验证 `recall=logistic(log2(H·dc²))`。② ~0.70 天花板的成因(疑 elu+1 特征图全正→串扰)是开放问题,要破需换特征图(研究向);③ delta 留给**覆写/状态追踪**任务(其设计强项,本任务未测)。
   - **观察力/思维力**:induction(token→token,已 1.000)= 观察力;multihop(value-of-value,仅 0.204)= 思维力,受限于记忆力 → 修记忆力抬高其上限。
3. **深度梯度失衡**:极深(>32 层)first/last 比增长 → depth-scaled init / LayerScale。
4. **O(1) 解码**:`generate()` 加步缓存以兑现 O(1)/步。
5. **n_layers 标定**:L=2 必需已坐实;L=4 vs L=12 对推理的边际收益未坐实。
6. 满规模(H=256)参数代价大(~94M/层),需权衡 cortex 数 vs `d_cell` 容量(容量律指导)。
7. **lin_read 全反向融合**:`lin_read_triton.py` 当前只融前向(反向走已验证 torch 路径),Triton 反向核是训练全程提速的下一杠杆。

---

## 14. 文件清单(当前,已去 `v2` 后缀)

| 文件 | 内容 | 本会话状态 |
|---|---|---|
| `model_lerp.py` | 四级骨架(**单读:删 lerp 门 + hop2 → 四投影**) + 八升级 + 头线性 base(**head_born 默认关**) + chunk_len 256 + **O(1) 增量解码 + CUDA-graph/compile 快路(eager 回退)** + Delta Rule(`use_delta`,§6.2) | CPU 自检 **T0–T9 全过**(T9 增量==整段 forward EXACT 2.68e-7);**253.22M**(原 298M,删门/hop2/born 释放 44.7M) |
| `train.py` | 工业训练器(`--no_head_linear_base` / **`--head_born`(默认关,纯线性头)** / `--use_delta`,**chunk_len 256**,torch.compile 默认开) | 全管线烟测通,loss 4.10→2.37(TinyStories);FineWeb val ~4.0 = 数据难度非头地板(§6.1) |
| `prepare_data.py` | 流式拉取 **FineWeb sample-10BT**(通用高质量网页)→ token 化成 train.py 吃的 `.bin`(复用 train 的 tokenizer);**按 `--max_gb`(默认 10GB 文本)停止下载,绝不全量拉 TB**;**替代 TinyStories 玩具集** | 核心写盘逻辑 CPU 单测通(按 GB 卡停、val/train 不相交、dtype、连续);流式部分需用户机联网跑 |
| `test.py` | 72 测试诊断台 | **65 PASS / 6 WARN / 0 FAIL**(delta 默认关无回归) |
| `ablation.py` | **phi 结论 + delta-vs-additive recall**(`--exp phi/delta/both`);GPU-native+bf16,通用 SE 显著性 | CPU smoke 通;待 A5000 `--exp delta` 验记忆力收益 |
| `lin_read_triton.py` | lin_read 融合提速尝试:torch 分块 fwd + 手写 bwd + Triton 前向核 + drop-in | **负结果(A5000 实测)**:ieee 数值正确(归一化 4.77e-7),但**比原 lin_read 慢**(Triton x0.84 / torch-fast x0.93)。默认关。提速正解=`torch.compile`。详见 §10.12 |
| `test_triton.py` | Triton 核仪器化诊断(精度/块数/维度/逐位漂移),出结构化报告 | A5000 跑通:定位根因=TF32 |
| `anchoring_group.py` | 受控对照:配平标准 Transformer(SDPA=flash、绝对位置嵌入)+ 8 面板 `comparison.png`(资源/质量/雷达 + 推理吞吐/显存/延迟 vs 长度 + 样本)。`--compare <lerp_ckpt>` 加载两 ckpt 评测出图;`match_dff` 按 ctx 自动配平 d_ff | 默认 bs2/block2048/accum4 对齐 train.py;ctx2048→d_ff 6069 仍配平 253M。产出 §10.14 |
| `emergent_probe.py` | 涌现行为探针:6 个从结构推的可证伪探针(P1 长度外推 / P2 归纳·容量 / **P3 状态轨迹·独有** / P4 熵·校准 / P5 重复·吸引子 / P6 扰动·自愈),复用 `load_lerp/load_anchor`,出多面板 `emergent.png` + 每探针 预测/实测/判定 | **CPU tiny 全 6 探针接口/出图验证通**(`--selftest`,byte tok 无需 tiktoken/ckpt);真实数字待 A5000 跑真 ckpt。框架见 §10.15 |

> 注:`model_lerp.py` 现为本架构内容(已覆盖旧扁平版)。其他仍 `from model_lerp import` 的旧脚本(`generate.py`、`experiments.py` 等)引用旧扁平接口,需同步才能跑。

---

## 15. 出处与诚实声明(三类标注)

为遵守"以假装理解为耻、以诚实无知为荣":

| 类别 | 含义 | 本档案出处 |
|---|---|---|
| **SPEC** | 架构定义/数学 | §2–§8(代码即定义) |
| **CPU-VERIFIED(本会话)** | 沙盒 tiny 配置实测,验正确性/因果/不爆炸/可学习 | §9 全部数字(我的本会话工具输出) |
| **GPU-EMPIRICAL(历史)** | 此前会话的规模化 GPU 实验,数字已对 transcript 核验 | §10 全部数字(本会话**未重跑**) |

- §9 = 本会话直接产出。§10 = 历史结论(transcript 核验:`0.743/0.862, 0.358, 0.931/0.966, 0.0184/0.0218, 0.052/29.7/98.1, 7.2e-7/1.4e-4, 0.207, 0.904/0.876, 0.827/0.977, eff_rank=19` 等已比对)。
- 规模化重跑请在 A5000 上:`python train.py --tokenizer neo --n_layers 8 --batch_size 12`;recall 专项与组件功能消融用 `python ablation.py --device cuda --seeds 3 --steps 4000`。
