# PESO 论文笔记：Continual Low-Rank Adapters for LLM-based Generative Recommender Systems

> 组会汇报目标：让听众理解这篇论文为什么要做、方法怎么做、核心公式和图表说明什么、实验结果能证明什么。

---

## 1. 一句话总结

这篇论文研究 **LLM-based Generative Recommender Systems** 中的连续学习问题：用户、商品和用户偏好会随时间变化，模型需要持续适应新数据，同时保留仍然有用的长期偏好。

论文提出 **PESO**：

> **PESO = Proximally rEgularized Single evolving lOra**

核心思想是：

> 只维护一个持续演化的 LoRA adapter，并在每个新时间阶段训练时，用一个 proximal regularizer 约束当前 LoRA 不要离上一阶段 LoRA 太远。

它不是简单保存所有旧 adapter，也不是完全自由地持续微调，而是在 **适应新兴趣** 和 **保留长期偏好** 之间做平衡。

---

## 2. 背景：LLM 如何做生成式推荐

传统推荐系统通常是排序任务：

```text
给定用户 u 和候选商品集合 I，给每个商品打分，然后排序。
```

这篇论文关注的是 **生成式推荐**：

```text
给定用户历史交互序列，让 LLM 直接生成下一个 item 的 token。
```

例如用户历史商品被表示成 semantic ID：

```text
<a_144><b_72><c_103><d_217>
```

一个商品不是一个普通 item ID，而是由多个语义 token 组成。模型输入类似：

```text
Based on the items that the user has interacted with:
<a_i1><b_j1><c_k1><d_l1>,
<a_i2><b_j2><c_k2><d_l2>,
...
can you determine what item would be recommended to the user next?
```

然后模型自回归生成下一个 item 的 semantic ID。

---

## 3. 核心问题：推荐场景中的 continual learning 有什么特殊性

真实推荐系统的数据不是静态的，而是按时间连续到来：

```text
D1 -> D2 -> D3 -> D4 -> D5
```

每个阶段都会出现：

- 新用户；
- 新商品；
- 老用户的新行为；
- 用户兴趣漂移。

例如：

```text
用户以前喜欢动作片，现在开始喜欢爱情片。
用户以前购买学生用品，毕业后开始购买办公用品。
用户短期关注旅行用品，但长期仍然喜欢某个品牌。
```

因此推荐系统中的 continual learning 不是简单地“记住所有旧任务”。

论文强调：推荐任务中的稳定性和可塑性有特殊含义。

| 概念 | 在普通 continual learning 中 | 在推荐系统中 |
|---|---|---|
| Stability | 保持旧任务性能 | 保留仍然有效的长期偏好 |
| Plasticity | 学会新任务 | 适应新兴趣、新用户、新商品 |
| Forgetting | 通常是坏事 | 有时遗忘过时偏好是必要的 |

所以推荐系统不是为了预测用户过去喜欢什么，而是为了预测用户未来可能喜欢什么。

---

## 4. 论文要解决的具体问题

论文考虑一个 LLM recommender：

1. 先在初始数据块 `D1` 上训练。
2. 后续数据块 `D2, ..., DT` 按时间到来。
3. 每个阶段只用当前数据更新模型。
4. 希望模型既能适应当前数据，又不丢掉有用的长期偏好。

形式化任务：

```text
Given:
  已经在 D1 上训练好的 LLM-based recommender
  后续时间数据块 D2, ..., DT

Goal:
  在每个阶段 t，用 Dt 更新模型，使其更好地推荐未来 item。
```

---

## 5. 必须理解的基础公式

### 5.1 阶段数据定义

论文将第 `t` 个阶段的数据写成：

```math
D_t = \{(x_u, y_u): u \in U_t\}
```

其中：

- `x_u`：用户 `u` 的历史交互序列；
- `y_u`：用户下一个交互 item；
- `U_t`：第 `t` 个阶段活跃用户集合。

可以理解为：

```text
输入：用户过去交互过的 item 序列
输出：用户下一个 item
```

---

### 5.2 生成式推荐损失

目标 item 由多个 semantic ID token 组成：

```math
y = (y_1, y_2, ..., y_M)
```

LLM 自回归生成：

```math
p_\theta(y \mid x)
= \prod_{m=1}^{M} p_\theta(y_m \mid x, y_{<m})
```

训练损失是交叉熵：

```math
L_{\mathrm{ce}}^{D_t}
=
\mathbb{E}_{(x,y)\sim D_t}
\left[-\log p_\theta(y \mid x)\right]
```

含义：

> 模型不是一次性分类出 item ID，而是像语言模型一样逐 token 生成 item 的 semantic ID。

---

### 5.3 LoRA

LoRA 冻结原始大模型参数，只训练低秩增量：

```math
W = W_0 + \Delta W
```

```math
\Delta W = BA
```

其中：

- `W0`：冻结的 LLM 原始权重；
- `A, B`：低秩矩阵；
- 训练时只更新 `A, B`。

LoRA 适合 continual learning 的原因：

- 参数量小；
- 训练成本低；
- 每个阶段可以只更新 adapter；
- 不需要全量微调 LLM。

---

## 6. Baseline 1：Single Evolving LoRA

Single Evolving LoRA 是最直接的方法。

每个阶段只维护一个 LoRA：

```math
W_t = W_0 + B_t A_t
```

第 `t` 阶段的 LoRA 从上一阶段初始化：

```text
A_t <- A_{t-1}
B_t <- B_{t-1}
```

然后在当前数据 `D_t` 上继续训练。

### 优点

它很灵活，适应新数据快。

### 缺点

它会不断覆盖上一阶段参数，所以容易遗忘有用的长期偏好。

可以把它理解为：

```text
Single Evolving LoRA = 追逐现在，但可能忘记过去。
```

---

## 7. Baseline 2：Cumulative LoRA

Cumulative LoRA 是视觉 continual learning 中常见的方法。

它保存过去每个阶段的 frozen adapter，并在当前阶段再训练一个新 adapter：

```math
W_t
=
W_0
+ \sum_{i=1}^{t-1} \alpha_i \hat{B}_i \hat{A}_i
+ B_t A_t
```

其中：

- 过去 adapter 冻结；
- 当前 adapter 可训练；
- `α_i` 是过去 adapter 的权重；
- `All` 表示使用所有过去 adapter；
- `Latest` 表示只使用最近一个过去 adapter。

### 直觉上的优点

保存过去 adapter，似乎可以防止遗忘。

### 推荐场景中的问题

推荐中的用户偏好不是相互独立的任务，而是连续变化的。

过去 adapter 中混合了两类信息：

```text
仍然有用的长期偏好
已经过时的短期偏好
```

如果直接冻结并累加旧 adapter，模型无法区分哪些该保留、哪些该遗忘。

可以把它理解为：

```text
Cumulative LoRA = 保存过去，但可能过于僵硬。
```

---

## 8. Table 1：为什么 Cumulative LoRA 不适合自然推荐场景

Table 1 是这篇论文非常重要的动机表。它回答：

> 为什么已有 continual LoRA 方法在推荐任务中不够好？

表中的结果是相对于 **Single Evolving LoRA** 的 NDCG@5 提升率。

```text
0%：和 Single Evolving LoRA 一样
正数：比 Single Evolving LoRA 好
负数：比 Single Evolving LoRA 差
```

---

### 8.1 Table 1 的 design choice

Table 1 比较了三个设计维度。

#### 1. Learnable mag.

是否学习过去 adapter 的权重大小 `α_i`。

如果不可学习：

```text
过去 adapter 权重固定。
```

如果可学习：

```text
模型学习过去 adapter 该占多大比例。
```

对应方法如 SD-LoRA。

---

#### 2. Only latest

是否只使用最近一个过去 adapter。

如果不是 only latest：

```text
使用 LoRA_1 + LoRA_2 + ... + LoRA_{t-1}
```

如果是 only latest：

```text
只使用 LoRA_{t-1}
```

这个设计很重要，因为推荐中很久以前的兴趣可能已经过时。

---

#### 3. Param inherit

当前阶段的 LoRA 是否从上一阶段参数初始化。

如果没有 parameter inheritance：

```text
当前 LoRA 重新初始化。
```

如果有 parameter inheritance：

```text
当前 LoRA 从上一阶段 LoRA 初始化。
```

推荐中的用户兴趣通常是连续演化的，所以参数继承更符合真实场景。

---

### 8.2 两种任务设置

Table 1 比较了两种数据划分。

#### User-disjoint split

不同阶段的用户基本不重叠：

```text
D1: 一批用户
D2: 另一批用户
D3: 再一批用户
```

这更接近传统 continual learning 的设置，任务之间相对独立。

#### Natural split

按照真实时间划分：

```text
D1 -> D2 -> D3 -> D4 -> D5
```

同一个用户可能跨阶段反复出现，兴趣会随时间漂移。

这才是推荐系统更真实、更重要的设置。

---

### 8.3 SumLoRA All

设计：

```text
Learnable mag. = 否
Only latest = 否
Param inherit = 否
```

含义：

```text
每个阶段新建一个 LoRA；
过去所有 LoRA 都冻结并累加；
当前 LoRA 不从上一阶段继承。
```

结果：

```text
User-disjoint: -8.13%
Natural split: -26.77%
Diff: 18.64%
```

解释：

SumLoRA All 在真实时间划分下非常差。原因是它把所有过去 adapter 都强行保留下来，而很早以前的 adapter 里可能有大量过时偏好。

它反映的问题：

> 直接保留所有旧 LoRA 不适合自然推荐场景。

---

### 8.4 SumLoRA Latest

设计：

```text
Learnable mag. = 否
Only latest = 是
Param inherit = 否
```

含义：

```text
只使用最近一个冻结 adapter；
当前 LoRA 仍然重新初始化。
```

结果：

```text
User-disjoint: -12.20%
Natural split: -22.05%
Diff: 9.85%
```

解释：

只使用最近 adapter，确实减少了更早历史的干扰，所以 natural split 比 SumLoRA All 稍微好一些。

但它仍然很差，因为当前 LoRA 不继承上一阶段参数，每个阶段都像重新开始训练，破坏了用户偏好的连续演化。

它反映的问题：

> 只保留最近 adapter 还不够，参数连续继承很重要。

---

### 8.5 SumLoRA All + Inherit

设计：

```text
Learnable mag. = 否
Only latest = 否
Param inherit = 是
```

含义：

```text
累加所有过去 adapter；
当前 LoRA 从上一阶段继承参数。
```

结果：

```text
User-disjoint: -3.25%
Natural split: +1.57%
Diff: -4.82%
```

解释：

加入 parameter inheritance 后，natural split 从 `-26.77%` 大幅提升到 `+1.57%`。

这说明：

> 推荐任务中的阶段不是彼此独立的，当前阶段应该从上一阶段平滑演化。

但它仍然使用所有过去 adapter，因此还会受到旧偏好和过时偏好的干扰。

它反映的问题：

> 参数继承非常关键，但累加所有过去 adapter 仍然不理想。

---

### 8.6 SumLoRA Latest + Inherit

设计：

```text
Learnable mag. = 否
Only latest = 是
Param inherit = 是
```

含义：

```text
只使用最近一个过去 adapter；
当前 LoRA 从上一阶段继承。
```

结果：

```text
User-disjoint: 0.00%
Natural split: +2.36%
Diff: -2.36%
```

解释：

这是 SumLoRA 变体中最合理的设计，因为它同时满足：

```text
不累加太久远的 adapter；
当前 adapter 从上一阶段连续演化。
```

所以它在 natural split 中比 Single Evolving LoRA 更好。

但提升仍然有限，因为它还是使用一个冻结的过去 adapter。冻结 adapter 不能精细区分哪些旧偏好仍然有效、哪些已经过时。

它反映的问题：

> 最近 adapter + 参数继承比原始 cumulative LoRA 更适合推荐，但仍然偏僵硬。

---

### 8.7 SD-LoRA Latest + Inherit

设计：

```text
Learnable mag. = 是
Only latest = 是
Param inherit = 是
```

含义：

```text
只使用最近 adapter；
当前 adapter 继承上一阶段参数；
并且学习过去 adapter 的权重大小。
```

结果：

```text
User-disjoint: +3.25%
Natural split: +0.79%
Diff: 2.46%
```

解释：

它在 user-disjoint setting 中表现不错，因为当任务较独立时，学习 adapter 权重可以帮助选择过去 adapter 的贡献。

但在 natural split 中，它不如 SumLoRA Latest + Inherit。

原因是：

```text
一个 adapter 内部同时包含有用长期偏好和过时短期偏好。
学习一个整体权重 α，只能决定整个 adapter 多用还是少用。
它无法在 adapter 内部细粒度地区分哪些方向该保留、哪些方向该更新。
```

它反映的问题：

> adapter-level weighting 太粗糙，不能解决推荐偏好漂移中的细粒度纠缠问题。

---

### 8.8 Diff 列说明什么

Diff 是：

```text
User-disjoint 结果 - Natural split 结果
```

Diff 越大，说明该方法在真实时间推荐场景中退化越严重。

例如：

```text
SumLoRA All:
User-disjoint = -8.13%
Natural split = -26.77%
Diff = 18.64%
```

这说明 SumLoRA All 依赖“不同阶段任务相对独立”的假设。一旦进入真实时间推荐场景，用户兴趣连续漂移，旧 adapter 就会强烈干扰当前推荐。

---

### 8.9 Table 1 的核心结论

Table 1 提炼出五个结论。

1. **累加所有旧 adapter 是有害的。**

   `SumLoRA All` 在 natural split 中最差，说明不能盲目保存所有旧偏好。

2. **只使用最近 adapter 比使用所有 adapter 更合理。**

   因为越久远的信息越可能过时。

3. **参数继承非常重要。**

   加入 inherit 后，结果大幅改善，说明推荐偏好是连续演化的。

4. **学习 adapter 权重不够。**

   因为旧 adapter 内部有用信息和过时信息纠缠在一起，一个整体权重无法细粒度处理。

5. **推荐任务需要 controlled stability，而不是 rigid reuse。**

   也就是既不能完全自由微调，也不能僵硬复用旧 adapter。

这正是 PESO 的动机：

> 维护一个单一演化 adapter，并用 proximal regularizer 控制它相对上一阶段的变化。

---

## 9. PESO 方法

PESO 的设计原则是：

1. 不维护多个 LoRA adapter，避免假设不同阶段任务独立。
2. 只维护一个持续演化的 LoRA。
3. 每个新阶段训练时，用正则项让当前 LoRA 靠近上一阶段 LoRA。
4. 让数据损失和正则项竞争，自动决定哪些参数该变、哪些参数该保留。

---

## 10. PESO 的核心目标函数

论文把所有 LoRA 参数拉平成向量：

```text
v_t = 当前阶段 LoRA 参数
v_{t-1} = 上一阶段 LoRA 参数
```

然后按模块分组：

```text
g = 某一层某个 LoRA A 或 LoRA B 的参数组
```

PESO 的通用目标函数：

```math
\begin{aligned}
L_t
&=
L_{\mathrm{ce}}^{D_t}
+ \frac{\lambda}{2}
\sum_{g=1}^{G}
\left\lVert
v_t^{(g)} - v_{t-1}^{(g)}
\right\rVert^2_{H_{t-1}^{(g)}} .
\end{aligned}
```

其中：

- 第一项 `L_ce`：让模型拟合当前阶段数据；
- 第二项 proximal term：让当前 LoRA 不要离上一阶段 LoRA 太远；
- `λ`：控制稳定性和可塑性的权衡。

直观理解：

```text
当前数据损失：拉着模型去适应新偏好。
近端正则项：拉着模型保留上一阶段的知识。
最终模型：在这两股力量之间找到平衡。
```

---

## 11. PESO 的理论解释：为什么它能自适应平衡

论文将当前阶段数据损失近似为二次形式：

```math
L_{D_t}(v)
\approx
\frac{1}{2}
\left(v - v_t^{\star}\right)^\top
\Sigma_t
\left(v - v_t^{\star}\right).
```

其中：

- `v_t^*`：如果只看当前数据，最理想的 LoRA 参数；
- `Σ_t`：当前数据在 LoRA 参数空间中支持哪些方向；
- `v_{t-1}`：上一阶段 LoRA 参数。

加入 proximal regularizer 后，在某个方向 `q_k` 上，最优解近似是：

```math
\begin{aligned}
\left\langle \hat{v}_t, q_k \right\rangle
&=
\frac{\sigma_k^2}{\sigma_k^2+\lambda}
\left\langle v_t^{\star}, q_k \right\rangle
+
\frac{\lambda}{\sigma_k^2+\lambda}
\left\langle v_{t-1}, q_k \right\rangle .
\end{aligned}
```

这是理解 PESO 最重要的公式。

---

### 11.1 这个公式怎么理解

如果 `σ_k^2` 很大：

```text
当前数据强烈支持这个方向。
```

则：

```text
σ_k^2 / (σ_k^2 + λ) 较大
```

模型更靠近当前数据最优解 `v_t^*`。

含义：

> 有充分数据证据时，模型大胆更新。

---

如果 `σ_k^2` 很小：

```text
当前数据不支持这个方向，或者证据不足。
```

则：

```text
λ / (σ_k^2 + λ) 较大
```

模型更靠近上一阶段参数 `v_{t-1}`。

含义：

> 没有充分数据证据时，模型保留旧知识。

---

所以 PESO 的理论核心是：

> 它不是简单少改参数，而是在 LoRA 参数空间中，按当前数据对不同方向的支持程度，自适应决定更新幅度。

这就是论文所说的：

```text
data-aware, direction-wise guidance
```

---

## 12. PESO 实际使用的 softmax-KL proximal

通用 proximal 可以用 L2，但论文最终采用 per-module softmax-KL。

目标函数：

```math
\begin{aligned}
L_t
&=
L_{\mathrm{ce}}^{D_t}
+ \lambda \sum_{g=1}^{G}
D_{\mathrm{KL}}\!\left(
\operatorname{softmax}\!\left(v_t^{(g)}\right)
\,\middle\Vert\,
\operatorname{softmax}\!\left(v_{t-1}^{(g)}\right)
\right).
\end{aligned}
```

含义：

1. 对每个 LoRA 模块的参数做 softmax；
2. 得到一个模块内部的参数分布；
3. 计算当前参数分布和上一阶段参数分布的 KL 距离；
4. 如果模块内部结构变化太大，就惩罚。

---

### 12.1 为什么不用普通 L2

普通 L2：

```text
平等惩罚所有参数变化。
```

问题是：

```text
LoRA 模块内部不同参数的重要性不同。
```

softmax-KL：

```text
关注模块内部相对结构是否发生剧烈变化。
```

它更适合保护 LoRA adapter 内部已经形成的结构，同时允许模型在有数据证据的地方更新。

---

### 12.2 softmax-KL 的局部解释

论文证明 softmax-KL 在局部等价于二次正则：

```math
H = \operatorname{diag}(p) - pp^\top
```

并且可以理解为：

```math
K_{\mathrm{blk}}
\propto
\sum_g
\operatorname{Var}_{p^{(g)}}\!\left(\Delta^{(g)}\right)
```

也就是说：

> PESO 主要惩罚模块内部重要参数的相对结构被剧烈打乱，而不是完全禁止参数变化。

---

## 13. Figure 1：必须讲清楚的方法图

Figure 1 对比了 Cumulative LoRA 和 PESO。

### Cumulative LoRA

```text
W0 + LoRA1 + LoRA2 + ... + LoRAt
```

特点：

- 保存多个 adapter；
- 过去 adapter 冻结；
- 推理时累加；
- 存储随阶段数增长；
- 容易保留过时偏好。

### PESO

```text
只保留一个正在演化的 LoRA
当前 LoRA 训练时被 proximal regularizer 拉向上一阶段 LoRA
```

特点：

- 只有一个 adapter；
- 不累加所有历史 adapter；
- 通过正则控制参数变化；
- 存储复杂度是 `O(1)`；
- 更适合连续偏好漂移。

组会讲 Figure 1 时可以这样总结：

> Cumulative LoRA 是把过去显式加回来；PESO 是让当前 adapter 在上一阶段基础上受控演化。

---

## 14. 实验设置

### 14.1 数据集

论文使用 Amazon Review 数据集的三个类别：

- Musical Instruments；
- Movies & TVs；
- Books。

划分方式：

```text
前 60% 数据作为 D1，用于预训练；
后 40% 数据平均分成 D2, D3, D4, D5。
```

这模拟真实推荐系统：

```text
先有一批历史数据训练模型；
之后数据按时间持续到来。
```

---

### 14.2 模型和训练

Backbone：

```text
Llama-3.2 1B
```

训练方式：

- `D1` 上得到初始 LLM recommender；
- 后续阶段使用 LoRA continual fine-tuning；
- 每个 item 用 semantic ID 表示；
- 训练用 sliding window，窗口大小为 20。

---

### 14.3 评估方式

每个阶段使用 leave-one-out：

```text
倒数第二个 item：validation
最后一个 item：test
```

生成时：

```text
用 constrained beam search 生成 10 个合法 item。
```

评价指标：

- Hit@5；
- Hit@10；
- NDCG@5；
- NDCG@10。

Hit@k：

```text
真实 item 是否出现在前 k 个推荐中。
```

NDCG@k：

```text
真实 item 出现得越靠前，分数越高。
```

---

## 15. Table 2：主实验结果

Table 2 是论文最重要的主结果表。

比较方法包括：

- Pretrain；
- Single Evolving LoRA；
- InfLoRA；
- SumLoRA；
- SD-LoRA；
- PESO。

核心观察：

1. 所有 continual learning 方法整体优于 Pretrain。

   说明只用旧数据训练的模型无法适应用户兴趣变化。

2. Single Evolving LoRA 和 Cumulative LoRA 都不是稳定最优。

   Single Evolving LoRA 适应强但会忘；
   Cumulative LoRA 保存旧知识但过于僵硬。

3. PESO 在三个数据集和四个指标上整体最好。

   这说明 proximal regularization 能更好地平衡稳定性和可塑性。

---

## 16. Figure 2：不同正则项的消融

Figure 2 比较了不同 regularizer：

- Orthogonality；
- L2 proximal；
- LoRA-output KL；
- Per-rank KL；
- PESO 的 per-module softmax-KL。

重要结论：

1. Orthogonality 表现很差。

   在视觉任务中，正交约束常用于减少任务干扰。但推荐任务不是要让阶段之间互不干扰，而是要建模连续演化的偏好。

2. L2 proximal 不如 PESO。

   普通 L2 对所有参数一视同仁，缺少模块内部结构感知。

3. per-module softmax-KL 更合适。

   它在参数空间中保留模块内部结构，同时允许必要更新。

---

## 17. Table 3：稳定性和可塑性分析

Table 3 将用户分成两类：

### Dormant Users

早期活跃，中间消失，后期又回来。

它测试：

```text
模型是否保留长期偏好。
```

### New Users

只在后期出现。

它测试：

```text
模型是否适应新信号。
```

结果说明：

```text
Single Evolving LoRA:
  对新用户较好，但对 dormant users 较差，说明容易遗忘。

Cumulative LoRA:
  对 dormant users 较好，但对新用户适应不足，说明过于保守。

PESO:
  两类用户上都最好，说明稳定性和可塑性平衡更好。
```

这张表非常适合在组会中解释 PESO 的价值。

---

## 18. Figure 3：λ 的影响

`λ` 是 proximal regularizer 的权重。

当 `λ = 0` 时：

```text
PESO 退化成 Single Evolving LoRA。
```

当 `λ` 较小：

```text
模型更偏可塑性，适应新数据更强，但更容易忘。
```

当 `λ` 较大：

```text
模型更偏稳定性，不容易忘，但可能适应不足。
```

Figure 3 的趋势是：

```text
λ 从 0 增大时，性能先提升；
过大之后下降或进入平台期。
```

这说明：

> PESO 的关键不是正则越强越好，而是找到稳定性和可塑性的平衡点。

---

## 19. 附录中值得知道的补充实验

### 19.1 学习率敏感性

附录指出，continual stage 的数据远小于 pretraining data。

如果后续阶段仍使用预训练阶段的大学习率，容易过拟合或破坏已有知识。

经验结论：

```text
后续阶段学习率应缩小到预训练学习率的 0.05 到 0.1 倍左右。
```

---

### 19.2 与 full fine-tuning 和 retraining 比较

论文比较了：

- Full-parameter fine-tuning；
- 用 cumulative data 重新训练 LoRA；
- Single Evolving LoRA。

结果显示 Single Evolving LoRA 反而更好。

解释：

- 全量微调容易灾难性遗忘或适应不足；
- cumulative retraining 会平等对待旧数据和新数据，削弱近期偏好信号；
- LoRA 本身具有一定结构正则效果，适合连续微调。

---

### 19.3 效率分析

PESO 的效率优势：

```text
只存一个上一阶段 LoRA adapter。
```

存储复杂度：

```text
PESO: O(1)
Cumulative LoRA: O(T)
```

计算开销：

```text
PESO 只是在 loss 中增加一个 KL 正则项，不需要额外 forward pass。
```

---

## 20. 这篇论文的主要贡献

可以从三个层次讲。

### 贡献 1：重新分析推荐中的 continual learning

论文指出推荐任务和视觉 continual learning 不同。

推荐任务中：

```text
旧偏好不一定都该保留；
过时偏好可能会伤害当前推荐；
模型应该保留长期稳定偏好，同时忘掉过时短期偏好。
```

---

### 贡献 2：发现 cumulative LoRA 在自然推荐场景中不适合

Table 1 说明：

```text
Cumulative LoRA 在 user-disjoint setting 中还能工作；
但在 natural chronological split 中明显退化。
```

原因是：

```text
推荐中的用户偏好连续漂移；
旧 adapter 中有用信息和过时信息纠缠；
冻结并累加 adapter 太僵硬。
```

---

### 贡献 3：提出 PESO

PESO 的方法特点：

- 单一演化 LoRA；
- 使用 proximal regularizer；
- 实际采用 per-module softmax-KL；
- 理论上提供 data-aware、direction-wise 的更新解释；
- 实验上超过 Single Evolving LoRA 和 Cumulative LoRA。

---

## 21. 这篇论文的局限性

### 1. 提升幅度稳定但不算巨大

PESO 在多个数据集上稳定领先，但很多指标上的提升不是数量级变化。

这说明 LLM 推荐中的偏好漂移建模仍然很难。

---

### 2. Semantic ID tokenizer 是固定的

论文固定 item tokenizer，没有解决新 item 到来时 tokenizer 如何动态更新的问题。

真实系统中，新商品持续出现，tokenizer 自身也可能需要 continual adaptation。

---

### 3. `λ` 需要调参

不同数据集最优 `λ` 不同。

论文搜索范围包括：

```text
[0.5, 1.0, 2.0, 5.0, 8.0]
```

这说明 PESO 仍然依赖超参数选择。

---

### 4. 没有显式用户 embedding

传统 two-tower recommender 有显式 user embedding，更容易建模用户漂移。

LLM-based recommender 虽然有语义泛化能力，但如何显式建模用户兴趣变化仍然是开放问题。

---

## 22. 组会讲解建议

建议按下面顺序讲，逻辑最清楚。

### Step 1：先讲背景

可以这样说：

> 现在很多工作把推荐建模成 LLM 的生成任务，输入用户历史 item，输出下一个 item 的 semantic ID。但真实推荐数据是连续到来的，用户兴趣会发生漂移，所以模型需要 continual adaptation。

---

### Step 2：讲推荐中的 stability-plasticity 特殊性

可以这样说：

> 在普通 continual learning 中，我们通常希望保留所有旧任务性能。但推荐系统不是为了预测过去，而是为了预测未来。有些旧偏好是长期稳定偏好，应该保留；有些旧偏好已经过时，应该允许遗忘。

---

### Step 3：讲两个 baseline 的问题

```text
Single Evolving LoRA:
  一个 adapter 连续训练，适应强，但容易忘。

Cumulative LoRA:
  保存多个旧 adapter，稳定性强，但容易把过时偏好也固定下来。
```

---

### Step 4：重点讲 Table 1

Table 1 要讲出三个结论：

1. All past adapters 在 natural split 中很差；
2. Parameter inheritance 很重要；
3. Learnable adapter magnitude 不能解决 adapter 内部偏好纠缠。

然后自然引出：

> 所以我们需要一种 controlled stability，而不是 rigid reuse。

---

### Step 5：讲 PESO 方法

可以用一句话讲：

> PESO 保留 Single Evolving LoRA 的单 adapter 结构，但在每个阶段训练时加入 proximal regularizer，让当前 adapter 不要离上一阶段 adapter 太远。

核心公式：

```math
L_t
=
L_{\mathrm{ce}}^{D_t}
+ \frac{\lambda}{2}
\sum_g
\left\lVert v_t^{(g)} - v_{t-1}^{(g)} \right\rVert_H^2
```

---

### Step 6：讲理论直觉

最重要的是方向级插值公式：

```math
\hat{v}_t
\approx
\frac{\sigma^2}{\sigma^2+\lambda}v_t^{\star}
+
\frac{\lambda}{\sigma^2+\lambda}v_{t-1}
```

用自然语言解释：

> 如果当前数据强烈支持某个方向，模型就沿这个方向更新；如果当前数据证据不足，模型就更接近上一阶段参数。

---

### Step 7：讲 softmax-KL

可以这样说：

> 实际实现中，作者没有用普通 L2，而是对每个 LoRA 模块内部参数做 softmax，然后和上一阶段参数分布计算 KL。这样可以保护模块内部结构，而不是平等惩罚所有参数变化。

---

### Step 8：讲主实验

Table 2 说明：

```text
PESO 在三个 Amazon 数据集上整体最好。
```

Table 3 说明：

```text
PESO 同时照顾 dormant users 和 new users。
```

Figure 3 说明：

```text
λ 控制 stability 和 plasticity 的权衡。
```

---

## 23. 可以直接用于汇报的总结话术

这篇论文的核心观点是：

> LLM-based recommender 在真实系统中需要连续适应用户兴趣变化。已有 Single Evolving LoRA 虽然适应新数据快，但容易遗忘；Cumulative LoRA 虽然保留旧 adapter，但在推荐场景中过于僵硬，因为旧 adapter 中有用长期偏好和过时短期偏好纠缠在一起。PESO 通过维护一个单一演化 LoRA，并加入 softmax-KL proximal regularizer，让当前 adapter 在上一阶段 adapter 附近受控更新。理论上，它可以根据当前数据对不同参数方向的支持程度，自适应决定更接近当前最优解还是保留上一阶段参数；实验上，它在多个 Amazon 数据集和不同用户漂移模式下都取得更好的稳定性-可塑性平衡。

---

## 24. 最后记忆框架

如果只记住这篇论文的主线，可以记成：

```text
问题：
  推荐数据连续到来，用户偏好会漂移。

矛盾：
  既要适应新兴趣，又要保留长期偏好。

已有方法：
  Single Evolving LoRA: 太容易忘。
  Cumulative LoRA: 太僵硬，保留过时偏好。

关键观察：
  Table 1 证明 cumulative LoRA 在自然时间推荐场景中退化明显。

方法：
  PESO = 单一演化 LoRA + proximal regularizer。

实现：
  per-module softmax-KL 约束当前 LoRA 和上一阶段 LoRA。

理论：
  当前数据强支持的方向多更新，证据不足的方向多保留。

实验：
  PESO 在主结果、用户分组和正则消融中都表现更好。
```
