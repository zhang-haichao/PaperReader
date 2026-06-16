
## 1. 开场：这篇论文在问什么问题？

大家好，我今天分享的论文是 **Function Vectors in Large Language Models**，发表于 ICLR 2024。

这篇论文关注的是大语言模型中的 **in-context learning**，也就是模型不更新参数，只通过 prompt 里的几个例子临时学会一个任务。

比如我们给模型：

```text
hot:cold, big:small, happy:sad, fast:
```

模型可能会输出：

```text
slow
```

这里模型其实做了两件事：

1. 先识别当前任务：这是一个“找反义词”的任务；
2. 再执行任务：把 `fast` 映射成 `slow`。

这篇论文真正关心的是第一步：

> 模型内部有没有一个东西，表示“当前要执行什么任务”？

作者的核心发现是：
**有。大语言模型内部存在一种紧凑的任务向量，作者称为 Function Vector，简称 FV。**

可以把 FV 理解成模型内部的“任务开关”：

```text
Antonym FV        → 打开反义词模式
English-French FV → 打开英文转法文模式
Capitalize FV     → 打开首字母大写模式
```

它不是直接存答案，而是触发模型去执行某个函数。

---

## 2. 图 1：Function Vector 的整体直觉

我们先看论文的图 1。

图 1 展示了整篇论文最核心的想法：

> 从 ICL 示例中提取一个任务向量，再把这个向量插入到新的上下文中，模型就会执行同样的任务。

比如，模型看到一组反义词示例：

```text
arrive:depart, small:big, common:
```

我们从模型内部激活中提取出一个 **Antonym FV**。

然后把这个 FV 插入到一个完全不同的自然语言上下文中：

```text
The word "fast" means
```

模型就可能输出：

```text
slow
```

这说明 FV 不是简单记住原来的 prompt 格式，而是更像一个抽象的任务表示。

---

## 3. 图 2：最初的观察

作者一开始做了一个很简单的实验。

对于某个任务 $t$，比如反义词任务，收集很多 ICL prompt，然后记录模型在某一层最后一个 token 的 hidden state，并求平均：

$$
\bar{h}_{\ell}^{t}
$$

然后把这个平均 hidden state 加到一个 zero-shot prompt 中。

比如不给任何示例，只输入：

```text
simple:
```

如果加入反义词任务的平均激活，模型有时会输出：

```text
complex
```

图 2 中红线表示直接加入平均 hidden state 的效果，绿色表示后面构造出的 Function Vector 的效果。

图 2 的重点是：

> 模型的中间层 hidden state 里确实包含某种任务信息；而作者提出的 FV 比普通 hidden state 平均更有效。

这个观察引出了全文的核心方法问题：

> 能不能更精确地找到模型内部哪些组件在搬运任务信息？

---

## 4. Transformer 中作者分析什么？

作者重点分析的是 **attention heads**。

在 Transformer 中，第 $\ell$ 层最后一个 token 的 hidden state 可以写成：

$$
h_{\ell}
========

h_{\ell-1}
+
m_{\ell}
+
\sum_{j=1}^{J} a_{\ell j}
$$

其中：

* $h_{\ell}$ 是第 $\ell$ 层的 hidden state；
* $m_{\ell}$ 是 MLP 输出；
* $a_{\ell j}$ 是第 $\ell$ 层第 $j$ 个 attention head 的输出；
* $J$ 是每层 attention heads 的数量。

为什么重点看 attention heads？

因为 ICL 需要模型从前面的示例中读取输入输出关系，而 attention head 正是 Transformer 中负责在不同 token 之间传递信息的组件。

所以作者的问题可以具体化为：

> 哪些 attention heads 在 ICL 中真正负责传递“任务是什么”的信息？

---

## 5. ICL Prompt 的形式

对于任务 $t$，一个正常 ICL prompt 可以写成：

$$
p_i^t
=====

[(x_{i1}, y_{i1}), \cdots, (x_{iN}, y_{iN}), x_{iq}]
$$

其中：

* $(x_{i1}, y_{i1}), \cdots, (x_{iN}, y_{iN})$ 是前面的输入输出示例；
* $x_{iq}$ 是最后的 query input；
* $y_{iq}$ 是模型应该预测的正确答案。

比如反义词任务：

```text
hot:cold, big:small, happy:sad, fast:
```

这里：

* 前面三对是示例；
* `fast` 是 query；
* 正确答案是 `slow`。

---

## 6. 方法核心：Activation Patching

作者使用的是 **causal mediation analysis**，具体实现是 **activation patching**。

这部分是论文方法的核心。

它的目标不是看相关性，而是直接做因果干预：

> 如果我们替换某个 attention head 的激活，模型正确答案概率上升了，那这个 head 就很可能真的携带任务信息。

整个过程可以分成四步。

---

## 7. 第一步：计算每个 head 的任务平均激活

对于每个任务 $t$，每一层 $\ell$，每个 head $j$，作者先在正常 ICL prompt 上计算这个 head 的平均输出：

$$
\bar{a}_{\ell j}^{t}
====================

\frac{1}{|P_t|}
\sum_{p_i^t \in P_t}
a_{\ell j}(p_i^t)
$$

这里：

* $P_t$ 是任务 $t$ 的正常 ICL prompt 集合；
* $a_{\ell j}(p_i^t)$ 表示模型处理 prompt $p_i^t$ 时，第 $\ell$ 层第 $j$ 个 head 的输出；
* $\bar{a}_{\ell j}^{t}$ 表示这个 head 在任务 $t$ 上的平均激活。

直观理解：

> 当模型在做任务 $t$ 时，这个 head 平均传递了什么信息？

这一步是对所有 layer、所有 attention heads 分别做的。

---

## 8. 第二步：构造 shuffled-label prompt

接下来，作者构造一种“坏 prompt”。

正常 prompt 是：

```text
hot:cold, big:small, happy:sad, fast:
```

打乱标签后变成：

```text
hot:small, big:sad, happy:cold, fast:
```

这时输入和输出之间已经没有稳定规律，模型很难判断任务是什么。

作者把这种 corrupted prompt 记为：

$$
\tilde{p}_i^t
$$

为什么要这样做？

因为如果 prompt 是正常的，模型本来就可能答对。
但如果 prompt 被打乱，模型答对就困难了。

这时如果我们补回某个 head 的正常任务激活，模型正确答案概率明显上升，就说明这个 head 对任务识别有因果作用。

---

## 9. 第三步：替换 head 激活并计算 CIE

在模型处理 corrupted prompt 的时候，作者把某个 head 的输出替换成正常任务下的平均激活：

$$
a_{\ell j} := \bar{a}_{\ell j}^{t}
$$

然后观察正确答案 $y_{iq}$ 的概率是否上升。

作者定义 **CIE**，也就是 causal indirect effect：

$$
\begin{aligned}
CIE(a_{\ell j} \mid \tilde{p}_i^t)
&=
f(\tilde{p}_i^t \mid a_{\ell j} := \bar{a}_{\ell j}^{t})[y_{iq}]
-
f(\tilde{p}_i^t)[y_{iq}]
\end{aligned}
$$

这个公式可以直接理解为：

```text
CIE = 替换某个 head 后，正确答案概率增加了多少
```

第一项：

$$
f(\tilde{p}*i^t \mid a*{\ell j} := \bar{a}*{\ell j}^{t})[y*{iq}]
$$

表示在 corrupted prompt 上，把某个 head 换成正常任务激活后，模型给正确答案的概率。

第二项：

$$
f(\tilde{p}*i^t)[y*{iq}]
$$

表示不做替换时，模型给正确答案的概率。

两者相减，就是这个 head 的因果贡献。

如果 $CIE$ 很大，说明这个 head 很重要。
如果 $CIE$ 接近 0，说明替换它没有什么帮助。

---

## 10. 第四步：计算 AIE，找通用关键 heads

单个 $CIE$ 只是在一个任务、一个 prompt 上的效果。

作者想找的是对很多 ICL 任务都重要的 heads，所以进一步定义 **AIE**：

$$
\begin{aligned}
AIE(a_{\ell j})
&=
\frac{1}{|T|}
\sum_{t \in T}
\frac{1}{|\tilde{P}_t|}
\sum_{\tilde{p}_i^t \in \tilde{P}_t}
CIE(a_{\ell j} \mid \tilde{p}_i^t)
\end{aligned}
$$

这里：

* $T$ 是任务集合；
* $\tilde{P}_t$ 是任务 $t$ 的 corrupted prompt 集合；
* $AIE(a_{\ell j})$ 表示这个 head 在多个任务、多个 corrupted prompts 上的平均因果作用。

通俗地说：

```text
对每个 attention head：
    在很多任务上测试
    在很多 corrupted prompts 上测试
    看替换它之后正确答案概率平均提升多少
```

$AIE$ 越高，说明这个 head 越像一个通用的“任务信息搬运 head”。

---

## 11. 图 3：关键 heads 在哪里？它们看什么？

论文图 3 展示了 GPT-J 中每个 attention head 的 AIE。

图 3(a) 是一个 heatmap：

* 横轴是 layer；
* 纵轴是 head index；
* 颜色表示 AIE 大小；
* 粉色框标出 AIE 最高的 top heads。

作者发现：

> 高 AIE heads 主要集中在模型的早中层或中间层。

图 3(b) 展示这些 top heads 在 prompt 中关注哪些 token。

结论是：

> 这些 heads 主要关注 ICL 示例中的输出 token，也就是 label token。

比如：

```text
eggs:oeufs, sad:triste, sugar:sucre, read:
```

关键 heads 会重点关注：

```text
oeufs, triste, sucre
```

这非常合理。因为要判断任务是什么，模型必须观察输入和输出之间的关系，而输出标签是最直接的线索。

可以形象地说：

> 这些 attention heads 像是在认真看例题答案的学生。
> 它们通过观察每个输入对应的输出，总结“这道题到底要我做什么”。

---

## 12. Function Vector 如何构造？

找到高 AIE heads 后，把这些 heads 组成集合 $A$。

对于任务 $t$，作者把这些关键 heads 在任务 $t$ 上的平均输出加起来，得到 Function Vector：

$$
v_t
===

\sum_{a_{\ell j} \in A}
\bar{a}_{\ell j}^{t}
$$

这就是全文最核心的公式。

它的含义是：

```text
Function Vector
= 关键 attention heads 在该任务上的平均输出之和
```

注意几个点：

1. FV 不是每一层一个；
2. FV 是一个任务向量；
3. 它由少数高因果作用的 attention heads 构成；
4. 它的维度和 hidden state 一样，所以可以直接加到模型内部表示上。

---

## 13. 如何使用 Function Vector？

得到 $v_t$ 后，作者把它加到模型某一层 hidden state 上：

$$
h_{\ell} \leftarrow h_{\ell} + v_t
$$

这样相当于在模型内部注入一个任务信号。

比如我们得到 Antonym FV 后，只输入：

```text
fast:
```

本来模型不知道要做什么。
但如果在中间层加入 Antonym FV，模型就更可能输出：

```text
slow
```

所以 FV 的作用不是直接给答案，而是告诉模型：

```text
现在请按照这个任务规则处理输入
```

---

## 14. 图 4：为什么说 FV 是任务触发器？

图 4 展示的是：把同一个 FV 插入不同层，模型 zero-shot 执行任务的准确率如何变化。

结果很关键：

> 在早中层或中间层插入 FV 效果最好；在靠后的层插入，效果明显下降。

这说明 FV 不是简单的输出层 bias。

如果 FV 只是直接提高某些答案 token 的概率，那么它越靠近输出层应该越有效。
但实验发现不是这样。

因此作者认为：

> FV 更像是在中间层触发模型后续的非线性计算流程。

可以用一句话概括：

```text
FV 不是把答案塞进模型嘴里，
而是在模型脑子中间按下“执行这个任务”的按钮。
```

---

## 15. 实验结果简略说明

实验部分我简单讲，不展开太多。

作者测试了多个模型：

* GPT-J 6B；
* GPT-NeoX 20B；
* Llama 2 7B、13B、70B。

任务包括 40 多个，正文主要展示 6 个：

* Antonym；
* Capitalize；
* Country-Capital；
* English-French；
* Present-Past；
* Singular-Plural。

核心实验场景有两个。

第一个是 **shuffled-label**：
prompt 中有示例，但标签被打乱，模型本来很难识别任务。

第二个是 **zero-shot**：
没有任何示例，只给 query，然后插入 FV。

表 2 的代表性结果是：

| 模型          | Zero-shot baseline | 加 FV 后 |
| ----------- | -----------------: | -----: |
| GPT-J       |               5.5% |  57.5% |
| GPT-NeoX    |               6.7% |  57.1% |
| Llama 2 70B |               8.2% |  83.8% |

这说明：

> 即使没有任何 ICL 示例，只要插入 FV，模型也能在相当程度上执行对应任务。

作者还测试了不同 prompt 模板和自然语言上下文，发现 FV 仍然有迁移效果。

---

## 16. 这篇论文的核心贡献

这篇论文的贡献可以总结成三点。

第一，提出了 **Function Vector** 这个概念：

> LLM 内部存在表示任务或函数的紧凑向量。

第二，提出了提取 FV 的方法：

> 用 causal mediation analysis 找到少数关键 attention heads，再把它们的任务平均输出相加得到 FV。

第三，证明 FV 有因果作用：

> 把 FV 插入模型 hidden state，可以在 shuffled-label、zero-shot 和自然文本场景中触发对应任务。

---

## 17. 局限性

这篇论文也有一些局限。

第一，任务相对简单。
主要是词级或短语级任务，比如反义词、翻译、单复数、国家到首都等。复杂推理任务是否也有清晰 FV，还需要进一步研究。

第二，FV 不是 ICL 的完整解释。
它解释的是任务表示和任务触发的一部分机制，但还不能解释所有 ICL 行为。

第三，部分实验是在模型本来可以通过 10-shot ICL 做对的样本上评估的。
所以 FV 更像是在触发模型已有能力，而不是创造模型不会的新能力。

---

## 18. 最后总结

这篇论文可以用三句话总结：

1. 大语言模型在做 ICL 时，内部会形成表示任务的 Function Vector。
2. Function Vector 可以通过少数具有高因果作用的 attention heads 提取出来。
3. 把 Function Vector 插入模型中，可以在 zero-shot、shuffled-label 和自然文本场景中触发模型执行对应任务。

最后可以用一句形象的话收尾：

> Function Vector 就像大语言模型内部的“函数调用指令”：
> 从例子中提取出来，再插回模型，就能告诉模型“现在请执行这个任务”。
