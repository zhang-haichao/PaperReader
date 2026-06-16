\# Function Vectors in Large Language Models 组会讲解稿



\## 1. 这篇论文想解决什么问题？



这篇论文研究的是 \*\*大语言模型做 in-context learning 时，内部到底是怎么表示“当前任务”的\*\*。



我们平时看到 LLM 可以通过几个例子临时学会一个任务，比如：



```text

hot:cold, big:small, happy:sad, fast:

```



模型会输出：



```text

slow

```



这里模型其实做了两件事：



1\. 先识别任务：这是“反义词任务”；

2\. 再执行任务：把 `fast` 映射成 `slow`。



这篇论文关心的不是模型为什么知道 `fast` 的反义词，而是：



> 模型内部有没有一个向量，表示“现在要执行反义词这个函数”？



作者把这个向量叫做 \*\*Function Vector\*\*，简称 \*\*FV\*\*。



可以把 FV 理解成模型内部的一个“任务开关”：



```text

Antonym FV        → 打开反义词模式

English-French FV → 打开英文转法文模式

Capitalize FV     → 打开首字母大写模式

```



也就是说，FV 不是直接存答案，而是触发模型执行某个任务。



\---



\## 2. 结合图 1：Function Vector 的整体直觉



论文图 1 给出了整篇文章最核心的直觉。



图 1 的意思是：



1\. 先给模型一些 ICL 示例，比如反义词任务或英文到西班牙文翻译任务；

2\. 从模型内部激活中提取出一个向量；

3\. 把这个向量插入到一个完全不同的新上下文里；

4\. 模型就会倾向于执行原来的任务。



比如从下面的反义词例子中提取 Antonym FV：



```text

arrive:depart, small:big, common:

```



然后把 Antonym FV 插入到：



```text

The word "fast" means

```



模型就可能输出：



```text

slow

```



这说明 FV 不是简单记住 prompt 格式，而是更像一个抽象的任务表示。



\---



\## 3. 结合图 2：最初的观察



作者一开始做了一个很简单的实验。



对于某个任务 $t$，比如反义词任务，收集很多 ICL prompt，然后记录模型在某一层最后一个 token 的 hidden state，并求平均：



$$

\\bar{h}\_{\\ell}^{t}

$$



然后把这个平均 hidden state 加到一个 zero-shot prompt 中。



比如不给任何例子，只输入：



```text

simple:

```



如果加入反义词任务的平均激活，模型有时会输出：



```text

complex

```



图 2 说明：



\* 红线表示加入普通平均 hidden state 的效果；

\* 绿色表示加入作者后面构造出的 Function Vector 的效果；

\* 绿色更强，说明 FV 是一种更干净、更有效的任务表示。



这个观察引出全文核心问题：



> 能不能找到模型内部真正负责传递任务信息的组件，然后从这些组件中提取更好的任务向量？



\---



\## 4. Transformer 中要分析什么？



作者重点分析 attention heads。



在 Transformer 中，第 $\\ell$ 层最后 token 的 hidden state 可以写成：



$$

h\_{\\ell}

========



h\_{\\ell-1}

\+

m\_{\\ell}

\+

\\sum\_{j \\leq J} a\_{\\ell j}

$$



其中：



\* $h\_{\\ell}$：第 $\\ell$ 层 hidden state；

\* $m\_{\\ell}$：MLP 输出；

\* $a\_{\\ell j}$：第 $\\ell$ 层第 $j$ 个 attention head 的输出；

\* $J$：每层 attention head 的数量。



作者关注 attention head，是因为 ICL 需要模型从前面的输入输出例子中搬运信息，而 attention head 正是负责不同 token 之间信息传递的组件。



\---



\## 5. ICL Prompt 的形式



对于任务 $t$，一个正常 ICL prompt 写成：



$$

p\_i^t =

\[(x\_{i1}, y\_{i1}), \\cdots, (x\_{iN}, y\_{iN}), x\_{iq}]

$$



含义是：



```text

前面有 N 个输入-输出示例

最后给一个 query input

模型需要预测对应的 y\_iq

```



比如反义词任务：



```text

hot:cold, big:small, happy:sad, fast:

```



这里：



\* $x$ 是输入词；

\* $y$ 是输出词；

\* 最后的 $x\_{iq}$ 是 `fast`；

\* 正确答案 $y\_{iq}$ 是 `slow`。



\---



\## 6. 方法核心：如何找到携带任务信息的 attention heads？



作者使用的是 \*\*causal mediation analysis\*\*，具体实现是 \*\*activation patching\*\*。



目标是找到：



> 哪些 attention heads 对模型正确执行 ICL 任务有因果作用？



注意，这里不是看相关性，而是直接做干预。



\---



\## 7. 第一步：计算每个 head 在任务上的平均激活



对于每个任务 $t$，每一层 $\\ell$，每个 head $j$，作者计算这个 head 在正常 ICL prompt 上的平均输出：



$$

\\bar{a}\_{\\ell j}^{t}

====================



\\frac{1}{|P\_t|}

\\sum\_{p\_i^t \\in P\_t}

a\_{\\ell j}(p\_i^t)

$$



这里：



\* $P\_t$ 是任务 $t$ 的正常 prompt 集合；

\* $a\_{\\ell j}(p\_i^t)$ 是模型处理 prompt $p\_i^t$ 时，第 $\\ell$ 层第 $j$ 个 head 的输出；

\* $\\bar{a}\_{\\ell j}^{t}$ 表示这个 head 在任务 $t$ 上的平均激活。



直观理解：



> 这个平均激活表示：当模型在做任务 $t$ 时，这个 head 通常传递什么信息。



这一步会对所有 layer、所有 attention heads 都做。



\---



\## 8. 第二步：构造 shuffled-label prompt



为了判断某个 head 是否真的重要，作者构造一种“坏 prompt”。



正常 prompt：



```text

hot:cold, big:small, happy:sad, fast:

```



打乱标签后：



```text

hot:small, big:sad, happy:cold, fast:

```



这时输入和输出之间没有稳定规律，模型很难判断任务是什么。



作者把这种 corrupted prompt 记为：



$$

\\tilde{p}\_i^t

$$



这样做的好处是：



> 如果在坏 prompt 中补回某个 head 的正常任务激活，模型正确答案概率上升，就说明这个 head 真的携带了任务信息。



\---



\## 9. 第三步：Activation Patching 和 CIE



在模型处理 corrupted prompt 时，作者把某个 head 的激活替换成它在正常任务中的平均激活：



$$

a\_{\\ell j} := \\bar{a}\_{\\ell j}^{t}

$$



然后观察正确答案 $y\_{iq}$ 的概率变化。



作者定义 \*\*CIE\*\*，即 causal indirect effect：



$$

CIE(a\_{\\ell j} \\mid \\tilde{p}\_i^t)

==================================



\## f(\\tilde{p}\*i^t \\mid a\*{\\ell j}:=\\bar{a}\*{\\ell j}^{t})\[y\*{iq}]



f(\\tilde{p}\*i^t)\[y\*{iq}]

$$



这个公式非常关键。



它的意思是：



```text

CIE = 替换某个 head 后，正确答案概率的提升量

```



如果 $CIE$ 很大，说明这个 head 对恢复正确答案很重要。



所以可以这样讲：



> 作者把每个 attention head 都单独“补回正确任务信息”试一遍，看哪个 head 最能让模型答对。



\---



\## 10. 第四步：计算 AIE，找通用关键 heads



单个 CIE 只是在一个任务、一个 prompt 上的效果。



作者想找的是对很多任务都重要的 heads，所以定义 \*\*AIE\*\*：



$$

AIE(a\_{\\ell j})

===============



\\frac{1}{|T|}

\\sum\_{t \\in T}

\\frac{1}{|\\tilde{P}\*t|}

\\sum\*{\\tilde{p}\_i^t \\in \\tilde{P}\*t}

CIE(a\*{\\ell j} \\mid \\tilde{p}\_i^t)

$$



含义是：



```text

对每个 head：

&#x20;   在多个任务上算 CIE

&#x20;   在多个 corrupted prompts 上算 CIE

&#x20;   最后求平均

```



$AIE$ 越高，说明这个 head 越像一个通用的“任务信息搬运 head”。



\---



\## 11. 结合图 3：哪些 heads 最重要？



论文图 3 展示了 GPT-J 中每个 attention head 的 AIE 分数。



图 3(a) 是一个 heatmap：



\* 横轴是 layer；

\* 纵轴是 head index；

\* 颜色越明显，说明 AIE 越高；

\* 粉色框标出了 top 10 heads。



作者发现这些高 AIE heads 主要集中在模型的早中层或中间层。



图 3(b) 展示这些 heads 在看哪些 token。



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



这非常合理。因为要判断任务是什么，模型必须观察输入和输出之间的关系，而输出标签是最重要的线索。



可以形象地说：



> 这些 heads 像是在课堂上认真看例题答案的学生，通过看输入对应的输出，总结老师在考什么任务。



\---



\## 12. Function Vector 如何构造？



找到高 AIE heads 后，把它们组成集合 $A$。



对于任务 $t$，Function Vector 定义为：



$$

v\_t

===



\\sum\_{a\_{\\ell j} \\in A}

\\bar{a}\_{\\ell j}^{t}

$$



也就是说：



```text

FV = 关键 attention heads 在该任务上的平均输出之和

```



注意：



\* FV 不是每一层一个；

\* FV 是一个任务向量；

\* 它由少数高因果作用的 attention heads 构成；

\* 它的维度和 hidden state 一样，所以可以直接加到模型内部表示上。



\---



\## 13. 如何使用 Function Vector？



得到 $v\_t$ 后，作者把它加到模型某一层 hidden state 上：



$$

h\_{\\ell} \\leftarrow h\_{\\ell} + v\_t

$$



然后看模型是否执行任务 $t$。



比如，输入：



```text

fast:

```



本来模型不知道要做什么。



如果在中间层加入 Antonym FV，模型就更可能输出：



```text

slow

```



所以 FV 的作用是：



```text

在模型内部注入一个任务信号

```



\---



\## 14. 结合图 4：为什么说 FV 是“任务触发器”？



图 4 展示了一个重要现象：



> 把 FV 插入不同层，效果不一样。



结果是：



\* 在早中层或中间层插入效果最好；

\* 在很靠后的层插入效果明显下降。



这说明 FV 不是简单地直接修改最终输出概率。



如果 FV 只是一个输出层 bias，那么越靠近最后一层应该越有效。

但实验不是这样。



因此作者认为：



> FV 更像是在中间层触发模型后续的非线性计算流程。



可以这样解释：



```text

FV 不是把答案直接塞给模型

而是在模型中间层按下“执行这个任务”的按钮

```



\---



\## 15. 实验结果简略说明



作者测试了多个模型，包括：



\* GPT-J 6B；

\* GPT-NeoX 20B；

\* Llama 2 7B / 13B / 70B。



任务包括 40 多个，正文主要展示 6 个：



\* Antonym；

\* Capitalize；

\* Country-Capital；

\* English-French；

\* Present-Past；

\* Singular-Plural。



核心结果是：



1\. 在 shuffled-label prompt 中，原本任务信息被破坏，加入 FV 后模型能恢复任务能力；

2\. 在 zero-shot prompt 中，没有任何示例，加入 FV 后模型也能执行任务；

3\. FV 对不同 prompt 模板和自然语言上下文也有一定迁移性。



一个代表性结果是：



| 模型          | Zero-shot baseline | 加 FV 后 |

| ----------- | -----------------: | -----: |

| GPT-J       |               5.5% |  57.5% |

| GPT-NeoX    |               6.7% |  57.1% |

| Llama 2 70B |               8.2% |  83.8% |



这个结果说明：



> Function Vector 确实能在没有示例的情况下触发模型执行对应任务。



\---



\## 16. 这篇论文的核心贡献



这篇论文最重要的贡献可以总结为三点。



第一，提出 \*\*Function Vector\*\*：



> LLM 内部存在表示任务或函数的紧凑向量。



第二，提出提取方法：



> 用 causal mediation analysis 找到少数关键 attention heads，再把它们的任务平均输出相加得到 FV。



第三，证明 FV 有因果作用：



> 把 FV 插入模型内部 hidden state，可以在 shuffled-label、zero-shot 和自然文本场景中触发对应任务。



\---



\## 17. 一句话总结



这篇论文说明：



> 大语言模型做 ICL 时，不只是表面上模仿 prompt，而是在内部形成了某种任务表示。这个表示可以被定位、提取，并作为 Function Vector 插入模型，从而触发模型执行对应函数。



可以用一句更形象的话收尾：



> Function Vector 就像模型内部的“函数调用指令”：从例子中提取出来，再插回模型，就能告诉模型“现在请执行这个任务”。



