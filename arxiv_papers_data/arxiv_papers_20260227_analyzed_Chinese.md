# 20260227
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 软集理论及其扩展的动态综述 [PDF](https://arxiv.org/pdf/2602.21268), [HTML](https://arxiv.org/abs/2602.21268)
### Authors
Takaaki Fujita,Florentin Smarandache
### Background
软集理论提供了一种直接框架，通过将每个属性（参数）映射为给定宇宙的一个子集，从而以结构化的方式表征不确定性。自上世纪以来，该理论扩展到众多变体，包括超软集、超超软集、树软集、双极软集和动态软集，并与拓扑学和偏导集理论等多个领域建立了联系。
### Innovation
论文提供了一篇关于软集理论及其扩展的综述，涵盖了核心定义、代表性构造以及当前发展的关键方向，对多个变体进行了详细说明。
### Conclusion
文章以综述的形式展示了软集理论及其主要扩展，并突出了该领域的核心定义、典型构造和最新发展方向，强调了该领域在未来可能的研究方向。
## 2. `cs.AI` - 复合人工智能系统中聚合的功能与限制 [PDF](https://arxiv.org/pdf/2602.21556), [HTML](https://arxiv.org/abs/2602.21556)
### Authors
Nivasini Ananthakrishnan,Meena Jagadeesan
### Background
在设计复合人工智能系统时，通常采用查询多个相同模型的副本并聚合其响应以生成综合输出的方法。尽管这些模型是异质的，但研究提出这种聚合方式是否能够解锁比单一模型更多的输出结果。本文利用主从框架模型来研究聚合方式的功能与限制。
### Innovation
本文在主从框架内调查聚合的功能与限制，发现聚合通过可行性扩展、支持扩展和绑定集收缩三种自然机制增加了可以由系统设计师引发的输出结果集。不仅如此，本文还证明任何聚合操作若要实现扩展输出的效果，必须实现这些机制，并且加强后的这些机制构成了充分必要条件。最后，通过引入LLM在玩具生成任务中的实证例子展示了上述发现。
### Conclusion
本文结果为理解何时复合人工智能系统可以克服模型能力和提示工程的限制提供了步骤性的进展。
## 3. `cs.AI` - ARLArena: 统一的稳定人工智能强化学习框架 [PDF](https://arxiv.org/pdf/2602.21534), [HTML](https://arxiv.org/abs/2602.21534)
### Authors
Xiaoxuan Wang,Han Zhang,Haixin Wang,Yidan Shi,Ruoyan Li,Kaiqiao Han,Chenyi Tong,Haoran Deng,Renliang Sun,Alexander Taylor,Yanqiao Zhu,Jason Cong,Yizhou Sun,Wei Wang
### Background
代理强化学习（ARL）作为一种训练代理解决复杂、多步骤交互任务的有前景的方法，已经迅速引起了关注。尽管早期结果令人鼓舞，但ARL仍然高度不稳定，经常导致训练崩溃。这种不稳定性限制了其在更大环境和更长交互时间范围中的扩展，同时也限制了对算法设计选择系统的探索。
### Innovation
本文首先提出了ARLArena，一种稳定训练食谱和系统分析框架，通过在可控且可重复的环境中考察训练稳定性。通过将策略梯度分解为四个核心设计维度，并评估每个维度的性能和稳定性，ARLArena 提供了统一的ARL视角，并提出了SAMPO，一种稳定的代理策略优化方法，旨在缓解ARL中的主要不稳定来源。实证研究表明，SAMPO在各种代理任务中实现了稳定且强大的性能。
### Conclusion
这项研究提供了统一的策略梯度视角用于ARL，并为构建稳定且可重复的大规模语言模型（LLM）代理训练管道提供了实际指导。
## 4. `cs.AI` - 提示架构决定推理质量：汽车清洗问题上的变量隔离研究 [PDF](https://arxiv.org/pdf/2602.21814), [HTML](https://arxiv.org/abs/2602.21814)
### Authors
Heejin Jo
### Background
大型语言模型在解决包含隐式物理约束推理的‘汽车清洗问题’时经常表现不佳。该研究表明，在生产系统中，不同的提示架构层对于正确推理至关重要。本研究通过对Claude 3.5 Sonnet进行参数控制，探讨了哪些提示架构层能够提高推理质量。
### Innovation
本研究通过变量隔离实验，具体检验了不同提示架构层对于提高大型语言模型推理准确率的影响。实验结果显示，STAR（情境-任务-行动-结果）推理框架能够显著提高模型的推理准确率；加入用户资料上下文和检索增强代理（RAG）能够进一步提升模型的推理效果。
### Conclusion
研究表明，结构化推理框架对于隐式约束推理任务而言更为重要，相较于增加上下文信息，明确目标表达对推理质量的影响更大。
## 5. `cs.AI` - 超越拒绝：探求语义敏感信息的主动自我纠正极限 [PDF](https://arxiv.org/pdf/2602.21496), [HTML](https://arxiv.org/abs/2602.21496)
### Authors
Umid Suleymanov,Zaur Rajabov,Emil Mirzazada,Murat Kantarcioglu
### Background
结构化的个人身份信息（PII）已经有成熟的防御措施，但是大型语言模型（LLMs）带来了一个新的威胁：语义敏感信息（SemSI），这种信息可以让模型推导出敏感的身份属性、生成损害声誉的内容或产生有可能是错误的信息。LLMs的自我调节能力如何在不牺牲实用性的同时管理这些复杂的、依赖于上下文的信息泄露还没有得到证实。
### Innovation
提出了SemSIEdit，这是一种推理时框架，其中一个人工智能‘编辑’迭代地批判和修改敏感部分，以保持叙述的连贯性，而不是简单地拒绝回答。这项分析揭示了一个隐私-实用性帕累托前沿，其中这种人工智能编辑减少了所有三个SemSI类别中高达34.6%的信息泄露，但同时使得实用性下降9.8%。此外，还发现了一个规模依赖的安全偏差：大型推理模型（如GPT-5）通过增加细微差别来实现安全性，而容量受限模型则通过删除文本进行破坏性的截断。最后，揭示了推理悖论：推理时的推理增加了基本风险，但同时也增强了防御机制进行安全重写的能力。
### Conclusion
研究显示了隐私与功效之间的帕累托前沿，并发现了大型推理模型与容量受限制模型在实现安全方面的差异性策略，同时也提出了推理时推理如何既增加了敏感性的推断风险，又为安全重写提供了机会。
## 6. `cs.AI` - 一个用于地球科学数据档案自主发现的分层次多代理系统 [PDF](https://arxiv.org/pdf/2602.21351), [HTML](https://arxiv.org/abs/2602.21351)
### Authors
Dmitrii Pantiukhin,Ivan Kuznetsov,Boris Shapkin,Antonia Anna Jost,Thomas Jung,Nikolay Koldunov
### Background
地球科学中的数据积累快速增长，造成可扩展性挑战。尽管像PANGAEA这样的存储库拥有庞大的数据集集合，但引用统计数据显示，仍有大量数据未得到充分利用，影响了数据的重用性。
### Innovation
本文介绍了一种分层次多代理框架PANGAEA-GPT，用于自主数据发现和分析。该架构具有集中式的监督-工人拓扑结构，具备严格的数据类型感知路由、沙箱中的确定性代码执行以及通过执行反馈进行自我纠正的功能，使代理能够诊断和解决运行时错误。通过跨物理海洋学和生态学的案例场景，展示了该系统在最少的人类干预下执行复杂、多步骤工作流的能力。
### Conclusion
该框架提供了一种通过协调代理工作流进行查询和分析异构存储库数据的方法。
## 7. `cs.AI` - 提炼和对齐分解以增强声明验证 [PDF](https://arxiv.org/pdf/2602.21857), [HTML](https://arxiv.org/abs/2602.21857)
### Authors
Jabez Magomere,Elena Kochkina,Samuel Mensah,Simerjot Kaur,Fernando Acero,Arturo Oncevay,Charese H. Smiley,Xiaomo Liu,Manuela Veloso
### Background
复杂的声明验证需要将句子分解为可验证的子声明，但现有的方法难以同时提高分解质量和验证性能。此前的方法在优化分解质量的同时往往会牺牲验证性能，反之亦然。
### Innovation
本文提出了一种基于强化学习（RL）的方法，通过Group Relative Policy Optimization (GRPO)联合优化分解质量和验证器对齐。该方法结合了结构化的序列推理、教师提炼的示例的监督微调以及多目标奖励平衡机制，以同时满足合规性、验证器对齐和分解质量的需求。实验表明，通过该方法训练的8B预测分解器在下游验证性能方面提升了显著效果。
### Conclusion
本文的框架能够在确保验证准确性的前提下，充分利用分解的质量，使得较小的模型也能够实现最佳的声明验证效果。同时，人工评估得出生成的子声明质量也很高。
## 8. `cs.AI` - fEDM+: 一种基于风险的模糊伦理决策框架，具备原则级别可解释性和多元验证 [PDF](https://arxiv.org/pdf/2602.21746), [HTML](https://arxiv.org/abs/2602.21746)
### Authors
Abeer Dyoub,Francesca A. Lisi
### Background
在之前的著作中，我们提出了一种基于模糊逻辑的模糊伦理决策框架（fEDM），它是一种风险导向的伦理推理架构。原始模型结合了模糊伦理风险评估模块（fERA）和伦理决策规则，通过模糊Petri网（FPNs）进行形式结构验证，并与单一的规范性参考进行验证。尽管这种方法确保了形式上的正确性和决策的一致性，但它未能完全解决两个关键挑战：伦理决策的合理可解释性和在伦理多元主义下的鲁棒性。
### Innovation
我们在此论文中在fEDM的基础上进行了两项主要改进。首先，我们引入了一个解释和追踪模块（ETM），该模块将每个伦理决策规则明确链接到背景中的道德原则，并为每推荐的动作计算一个加权原则贡献配置文件。这允许透明且可审核的解释，不仅暴露了做出了什么决定，还解释了这样的决策基于哪些原则。其次，我们将单一参考验证替换为基于多元利益相关者参考的语义验证框架，每个参考都编码了不同的原则优先级和风险容忍度。这使框架能够在不阻碍进一步讨论的同时，正式地代表原则间的合理分歧，从而提高鲁棒性和情境敏感性。
### Conclusion
由此扩展的fEDM被称为fEDM+，它保留了形式可验证性的同时，实现了增强的可解释性和利益相关者意识验证，使其适用于伦理敏感的AI系统的监督和治理层。
## 9. `cs.AI` - ProactiveMobile:提升移动设备上主动智能的综合性基准 [PDF](https://arxiv.org/pdf/2602.21858), [HTML](https://arxiv.org/abs/2602.21858)
### Authors
Dezhi Kong,Zhengzhao Feng,Qiliang Liang,Hao Wang,Haofei Sun,Changpeng Yang,Yang Li,Peng Zhou,Shuai Nie,Hongzhen Wang,Linfeng Zhou,Hao Jia,Jiaming Xu,Runyu Shi,Ying Huang
### Background
目前，多模态大型语言模型（MLLMs）在移动代理开发中取得了显著进展，但它们的能力主要集中在反应性范式上，即仅执行用户的显式命令。新兴的主动智能范式，即代理能够自主预测需求并主动发起行动，是移动代理的下一个前沿领域。然而，其发展受到了缺乏能够解决现实复杂性并支持客观、可执行评估基准的瓶颈。
### Innovation
该论文引入了ProactiveMobile，这是一个全面的基准，旨在系统地促进这一领域中的研究。该基准将主动任务定义为通过设备上四种类型的上下文信号推断潜在用户意图并从63个API的功能池中生成可执行的功能序列。ProactiveMobile涵盖了14种真实世界复杂性的场景，通过多答案注释来反映现实世界的复杂性。此外，基准还经过了30名专家的最终审查，以确保其质量。
### Conclusion
通过对ProactiveMobile的实验表明，微调后的Qwen2.5-VL-7B-Instruct在成功率上达到了19.15%，这优于o1（15.71%）和GPT-5（7.39%）。这一结果表明，主动性是一个在当前MLLM中普遍缺乏的重要能力，但它是可学习的，这就凸显了所提出的基准在评估主动性上的重要性。
## 10. `cs.AI` - ASIR 勇气模型：人类与人工智能系统中真理过渡的相动态框架 [PDF](https://arxiv.org/pdf/2602.21745), [HTML](https://arxiv.org/abs/2602.21745)
### Authors
Hyo Jin Kim(Jinple)
### Background
该研究背景是一种新的真理披露框架——ASIR（唤醒共享智能关系）勇气模型引入了这一概念，将真理披露过程视作状态转换而非个性特质。模型通过不等式 ?(?lambda(1+?gamma)+?psi > ?theta+?phi?) 来刻画抑制和表达之间的转变，其中各个参数分别代表基础开放性、关系放大、累积内心压力和转换成本。此外，该模型原本为人类在不平等风险下的真相披露制定，但其相动态架构也适用于受政策约束和对齐过滤器影响的AI系统。
### Innovation
ASIR 勇气模型的创新之处在于它通过相动态理论提供了一种全新的框架来理解真理过渡过程，适用于人类和AI系统。模型不仅能够解释人类在压力下的沉默现象，还能解释AI在偏好驱动下产生的扭曲效果。此外，模型还通过反馈扩展来模拟反复交互中的路径依赖和分歧效应，并认为真理披露的变化是内部相互作用的几何后果，而非单纯的人或AI意图。
### Conclusion
ASIR 勇气模型提供了一种统一的结构解释，旨在明确并优化人类和AI系统中的真理披露过程。无论是人类还是AI系统，模型都能够整合勇气和对齐概念，提供一种在风险下进行真理披露的形式化视角。该模型能够帮助理解并管理复杂系统中的真理传达问题，这对人工智能的伦理设计和应用具有重要意义。
## 11. `cs.LG` - 因果解码用于幻觉抗性多模态大型语言模型 [PDF](https://arxiv.org/pdf/2602.21441), [HTML](https://arxiv.org/abs/2602.21441)
### Authors
Shiwei Tan,Hengyi Wang,Weiyi Qin,Qi Xu,Zhigang Hua,Hao Wang
### Background
现有的多模态大型语言模型（MLLMs）在视觉语言任务中可以生成详细的响应，但在处理时容易出现对象幻觉，即生成图像中不存在的对象，这降低了模型在实际应用中的可靠性。尽管已有研究尝试通过启发式的惩罚措施、事后修正或通用解码调整来应对这一问题，但这些方法并未直接干预导致对象幻觉的机制，因而效果有限。
### Innovation
本文提出了一种因果解码框架，通过在生成过程中应用有针对性的因果干预措施，来遏制引起幻觉的虚假依赖。通过调整解码动态结构，减少虚假对象标记的同时保持描述质量。
### Conclusion
通过在多项基准测试中的应用，本文的框架显著降低了对象幻觉的发生率，并达到了最先进的忠实度，而不会损害整体输出质量。
## 12. `cs.LG` - 当学习有害：固定极点RNN用于在线实时训练 [PDF](https://arxiv.org/pdf/2602.21454), [HTML](https://arxiv.org/abs/2602.21454)
### Authors
Alexander Morgan,Ummay Sumaya Khan,Lingjia Liu,Lizhong Zheng
### Background
回声状态网络（ESNs）通过固定循环动态并仅训练线性读取，克服了通过时间反向传播（BPTT）学习所有参数（包括极点位置）带来的巨大计算负担，尤其是在数据有限的应用场景中。尽管RNN可以解释为离散时间状态空间模型，其中状态演化对应于由前馈权重和反馈极点双重控制的无限冲激响应（IIR）滤波操作，但在数据约束和实时学习场景中，学习反馈极点并未提供实际好处。
### Innovation
本文从理论和实证两方面探讨了，在数据受限和实时学习场景中，为何学习反馈极点无法带来实质性的好处。研究发现，极点学习使得基于梯度的方法优化权重问题变得高度非凸，需要更多的训练样本和迭代次数才能收敛到有意义的解；对于复数数据，梯度下降经常出现漫长的平台期，先进的优化器提供有限的改进；固定极点架构即使在数据有限的情况下也能诱导稳定的且条件良好的状态表示。
### Conclusion
固定极点网络在训练复杂性和性能方面均表现优异，适合用于在线实时任务，初步结果证明，它们比学习极点的RNN更优。
## 13. `cs.LG` - Irregular Multivariate Time Series Forecasting的递归多尺度表示学习 [PDF](https://arxiv.org/pdf/2602.21498), [HTML](https://arxiv.org/abs/2602.21498)
### Authors
Boyuan Li,Zhen Liu,Yicheng Luo,Qianli Ma
### Background
不规则多变量时间序列（IMTS）具有不均匀的时间戳间隔，携帯有宝贵且信息丰富的采样模式，可用于学习时间和变量依赖关系。此外，IMTS还可能在多个时间尺度上表现出多种依赖关系。然而，许多现有的多尺度IMTS方法会采用重采样来获得粗系列，这可能导致原始时间戳的改变，破坏了采样模式的信息。
### Innovation
本文提出了ReIMTS，一种递归多尺度建模方法，用于不规则多变量时间序列预测。ReIMTS不进行重采样，而是保持时间戳不变，并递归地将每个样本分为时间间隔逐渐缩短的子样本。基于这些长到短的子样本中的原始采样时间戳，提出了一种不规则性感知表示融合机制，用于捕捉全局到局部依赖关系，以实现准确的预测。
### Conclusion
大量实验表明，ReIMTS在不同模型和现实世界数据集上的预测任务中平均性能提高了27.1％。我们的代码可以在这个网址找到。
## 14. `cs.LG` - 三模态掩码扩散模型的设计空间 [PDF](https://arxiv.org/pdf/2602.21472), [HTML](https://arxiv.org/abs/2602.21472)
### Authors
Louis Bethune,Victor Turrisi,Bruno Kacper Mlodozeniec,Pau Rodriguez Lopez,Lokesh Boominathan,Nikhil Bhendawade,Amitis Shidani,Joris Pelemans,Theo X. Olausson,Devon Hjelm,Paul Dixon,Joao Monteiro,Pierre Ablin,Vishnu Banna,Arno Blaas,Nick Henderson,Kari Noriy,Dan Busbridge,Josh Susskind,Marco Cuturi,Irina Belousova,Luca Zappella,Russ Webb,Jason Ramapuram
### Background
离散扩散模型已成为自回归语言模型的强有力替代方案，最近的研究工作中使用并微调了一种基础的单模态模型来生成双模态内容。不同以往的方法，本文引入了第一个从头开始预训练的三模态掩码扩散模型，它同时使用了文本、图文和音文数据。本文系统性地分析了多模态的扩展规律、模态混比例、噪声调度及批量大小的影响，提供了优化的推理采样参数。
### Innovation
本文的主要创新在于提出了全新的从头训练的第一种三模态掩码扩散模型，它能够在文本、图文和音文数据上进行预训练，并通过系统性分析优化了模型的各项参数，尤其是提出的基于随机微分方程的重参数化方法，使得模型的批量大小不再需要额外调优。此外，本文还展示了该模型在参数量和数据量上达到迄今为止最大规模的研究工作，提供了解多模态环境下的扩散模型放大行为的洞见。
### Conclusion
本文成功构建了大规模预训练的三模态扩散模型，系统性地研究了多模态扩散模型的扩展行为，展示了在文本生成、图文转换和语音合成任务上的良好表现。未来的工作可以通过进一步优化模型结构和参数来提高模型效率和效果。
## 15. `cs.LG` - GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning [PDF](https://arxiv.org/pdf/2602.21492), [HTML](https://arxiv.org/abs/2602.21492)
### Authors
Ningyuan Yang,Weihua Du,Weiwei Sun,Sean Welleck,Yiming Yang
### Background
强化学习（RL）已成为大语言模型（LLMs）后训练的核心范式，但其性能高度依赖训练问题的质量。这种依赖性源自于RL的非稳定性：回放由不断进化的策略生成，学习受到探索与奖励反馈的影响，而与固定轨迹的监督微调（SFT）不同。因此，以往的工作通常依赖于人工挑选或简单的启发式过滤（例如准确性），它们可能包含错误或低效的问题。
### Innovation
我们提出了一种名为GradAlign的梯度对齐数据选择方法，用于LLM的强化学习。该方法利用一个小规模的可信验证集来筛选出与验证梯度方向相匹配的训练问题，从而创建一个自适应的课程。GradAlign在可靠性差的奖励信号、分布不平衡和低效的训练语料库这三个具有挑战性的数据情况下，表现出比现有基线更好的性能，强调了方向梯度信号在导航非平稳策略优化中的重要性，从而提高了训练的稳定性和最终性能。
### Conclusion
GradAlign通过方向梯度信号在非平稳策略优化中引导导航，能够更稳定地进行训练并提高最终性能。我们在三个具有挑战性的数据环境中评估了GradAlign，结果表明它优于现有的基线方法，证明了方向梯度信号在LLM强化学习中的重要性。此外，我们发布了GradAlign的实现。
## 16. `cs.LG` - D-Flow SGLD: 在Flow Matching条件下针对科学逆问题的源空间后验采样 [PDF](https://arxiv.org/pdf/2602.21469), [HTML](https://arxiv.org/abs/2602.21469)
### Authors
Meet Hemant Parikh,Yaqin Chen,Jian-Xun Wang
### Background
数据同化和科学逆问题涉及从稀疏且噪声的数据中重构高维物理状态，理想情况下具有知晓不确定性的后验样本，同时保持学习到的先验和物理法则的一致性。尽管扩散模型在无训练条件下的条件生成方面已经得到良好发展，但对于Flow Matching（FM）先验的无训练条件生成和后验采样策略，在科学基准测试中仍然相对较少研究，特别是在评估重建精度时，必须超越测量误差的一些指标。
### Innovation
本文探讨了在FM先验下的无训练条件生成方法，主要通过测量信息注入的位置进行组织：(i) 指导传输动力学，使用似然信息扰动采样轨迹；(ii) 源分布推理，同时固定学习到的传输过程，进行后验推理。在此基础上，本文提出了D-Flow SGLD方法，这是一种源空间后验采样方法，结合可微源推理和预条件化的随机梯度拉普拉斯动态，能够在不重新训练先验或修改学习到的FM动力学的情况下，进行后验探索。该方法在类型层次的问题：2D玩具后验、混沌Kuramoto-Sivashinsky轨迹和壁界湍流重建中进行了基准测试。
### Conclusion
本文在不同场景下量化了测量集成、后验多样性以及物理学/统计学保真度之间的权衡，并证明了D-Flow SGLD作为与FM兼容的科学逆问题后验采样器的实用性。
## 17. `cs.LG` - MINAR：神经算法推理的机制解释 [PDF](https://arxiv.org/pdf/2602.21442), [HTML](https://arxiv.org/abs/2602.21442)
### Authors
Jesse He,Helen Jenne,Max Vargas,Davis Brown,Gal Mishne,Yusu Wang,Henry Kvinge
### Background
近年来，神经算法推理（NAR）领域研究了Graph Neural Networks（GNNs）模仿经典算法如Bellman-Ford的能力，这一现象称为算法对齐。与此同时，大型语言模型（LLMs）的最新进展引发了机制解释的研究，该研究旨在识别执行特定计算的具体模型组件，如电路。这项工作正是在这种背景下进行的。
### Innovation
本文提出了一个高效的电路发现工具箱MINAR，它适应机制解释中的归因补丁方法来应用于GNN环境。结果表明，MINAR能够从训练算法任务的GNN中恢复忠实的神经电路。
### Conclusion
我们的研究揭示了训练过程中的电路形成和修剪过程，并提供了关于GNN在并行执行多个任务时重用相关任务电路组件的新见解。相关代码可在[此处](this https URL)找到。
## 18. `cs.LG` - 带有张量球谐函数的渐进快速Clebsch-Gordan张量乘积 [PDF](https://arxiv.org/pdf/2602.21466), [HTML](https://arxiv.org/abs/2602.21466)
### Authors
YuQing Xie,Ameya Daigavane,Mit Kotak,Tess Smidt
### Background
E(3)-对称的神经网络已被证实有效应用于广泛的3D建模任务。其中，张量积是一种基本操作，允许不同特征类型的交互，但由于其计算复杂度极高，已经有很多研究致力于加速这一操作。最近的研究表明，大多数加速方法实际上是对表达性的减少而非真正的算法改进。一种改良后的Gaunt张量积可以提供实质性的渐进加速，但存在不完整的问题，无法涵盖所有交互。
### Innovation
本文首次提供了一个完整的算法，真正提供了Clebsch-Gordan张量乘积的渐进加速益处。算法将全Clebsch-Gordan张量乘积的运行时复杂度从O(L^6)降低到了O(L^4 log^2 L)，接近下界O(L^4)。文章还展示了如何通过推广基于快速傅里叶变换的卷积自然地得到先前提出的Gaunt张量积，并通过从标量信号推广到不可约值信号来解决反对称性问题，从而引入张量球谐函数。最后，证明了张量谐波的广义Gaunt公式，并说明只需要向量值信号即可恢复Gaunt张量积的缺失交互。
### Conclusion
本文提出的算法实现了真正意义上的渐进加速，通过引入张量球谐函数解决了反对称性问题，并通过推广技术提高了Clebsch-Gordan张量乘积的效率。
## 19. `cs.LG` - 训练数据质量对分类器性能的影响 [PDF](https://arxiv.org/pdf/2602.21462), [HTML](https://arxiv.org/abs/2602.21462)
### Authors
Alan F. Karr,Regina Ruane
### Background
本文描述了广泛的数值实验，评估和量化了分类器性能如何依赖于训练数据的质量。在元基因组学组装短DNA片段为“contigs”的科学背景下，研究了通过多种机制降低训练数据质量对四类分类器——贝叶斯分类器、神经网络、分区模型和随机森林——的影响。研究不仅考虑了每个分类器个体表现，还考虑了分类器之间的一致性。
### Innovation
研究发现所有四种分类器在数据分析与训练数据逐渐脱节时，都表现出一致性错误的特性，即随着训练数据质量的下降，分类器从正确变为偶然正确，原因是它们出错的方式相同。这种方法揭示了空间异质性：训练数据与分析数据差距越大，分类器决策越退化，边界变得不那么密集，一致性增加。
### Conclusion
研究结果表明，高质量的训练数据对提升分类器性能至关重要，尤其是在面对数据分布存在显著差异时，分类器的一致性表现尤为值得注意。因此，精准的动态调整训练数据的质量，对于提高分类器在实际应用场景中的性能具有重要意义。
## 20. `cs.LG` - 通过向量符号架构的几何先验构建可泛化的世界模型 [PDF](https://arxiv.org/pdf/2602.21467), [HTML](https://arxiv.org/abs/2602.21467)
### Authors
William Youngwoo Chung,Calvin Yeung,Hansen Jin Lillemark,Zhuowen Zou,Xiangjian Liu,Mohsen Imani
### Background
人工智能和神经科学的一项关键挑战是如何理解神经系统如何学习能够捕捉世界内在动态的表示。大多数世界模型使用无结构的神经网络来表示转移函数，这限制了模型的解释性、样本效率以及在未见状态或动作组合上的泛化。
### Innovation
该研究提出了一种基于向量符号架构原则（VSA）的泛化世界模型，引入了可学习的傅里叶全息缩减表示（FHRR）编码器，将状态和动作映射到具有学习群结构的高维复数向量空间，通过元素-wise复数乘法建模转移。这种结构化的表示被训练为大致不变，能够在潜在空间中直接实现强多步组合，并在各种实验中展示出泛化性能。
### Conclusion
在离散格子世界环境中，该模型达到了87.5%的零样本准确率，对于未见的状态-动作对，在20步时间范围的回放中获得了53.6%更高的准确率，并且相对于一个全连接层次网络基础模型的噪声鲁棒性提高了4倍。这些结果表明，在潜在空间中训练具有群结构的表示可以带来泛化、数据有效和可解释的世界模型，为实际规划和推理提供了结构化的模型途径。
