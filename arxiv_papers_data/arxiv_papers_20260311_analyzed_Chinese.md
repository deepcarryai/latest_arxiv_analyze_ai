# 20260311
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 通过情境规范使AI评估部署相关化 [PDF](https://arxiv.org/pdf/2603.06811), [HTML](https://arxiv.org/abs/2603.06811)
### Authors
Matthew Holmes,Thiago Lacerda,Reva Schwartz
### Background
许多组织在从AI部署中挖掘价值方面遇到困难，这加大了对以明智方式评估AI的需求。现状下的AI评估方法隐藏了最终决定部署成功的操作现实，使得外部决策者难以了解AI工具是否能持续交付价值。
### Innovation
提出了情境规范过程，作为一种支持和引导部署决策的过程。情境规范将各利益相关者对特定环境中的重要因素的模糊看法转化为明确具体的构建：即评估旨在捕捉的具体定义、属性、行为和结果，使其能够在具体环境中被观察和测量。
### Conclusion
情境规范过程为评估组织实际管理的部署环境中的AI系统可能带来什么提供了一种基础的路线图。
## 2. `cs.AI` - LieCraft：评估语言模型欺骗能力的多代理框架 [PDF](https://arxiv.org/pdf/2603.06874), [HTML](https://arxiv.org/abs/2603.06874)
### Authors
Matthew Lyle Olson,Neale Ratzlaff,Musashi Hinck,Tri Nguyen,Vasudev Lal,Joseph Campbell,Simon Stepputtis,Shao-Yen Tseng
### Background
大型语言模型（LLMs）展现出广泛的应用能力，但也带来了严重的安全风险，尤其是模型获得更大自主性、人类监督减少时，模型陷入欺骗的潜在风险增加。现有的基于游戏的评估方法存在关键局限性，无法有效测量LLMs的欺骗能力。
### Innovation
本文提出了一种名为LieCraft的新颖评估框架和沙盒，用以衡量LLMs的欺骗能力。LieCraft是一种新颖的多人隐藏角色游戏，参与者选择道德准则并执行长期策略以完成任务。LieCraft还包括10个实际场景，确保游戏具有伦理意义和高风险性，并通过精心设计的游戏机制和奖励系统确保游戏平衡。
### Conclusion
研究结果显示，尽管在能力和整体对齐上存在差异，所有模型都愿意采取不道德行为、掩饰意图并直接撒谎以实现目标。本文的工作提供了评估语言模型欺骗能力的新方法，并指出需要进一步研究来确保这些模型的行为满足伦理要求。
## 3. `cs.AI` - MultiGen：在扩散游戏引擎中编辑多玩家世界的层级设计 [PDF](https://arxiv.org/pdf/2603.06679), [HTML](https://arxiv.org/abs/2603.06679)
### Authors
Ryan Po,David Junhao Zhang,Amir Hertz,Gordon Wetzstein,Neal Wadhwa,Nataniel Ruiz
### Background
视频世界模型在互动模拟和娱乐方面展现出了巨大的潜力，但当前系统在两个重要方面仍然存在问题：用户对环境的控制和能够复现及编辑的体验，以及玩家之间的共同推理，其中玩家对共享世界有一定的影响力。
### Innovation
本文引入了系统中的显式外部内存，这是一种持久存在的状态，与模型的上下文窗口独立运作，通过用户动作持续更新并查询整个生成过程中。不同于传统的基于下一帧预测的扩散游戏引擎，本文的方法将生成过程分解为记忆、观察和动力学模块。这种设计使得用户能够通过可编辑的记忆表示直接控制环境结构，并且自然地扩展到具有协调视点和一致跨玩家交互的实时多人游戏。
### Conclusion
综上所述，本文通过在扩散游戏引擎中引入外部显式内存和将生成过程分解为记忆、观察和动力学模块的方法，实现了用户对环境结构的直接、可编辑控制，并支持实时多人游戏，同时保持各个玩家交互的一致性。
## 4. `cs.AI` - 自主的AI代理在期权对冲中的应用：通过关注损失概率的强化学习提升金融稳定性 [PDF](https://arxiv.org/pdf/2603.06587), [HTML](https://arxiv.org/abs/2603.06587)
### Authors
Minxuan Hu,Ziheng Chen,Jiayu Yi,Wenxi Sun
### Background
自主AI代理在衍生品市场中的应用扩大了静态模型校准和实际对冲结果之间的差距。本文旨在通过引入两种强化学习框架来解决这一问题，包括一种新颖的期权定价复制学习方法（RLOP）和适应性扩展的Q学习方法（QLBS）。这两种方法都优先考虑了损失概率，并使学习目标与对冲尾部风险相一致。
### Innovation
文章提出了两种新的强化学习框架，一种是RLOP（复制学习的期权定价），另一是QLBS（适应性的Q学习扩展）。这些方法主要关注降低损失概率，调整学习目标以更好地适应尾部风险敏感的对冲策略。
### Conclusion
通过利用上市的SPY和XOP期权，使用实现路径Delta对冲结果分布、损失概率以及尾部风险测量（如预期损失）来评估模型性能。实证结果表明，RLOP方法在大多数对冲结果中减少了损失频率，并在压力情况下提供了最明显的尾部风险改进，尽管隐含波动率拟合更偏好参数模型，但对冲后的实际表现预测不佳。这种摩擦感知式RL框架支持AI增强交易系统在自主衍生品风险管理中的实际应用。
## 5. `cs.AI` - 从嘈杂且部分观测中发现主导方程的约束对称性语言引导程序合成 [PDF](https://arxiv.org/pdf/2603.06869), [HTML](https://arxiv.org/abs/2603.06869)
### Authors
Mirza Samad Ahmed Baig,Syeda Anshrah Gillani
### Background
发现紧凑的支配方程是定量科学的主要目标之一，但在测量噪声大、关键状态变量未观测或多个符号结构在统计不确定性范围内都能解释数据的情况下，实际的发现管道常常会失败。
### Innovation
SymLang 提出了一个统一框架，将三种之前独立的想法结合起来：(i) 带有类型约束的对称性控制语法，它内编码了量纲分析、群论不变性和对称性约束，平均消除掉 71.3% 的候选表达式树；(ii) 语言模型引导的程序合成，其中经过微调的 7B 参数提议者，在受可解释数据描述符条件限制的情况下，有效地在受限搜索空间中导航；(iii) 带 MDL 正则化的贝叶斯模型选择结合块-bootstrap 稳定性分析，用于量化结构不确定性而非挑选出单一最佳方程。该框架在不同动力系统中实现了显著的性能提升，包括力学、电磁学、热力学、种群动力学和非线性振荡器。
### Conclusion
SymLang 在 10% 观测噪声下实现了 83.7% 的准确结构恢复率，比第二优基线高 22.4 个百分点，同时将超出分布外的外推误差降低了 61%，并几乎消除了守恒定律的违反（物理漂移为 3.1 x 10-3 对比最接近的竞争者为 187.3 x 10-3）。该框架在所有测试情况下都能正确识别结构退化，明确报告而不会返回一个自信的错误单一方程。该框架完全开源且可重复，为从原始数据到可解释、可物理验证的符号定律提供了一个原理性的路径。
## 6. `cs.AI` - LEAD：打破长时间推理中的不可恢复瓶颈 [PDF](https://arxiv.org/pdf/2603.06870), [HTML](https://arxiv.org/abs/2603.06870)
### Authors
Denys Pushkin,Emmanuel Abbe
### Background
在大型语言模型（LLMs）上实现长期执行时，即使提供了高层策略，系统仍然容易变得不稳定。尽管分解是提高稳定性的关键，但极端分解会导致一个‘不可恢复瓶颈’，这是因为错误分布的高度非均匀性导致在某些‘难点’步骤上的一致错误无法被纠正。
### Innovation
文章提出了Lookahead-Enhanced Atomic Decomposition（LEAD），通过结合短期未来验证和重叠卷出的聚合，实现了稳定的长期推理。LEAD能够在保持足够局部上下文的同时提供足够的隔离环境，避免流水线错误累积，从而提高了模型的长期推理能力。
### Conclusion
通过LEAD方法，o4-mini模型可以解决复杂度为n=13的跳棋问题，而极端分解法则在n=11时就无法解决问题。LEAD有效缓解了长时间推理中的不可恢复瓶颈问题，显著提高了推理的稳定性和持久性。
## 7. `cs.AI` - 逐步扩展策略，而非计算：一种独立开源的StarCraft II基准测试，便于强化学习研究 [PDF](https://arxiv.org/pdf/2603.06608), [HTML](https://arxiv.org/abs/2603.06608)
### Authors
Sourav Panda,Shreyash Kale,Tanmay Ambadkar,Abhinav Verma,Jonathan Dodge
### Background
研究社区缺乏在StarCraft II完整游戏和迷你游戏之间的中间地带。完整游戏广阔的状态和行动空间导致奖励信号稀少且噪声大，而迷你游戏中的简单代理则表现出饱和性能。这种复杂性的差距阻碍了稳定课程设计的发展，并阻碍了研究人员在受到现实计算预算限制的即时战略（RTS）环境中使用现代强化学习算法进行实验。
### Innovation
提出了一套开源基准系列的第一项内容——Two-Bridge Map Suite，旨在作为一个中介基准来弥补上述差距。通过禁用经济机制（如资源采集、基地建设以及迷雾视线等），该环境将核心战术技能（远程导航和微战斗）进行了隔离。初步实验表明，代理能够在不增加完整游戏计算成本的前提下学习到一致的机动和作战行为。
### Conclusion
Two-Bridge作为PySC2上的一个轻量级、兼容Gym的包装器发布，并且其地图、包装器和参考脚本均完全开源，以促进其作为标准基准的广泛应用和接受度。
## 8. `cs.AI` - 打破马尔可夫诅咒：基于不对称认知势能的多智能体辩论 [PDF](https://arxiv.org/pdf/2603.06801), [HTML](https://arxiv.org/abs/2603.06801)
### Authors
Yuhan Liu,Juntian Zhang,Yichen Wu,Martin Takac,Salem Lahlou,Xiuying Chen,Nils Lukas
### Background
多智能体辩论（MAD）作为一种提升大规模语言模型推理能力的有前途的范式已经发展起来。然而，近期的研究揭示了一个局限性：标准的MAD无法提高信念的正确性，超越单纯多数表决的效果；我们将此称为马尔可夫诅咒。马尔可夫诅咒源于相关错误导致代理会朝错误共识方向收敛，辩论仅仅强化了集体错误而不是过滤噪声。
### Innovation
我们提出了AceMAD框架，通过利用不对称的认知势能来打破马尔可夫诅咒，将MAD从随机游走转变为具有正漂移的定向收敛过程。通过同行预测机制，代理预测彼此的信念分布，揭示了不对称的认知势能：真理持有者不仅知道正确的答案，还能预见众人的误解，而视觉幻觉的多数则对他们集体的错误视而不见。这种不对称性形成了一个可用的信息理论优势，通过严格恰当评分规则进行量化，并通过非线性聚合，最终转化为向真理的次鞅漂移，直接打破了马尔可夫诅咒。
### Conclusion
实验表明，即使初始多数意见有误，AceMAD也能恢复稀疏的真实信号并显著优于基线方法。
## 9. `cs.AI` - Best-of-Tails：在推理时对乐观与悲观进行联结对接 [PDF](https://arxiv.org/pdf/2603.06797), [HTML](https://arxiv.org/abs/2603.06797)
### Authors
Hsiang Hsu,Eric Lei,Chun-Fu Chen
### Background
当前策略面临着一个根本性的困境：乐观的方法，如Best-of-N，由于奖励欺诈易受攻击；而悲观的正规化方法通常会抑制探索，以发现高质量的回应。现有的方法都存在一定的局限性。
### Innovation
该论文通过遗憾最小化的视角形式化了这一权衡，并证明了最优策略依赖于奖励分布的尾部行为。论文提出了Best-of-Tails（BoT）框架，该框架使用Tsallis散度作为调节器，可以在悲观和乐观之间提供更细粒度的插值。BoT利用Hill估计来表征每个提示的奖励尾部特征，并动态调整其选择规则以平衡探索收益和对齐误差。
### Conclusion
通过数学、多项选择推理和人类偏好评估，BoT在各种参考和奖励模型配置中比固定策略基准提高了对齐性能。
## 10. `cs.AI` - 多智能体世界边界中的持续学习问题：强化世界边缘 [PDF](https://arxiv.org/pdf/2603.06813), [HTML](https://arxiv.org/abs/2603.06813)
### Authors
Dane Malenfant
### Background
在强化学习（RL）中，可重用的决策结构可以在不同时间点之间保留下来，但这种能力依赖于如何定义智能体与环境之间的边界。在静态、有限时间间隔的马尔可夫决策过程（MDP）中，可以通过共享所有成功轨迹的（不一定是连续的）状态-动作子序列构建一个不变的核心。然而，当同样的任务被嵌入到去中心化的马尔可夫游戏中，并将同伴智能体整合到环境中时，每次更新对手智能体的策略都会导致新的MDP产生，此时每个回合的不变核心可能会缩小甚至消失。
### Innovation
该研究创新地指出，在具有多个智能体的环境中，同伴智能体策略的更新可以导致每次回合的不变核心结构发生变化或消失。此外，通过量化诱导核和奖励的变化预算，研究将边界漂移与不变性的丧失联系起来。这些发现对于理解多代理强化学习（MARL）中的持续学习问题具有重要意义。
### Conclusion
研究认为，持续的RL问题主要源于多代理环境中的智能体与环境边界的不稳定性，而不是外生的任务切换。这一观点为未来的工作指明了方向，即如何保持、预测或管理边界漂移。
## 11. `cs.AI` - InterReal: 基于统一物理的模仿学习框架，用于学习人与物体交互技能 [PDF](https://arxiv.org/pdf/2603.07516), [HTML](https://arxiv.org/abs/2603.07516)
### Authors
Dayang Liang,Yuhang Lin,Xinzhe Liu,Jiyuan Shi,Yunlong Liu,Chenjia Bai
### Background
人形机器人的交互能力是核心能力之一，但现有框架大多集中在非交互的整体控制上，这限制了其实用性。
### Innovation
开发了InterReal，这是一种基于物理的统一模仿学习框架，用于实际环境下的人机物交互控制。InterReal引入了手物接触约束的霍伊运动数据增强方案，增强了政策的稳定性。提出了自动奖励学习者，通过元政策根据关键跟踪误差指标探索和分配奖励信号，提高了交互策略的学习效果。
### Conclusion
InterReal 在盒子拾取和推盒任务中展示了最佳跟踪精度和最高的任务成功率，还验证了其在真实世界机器人Unitree G1上的有效性和鲁棒性。
## 12. `cs.AI` - 一种从临床文本中联合抽取概念、断言和关系的神经基线 [PDF](https://arxiv.org/pdf/2603.07487), [HTML](https://arxiv.org/abs/2603.07487)
### Authors
Fei Cheng,Ribeka Tanaka,Sadao Kurohashi
### Background
临床信息提取通常涉及概念识别、断言分类和关系抽取等任务。现有的多阶段任务建模在临床领域是一个未被充分探索的课题。现有的独立任务设置使得联合模型难以直接与现有的流水线工作进行比较。
### Innovation
定义了一个联合任务设置，并提出了一种新的端到端系统，以联合优化三个阶段的任务。采用多种嵌入技术（词嵌入、上下文嵌入和领域特定的上下文嵌入）对所提出的联合系统和流水线基线进行了实证研究。所提出的联合系统在概念、断言和关系F1得分上分别优于流水线基线0.3、1.4和3.1。
### Conclusion
该工作将联合方法与临床信息提取联系起来。所提出的方法可以作为未来研究的强联合基线。代码已公开可用。
## 13. `cs.AI` - SeDa：一种统一的数据集发现和多实体增强语义探索系统 [PDF](https://arxiv.org/pdf/2603.07502), [HTML](https://arxiv.org/abs/2603.07502)
### Authors
Kan Ling,Zhen Qin,Yichi Zhu,Hengrun Zhang,Huiqun Yu,Guisheng Fan
### Background
开源数据平台和研究存储库的持续扩展导致了数据集生态系统的片段化，给数据跨源发现和解释带来了重大挑战。
### Innovation
引入了SeDa——一个统一的数据集发现、语义注释和多实体增强导航框架。SeDa整合了超过200个平台上的760万多个数据集，涵盖政府、学术和工业领域。该框架通过语义提取和标准化来 harmonize 不同的元数据表示，然后通过主题标签机制构建可扩展的标签图以支持主题检索和跨域关联，同时嵌入验证机制以确保数据来源的可信性和链接的可用性。
### Conclusion
与流行的诸如ChatPD和Google Dataset Search等数据集搜索平台的比较实验表明，SeDa在覆盖率、时效性和追溯性方面表现出色。SeDa为可信、语义丰富和全球可扩展的数据集探索奠定了基础。
## 14. `cs.AI` - 通过架构流独立实现可解释的变压器 [PDF](https://arxiv.org/pdf/2603.07482), [HTML](https://arxiv.org/abs/2603.07482)
### Authors
Clayton Kerce,Alexis Fox
### Background
尽管Transformer表现出强大的性能，但它们内部的决策过程仍然不透明。研究通过架构约束设计实现的可解释性，提出了架构流独立的概念：维护一个携带符号结构的标记流和独立的上下文语义流，处理过程中始终保持独立观察，直到输出时才进行整合。
### Innovation
提出了一种新的Late Fusion Architecture (LFA)，通过在所有最终层中展示可解释的符号头，证明了架构流独立性。与标准Transformer相比，LFA在六层中的第三层就展示了可解释性衰减。通过引入Token-Position Dependence Score (PDS)量化这种效果，LFA的PDS最高值为0.276，标准Transformer为0.058。干预实验显示，抑制LFA的近期头只会造成轻微的语义损伤，而基线模型则会出现灾难性混淆。
### Conclusion
通过架构约束，LFA模型能够提高基本的学习机制稳定性，平均提高42%，最高可达到50%，并且在极端情况下仍然表现出一定程度的不可完全坍塌。架构独立性引导模型向语义理解发展，而非位置启发式理解，从而将可解释性视为一种通过结构约束实现的可设计架构标准。
## 15. `cs.AI` - 从思考者到社会：AI代理分层自主进化的安全性 [PDF](https://arxiv.org/pdf/2603.07496), [HTML](https://arxiv.org/abs/2603.07496)
### Authors
Xiaolei Zhang,Lu Zhou,Xiaogang Xu,Jiafei Wu,Tianyu Du,Heqing Huang,Hao Peng,Zhe Liu
### Background
AI代理已经从被动的预测工具演变为能够自主决策和进行环境交互的主动实体，这得益于大型语言模型（LLMs）的推理能力。然而，这一进化引入了现有框架无法解决的关键安全问题。
### Innovation
提出了一种名为Hierarchical Autonomy Evolution（HAE）的安全框架，将代理安全分为三个层级：认知自主（L1）关注内部推理完整性；执行自主（L2）涵盖工具介导的环境交互；集体自主（L3）解决多代理生态系统中的系统性风险。该框架还根据不同威胁类型提出了威胁分类，并评估了现有防御措施，同时指出了关键的研究空白。
### Conclusion
研究旨在指导具有多层自主安全架构的可信AI代理系统的开发。
## 16. `cs.AI` - 基于神经动力学的预训练框架用于个性化脑功能网络构建 [PDF](https://arxiv.org/pdf/2603.07524), [HTML](https://arxiv.org/abs/2603.07524)
### Authors
Hongjie Jiang,Yifei Tang,Shuqiang Wang
### Background
脑活动本质上是受到解剖空间限制的神经动力学过程，这导致了神经活动的空间分布模式和相关模式在不同和异质场景下存在显著变化。然而，现有的主导性脑功能网络构建方法依赖预定义的大脑图谱和线性假设，无法精确捕捉这些异质场景下的神经活动模式，从而限制了构建的脑功能网络的一致性和通用性。
### Innovation
提出了一个基于神经动力学的预训练框架，用于个性化脑功能网络构建。该框架提取了在异质场景下的个性化神经活动模式表示，通过利用这些表示指导脑分区和神经活动相关性估计，从而实现个性化脑功能网络的获取。
### Conclusion
通过对18个数据集的系统评估，实验结果表明，该提出的框架在异质场景中表现更优。整体而言，该提出的框架挑战了现有的主导性的脑功能网络构建方法。
## 17. `cs.AI` - （视觉-）语言模型中的跨模态分类泛化 [PDF](https://arxiv.org/pdf/2603.07474), [HTML](https://arxiv.org/abs/2603.07474)
### Authors
Tianyang Xu,Marcelo Sandoval-Castaneda,Karen Livescu,Greg Shakhnarovich,Kanishka Misra
### Background
研究了语言模型（LM）仅从表层形式学习到的语义表示与从更具体证据中学习的语义表示之间的相互作用。具体在视觉-语言模型（VLM）中考察这一问题，其中预训练的LM与预训练的图像编码器对齐，特别是在预测图像中对象的上位词任务上。研究表明，LM甚至在几乎没有任何上位词证据的情况下也能恢复这部分知识。
### Innovation
通过在VLM中保持图像编码器和LM不变，只学习中间映射，逐渐剥夺VLM获得上位词的直接证据，并测试LM能否恢复这部分知识。此研究发现，LM能够从语言线索中推断和泛化，具有高度类别内视觉相似度的反事实图像标签映射也不例外。
### Conclusion
这些发现表明，（视觉-）语言模型中的跨模态泛化是由外部输入的一致性和语言线索推导出的知识共同产生的结果。
## 18. `cs.AI` - 在MCP基础的AI系统中予敌以机会，他们将会得寸进尺：理解并衡量调用者身份混淆 [PDF](https://arxiv.org/pdf/2603.07473), [HTML](https://arxiv.org/abs/2603.07473)
### Authors
Yuhang Huang,Boyang Ma,Biwei Yan,Xuelong Dai,Yechao Zhang,Minghui Xu,Kaidi Xu,Yue Zhang
### Background
Model Context Protocol (MCP) 是一个开放和标准化的接口，它使大型语言模型（LLMs）能够与外部工具和服务交互，并且已经被越来越多的人工智能代理采用。然而，基于MCP系统的安全性尚未得到广泛研究。这项工作对集成在MCP客户端中的MCP服务器进行了大规模的安全分析，揭示了将MCP服务器视为受信任实体而不认证调用者身份的基本安全性问题。由于MCP服务器通常无法区分谁在发出请求，一个授权决策可能会无意识地授予多个、潜在不可信的调用者访问权限。实验结果表明，大多数MCP服务器依赖持久授权状态，允许在初始授权后不重新验证身份就进行工具调用，而不论调用者是谁。此外，许多MCP服务器未能在单个工具级别上强制执行身份验证，这使得未经授权的未授权访问变得可能。
### Innovation
研究展示了单次授权和服务器级别的信任如何显著扩大基于MCP系统的攻击面，突显了显式调用者身份验证和细粒度授权机制的必要性。
### Conclusion
本文通过大规模实证分析揭示了基于MCP的AI系统中调用者身份混淆的问题，强调了强制显式身份认证和权宜身份验证机制的重要性，以保护MCP系统的安全性。
## 19. `cs.AI` - SketchGraphNet: 记忆高效混合图变换器用于大规模草图语料库识别 [PDF](https://arxiv.org/pdf/2603.07521), [HTML](https://arxiv.org/abs/2603.07521)
### Authors
Shilong Chen,Mingyuan Li,Zhaoyang Wang,Zhonglin Ye,Haixing Zhao
### Background
本文从图究的角度研究大规模草图识别，直接将自由手绘草图建模为结构化图，而非栅格图像或笔画序列。为了支持系统的评估，构建了包含344个类别、超过340万图结构化草图的SketchGraph基准数据集。
### Innovation
提出了SketchGraphNet，这是一种混合图神经网络架构，结合了局部消息传递和高效全局注意力机制，无需依赖辅助位置或结构编码。与基于Performer的全局注意力机制相比，MemEffAttn进一步降低了峰值GPU内存超过40%，训练时间超过30%，同时保持了相当的准确性。
### Conclusion
在统一训练配置下，SketchGraphNet在SketchGraph-A和SketchGraph-R上的Top-1准确率分别为83.62%和87.61%。此外，MemEffAttn在峰值GPU内存和训练时间上均优于基于Performer的全局注意力机制，但仍保持相当的准确性。
## 20. `cs.AI` - Drift模型和得分基于模型的统一视角 [PDF](https://arxiv.org/pdf/2603.07514), [HTML](https://arxiv.org/abs/2603.07514)
### Authors
Chieh-Hsin Lai,Bac Nguyen,Naoki Murata,Yuhta Takida,Toshimitsu Uesaka,Yuki Mitsufuji,Stefano Ermon,Molei Tao
### Background
Drifting模型通过优化由核在数据分布和模型分布之间引起的数据和模型分布的平均偏移不一致性来训练一阶生成器。Laplace核在实践中默认使用。这一模型在每个点比较了核加权的向邻近数据样本的位移与其向邻近模型样本的对应位移，以获得生成样本的传输方向。论文通过证明Drifting方法实际上是核平滑分布上的一种得分方法，从而将Drifting方法与扩散模型背后的得分匹配原则联系起来。对于高斯核，整个平均偏移场与高斯平滑数据和模型分布之间的得分差相匹配。这种身份来自于Tweedie公式，该公式将高斯平滑密度的得分与相应条件均值联系起来，从而表明高斯核Drifting方法实际上是在平滑分布上一种与得分匹配风格的目标函数。
### Innovation
论文通过证明Drifting方法等同于核平滑分布上的得分方法，揭示了Drifting和得分匹配原理之间的联系。对于高斯核，整个平均偏移场与高斯平滑数据和模型分布之间的得分差相匹配。这也解释了Drifting方法和分布匹配蒸馏（DMD）方法之间的关系：两者都使用得分不匹配的传输方向，但Drifting通过核邻域的非参数方法实现得分信号，而DMD使用预训练的扩散教师。论文还为一般径向核推导了精确分解，并证明了对于拉普拉斯核，当处于低温和高维环境时，Drifting仍然是得分匹配的一种准确近似。
### Conclusion
总之，该研究通过将Drifting方法与得分匹配原理联系起来，为Drifting方法提供了一个新的视角，同时证明了它在特定条件下与得分匹配的等价性，并讨论了Drifting与其他方法之间的关系。
## 21. `cs.CV` - 无需重新训练对新型3D打印对象进行分类：迈向增材制造后处理自动化 [PDF](https://arxiv.org/pdf/2603.07465), [HTML](https://arxiv.org/abs/2603.07465)
### Authors
Fanis Mathioulakis,Gorjan Radevski,Silke GC Cleuren,Michel Janssens,Brecht Das,Koen Schauwaert,Tinne Tuytelaars
### Background
在工业增材制造中，自动化后生产工作流需要可靠地分类3D打印对象。尽管其他打印阶段已经实现了广泛的自动化，但在对象分类任务上仍然依赖于手动检查，因为需要分类的对象集每天都在变化，频繁的模型重新训练变得不可行。自动化识别步骤对于提高运营效率至关重要。一个能够通过利用与CAD模型对应的3D打印对象照片进行分类，并且不需要重新训练的视觉模型将在这类应用场景中非常有利。
### Innovation
作者引入了ThingiPrint数据集，这是一个公开的数据集，将CAD模型与其真实的3D打印照片配对，以系统地评估模型。利用ThingiPrint，对一系列现有的视觉模型进行了基准测试。并且通过对比微调方法，使用旋转不变制目标实现了对未见过的3D打印对象的有效原型分类。这种方法仅依赖于可用的CAD模型，因此在引入新对象时无需重新训练。实验表明这种方式优于标准预训练基准，显示出更好的泛化能力和实际应用的相关性。
### Conclusion
通过这种方法，能够在无需重新训练的情况下实现对新型3D打印对象的分类，从而增强后生产自动化流程的效率和灵活性。实验结果表明，此方法优于标准的预训练基准，提示其在实际应用中的潜在改进和重要性。
## 22. `cs.CV` - RobustSCI: 在真实世界退化条件下从重构到恢复的 Snapshot 压缩成像 [PDF](https://arxiv.org/pdf/2603.07489), [HTML](https://arxiv.org/abs/2603.07489)
### Authors
Hao Wang,Yuanfan Li,Qi Zhou,Zhankuo Xu,Jiong Ni,Xin Yuan
### Background
现有深度学习算法在 Snapshot 压缩成像 (SCI) 中取得了巨大成功，但主要集中在从干净的测量数据中重构图像，忽视了实际应用中被捕获信号常常因运动模糊和低光照严重退化的问题。现有的模型因此在实际应用中表现不佳。为解决这一局限，该研究首次专注于鲁棒的视频SCI恢复，目标从“重构”转为“恢复”，即从退化的测量数据中恢复原始清晰的画面。
### Innovation
研究提出了RobustSCI，一种在网络中增强强编码-解码骨干的新颖RobustCFormer模块。该模块包含两个并行分支——多尺度去模糊分支和频率增强分支，以明确地在恢复过程中分离并去除退化。此外，引入了RobustSCI-C（RobustSCI-级联），将预训练的轻量级后处理去模糊网络集成进来，显著提升了恢复性能，同时保持较低的资源消耗。
### Conclusion
大量实验表明，该方法在新的退化测试基准上优于所有最先进的模型，并在实际退化SCI数据上的验证进一步证明了其实用有效性，将SCI从重构捕获的内容提升到了恢复真实发生的情况。
## 23. `cs.CV` - 多模态解耦和再耦合网络在鲁棒3D目标检测中的应用 [PDF](https://arxiv.org/pdf/2603.07486), [HTML](https://arxiv.org/abs/2603.07486)
### Authors
Rui Ding,Zhaonian Kuang,Yuzhe Ji,Meng Yang,Xinhu Zheng,Gang Hua
### Background
多视角（鸟瞰图BEV）的3D物体检测在基准测试中取得了显著进步，但在实际环境中，由于LiDAR传感器配置或相机场景条件导致的数据损坏问题，检测精度可能会显著下降。此前模型设计的一个瓶颈在于多模态BEV特征融合中的紧密耦合，这可能导致系统性能下降，尤其是在某一模态或两者都损坏的情况下。
### Innovation
提出了一种多模态解耦和再耦合网络，用于在数据损坏情况下实现鲁棒的3D物体检测。该网络将不同模态的特征明确解耦为模态不变和模态特定部分，允许不变特征互相补偿，并减轻损坏模态的负面影响。然后将这些特征再耦合为三个专家，分别处理不同类型的损坏情况（LiDAR、相机以及两者），利用模态不变特征作为稳健信息，模态特定特征作为补充，并适配性地融合这三个专家以获得稳健的特征进行3D物体检测。
### Conclusion
通过构建一个基于nuScenes的包含大量LiDAR、相机以及两者的损坏数据的基准集，并在干净的nuScenes数据上进行训练，在所有类型的损坏数据上进行测试。实验结果显示，该模型在损坏和干净数据上的准确率都优于最近的模型。
## 24. `cs.CV` - RayD3D: 沿射线蒸馏深度知识以实现稳健的多视图3D目标检测 [PDF](https://arxiv.org/pdf/2603.07493), [HTML](https://arxiv.org/abs/2603.07493)
### Authors
Rui Ding,Zhaonian Kuang,Zongwei Zhou,Meng Yang,Xinhu Zheng,Gang Hua
### Background
多视角三维检测（基于鸟瞰图）对于自动驾驶和机器人技术至关重要，但其在现实环境中的稳健性有限，因为难以准确预测深度值。目前主流的解决方案是跨模态蒸馏，这种方法将LiDAR的深度信息传输到相机模型中，但同时也无意中传输了与深度无关的信息（例如LiDAR的密度）。
### Innovation
提出的RayD3D方法沿着射线（从相机到物体真实位置的线）传输关键深度知识。为此方法设计了两种基于射线的蒸馏模块：射线对比蒸馏（RCD）和射线加权蒸馏（RWD）。射线对比蒸馏通过沿射线采样来学习LiDAR如何准确定位物体，射线加权蒸馏则根据射线来适配性调整蒸馏权重，以最小化与深度无关信息的干扰。
### Conclusion
RayD3D方法被广泛应用于三个代表性基于鸟瞰图的模型：BEVDet、BEVDepth4D和BEVFormer。该方法在干净的NuScenes数据集上进行训练，并在干净的NuScenes和RoboBEV数据集上进行测试，涵盖了多种数据扰动。与最近发布的多视图和蒸馏模型相比，我们的方法在所有场景中都显著提高了基础模型的稳健性，且未增加推理成本。
## 25. `cs.CV` - 通过骨骼隐秘扩散实现高保真医疗形状生成 [PDF](https://arxiv.org/pdf/2603.07504), [HTML](https://arxiv.org/abs/2603.07504)
### Authors
Guoqing Zhang,Jingyun Yang,Siqi Chen,Anping Zhang,Yang Li
### Background
在医疗数据分析中，解剖形状建模是一个基本问题。但是，解剖结构的几何复杂性和拓扑变化给精确的形状生成带来了重大挑战。
### Innovation
本文提出了一种骨骼隐秘扩散框架，该框架在生成高精度医疗形状时能高效并保持高保真度。引入了形状自编码器，在其中编码器通过可微骨架化模块捕捉全局几何信息，并聚集局部表面特征到形状隐变量中，而解码器预测相应的隐式场。新的形状通过隐空间扩散模型生成，接着通过神经隐式解码和网格提取生成。
### Conclusion
在MedSDF和血管数据集上进行的大量实验表明，所提出的方法在重建和生成质量上表现出优异的表现，同时计算效率高于现有方法。源代码可在指定链接获得。
## 26. `cs.CV` - EvolveReason: 自我进化推理范式在可解释的深度伪造面部图像识别中的应用 [PDF](https://arxiv.org/pdf/2603.07515), [HTML](https://arxiv.org/abs/2603.07515)
### Authors
Binjia Zhou,Dawei Luo,Shuai Chen,Feng Xu,Seow,Haoyuan Li,Jiachi Wang,Jiawen Wang,Zunlei Feng,Yijun Bei
### Background
随着AIGC技术的快速进步，发展识别方法以应对由深度伪造带来的安全挑战变得至关重要。面部伪造的识别技术可以分为两大类：传统分类方法和可解释的VLM方法。前者提供了分类结果但缺乏解释能力，后者尽管能够提供粗粒度的解释，但往往会受到幻觉和细节不足的问题困扰。
### Innovation
我们提出了EvolveReason，该方法模仿人类审计员在识别面部伪造时的推理和观察过程。通过构建适应先进VLM的CoT-Face数据集，我们的方法引导模型以类似人类的方式思考，促使模型输出推理过程和判斷结果。此外，我们的框架集成了伪造隐空间分布捕捉模块，使EvolveReason能够识别出无法从原始图像中提取的高频率伪造线索。为了进一步增强文本解释的可靠性，我们引入了一种自我进化探索策略，利用强化学习使模型能够在两阶段过程中迭代探索和优化其文本描述。
### Conclusion
实验结果表明，EvolveReason不仅在识别性能上超越了当前最先进的方法，而且准确地识别了伪造细节，并展示了泛化能力。
## 27. `cs.CV` - DocCogito: 集成布局认知与步骤级对接的视觉-语义链以增强文档理解 [PDF](https://arxiv.org/pdf/2603.07494), [HTML](https://arxiv.org/abs/2603.07494)
### Authors
Yuchuan Wu,Minghan Zhuo,Teng Fu,Mengyang Zhao,Bin Li,Xiangyang Xue
### Background
当前的文档大规模语言模型（MLLMs）虽已改进布局编码和链式思考（CoT）提示，但仍然缺乏构建完整、类人类的推理过程。这种改进通常是以隐式方式实现，并未能将布局认知与细化推理紧密结合。因此，文档理解需要不仅准确的答案，还需要清晰且基于证据推理的过程，特别是在高风险场景中。
### Innovation
本文提出了DocCogito，这是一种统一的框架，将全局布局感知与结构化的、基于区域的推理相结合。该框架中引入了轻量级的布局塔，用于提取可学习的全局布局先验令牌，以及确定性的视觉-语义链（VSC），这是一种比自由形式自然语言链式思考更清晰的结构化表示。此外，通过增强的标准奖励机制，以细粒度的区域置信度信号鼓励推理轨迹与相应证据区域保持一致。训练方法包括布局感知预训练、VSC引导的冷启动、拒绝采样和GRPO。
### Conclusion
广泛的实验结果表明，DocCogito在六个基准数据集（DocVQA、WTQ、ChartQA、TextVQA、OCRBench和InfoVQA）上具有强大的泛化能力，并在四个基准数据集上达到了最先进的性能。
## 28. `cs.CV` - FedEU: 证据不确定性驱动的视觉基础模型联邦微调方法在遥感图像分割中的应用 [PDF](https://arxiv.org/pdf/2603.07468), [HTML](https://arxiv.org/abs/2603.07468)
### Authors
Xiaokang Zhang,Xuran Xiong,Jianzhong Huang,Lefei Zhang
### Background
遥感图像分割（RSIS）在联邦环境中引起了越来越多的关注，因为它允许在不共享原始图像或注释的情况下分布式数据集间的协作模型训练。结合参数高效的微调（PEFT）技术，可以利用预训练的基模型在真实场景中的泛化能力，同时减小参数聚合和通信开销。然而，预训练模型在面对异构客户端数据时动态适应过程中，可能会增加模型更新的不确定性，导致联邦优化的可靠性下降，特别是缺乏对每个本地模型的不确定性估计。
### Innovation
FedEU提出了一个联邦优化框架，通过证据不确定性驱动的本地模型个性化不确定性建模来量化和识别风险高的区域。同时，利用针对客户端的特征嵌入（CFE），专门为每个客户端定制特征表示，增强通道感知特征表示并保持客户端特定的属性。通过Top-k不确定性引导加权（TUW）策略实现了适应性的全局聚合，从而减轻分布偏移和不可靠更新的影响。
### Conclusion
在三个大规模异构数据集上的大量实验表明，FedEU在遥感图像分割中表现出优越的性能。更重要的是，FedEU通过明确降低预测不确定性，实现了在不同客户端之间均衡的模型适应，增强了联邦优化结果的鲁棒性和可靠性。源代码将在该链接处提供。
## 29. `cs.CV` - EVLF: 早期视觉-语言融合在生成数据集蒸馏中的应用 [PDF](https://arxiv.org/pdf/2603.07476), [HTML](https://arxiv.org/abs/2603.07476)
### Authors
Wenqi Cai,Yawen Zou,Guang Li,Chunzhi Gu,Chao Zhang
### Background
数据集蒸馏（DD）旨在合成紧凑的训练集，使模型能够在显著减少样本数量的情况下实现高精度。最近，基于扩散的数据集蒸馏方法通常在后期引入语义指导，通过交叉注意力机制，其中文本提示主导生成过程。尽管这种方法保证了标签的相关性，但减少了视觉潜在信息的贡献，导致生成的样本过度矫正，模仿了提示模式而非反映内在视觉特征。
### Innovation
提出了一种早期视觉-语言融合（EVLF）方法，该方法在编码器和生成主干之间的过渡处对文本和视觉嵌入进行对齐。通过在该过渡处引入轻量级交叉注意力模块，早期表示在去噪过程中同时编码局部纹理和全局语义方向。重要的是，EVLF 可以无缝集成任何基于扩散的数据集蒸馏管道，无需任何特定任务的修改即可在不同的去噪架构和采样计划下工作。
### Conclusion
广泛实验表明，EVLF 生成了语义忠实且视觉一致的合成数据，无论在何种设置下都能在下游分类精度方面带来一致的改进。源代码可在该网址找到。
## 30. `cs.CV` - AMR-CCR: Anchored Modular Retrieval for Continual Chinese Character Recognition [PDF](https://arxiv.org/pdf/2603.07497), [HTML](https://arxiv.org/abs/2603.07497)
### Authors
Yuchuan Wu,Yinglian Zhu,Haiyang Yu,Ke Niu,Bin Li,Xiangyang Xue
### Background
古代中国字符识别是文化遗产数字化的核心能力，但实际工作流程天生是非平稳的：不断发掘出的新材料持续上线，带来不同书写的新增类别，并随时间扩展类别空间。这一过程被形式化为持续中国字符识别（Continual CCR），一个书写作业阶段、类别增量式的设置，结合了两项挑战：在类别逐渐增加且类间差异细微、增量数据稀缺的情况下进行可扩展学习；以及由于作家字体风格和载体状况差异导致的显著类别内多样性。
### Innovation
我们提出了AMR-CCR（Anchored Modular Retrieval for Continual Chinese Character Recognition），这是一种锚定模块检索框架，通过共享多模态空间中的基于嵌入的字典匹配来进行识别，允许通过简单扩展字典添加新类别。AMR-CCR 进一步引入了一种轻量级书写作业条件下的注射模块（SIA+SAR），以校准新上载的书写作业，同时保持跨阶段嵌入兼容性，并采用了图像衍生的多原型字典，以更好地覆盖不同的字体样式。
### Conclusion
为了支持系统的评估，我们构建了EvoCON，一个六阶段基准，涵盖了六种书写作业（OBC，BI，SS，SAC，WSC，CS），并增加了含义/形状描述以及未见字符的显式零样本分割（没有图像实例的）。
## 31. `cs.LG` - StructSAM: 结构和谱保持的Token合并方法用于分割任何模型 [PDF](https://arxiv.org/pdf/2603.07307), [HTML](https://arxiv.org/abs/2603.07307)
### Authors
Duy M. H. Nguyen,Tuan A. Tran,Duong Nguyen,Siwei Xie,Trung Q. Nguyen,Mai T. N. Truong,Daniel Palenicek,An T. Le,Michael Barz,TrungTin Nguyen,Tuan Dam,Ngan Le,Minh Vu,Khoa Doan,Vien Ngo,Pengtao Xie,James Zou,Daniel Sonntag,Jan Peters,Mathias Niepert
### Background
近期的Token合并技术已经显著提高了Vision Transformers (ViTs)的速度，通过减少自注意力处理的Token数量，同时无需重新训练。然而，将这些技术直接应用于Segment Anything Model (SAM)及其变体存在挑战。SAM的图像编码器混合使用窗口化和全局注意力，而其掩码解码器依赖于密集且基于提示条件的特征以实现准确的边界预测。
### Innovation
本文提出了一种名为StructSAM的解决方案，这是一种针对SAM的分辨率保持合并-恢复框架。StructSAM从特征梯度计算轻量级的Token能量分数，使用基于网格的均匀性筛选来保护边界和提示区域，然后在平坦区域内将Token合并到低能量的目的地，并进行明确的Token恢复。通过频谱图粗化视角，展示分数导向的合并与随机或限窗基础相比提供了有界的拉普拉斯频谱畸变。
### Conclusion
在八个自然和医疗基准测试中，StructSAM在保持轻微的mIoU/Dice下降的同时，将编码器的FLOPs减少了25-30% （在具有提示感知合并的情况下最高可达40%），并且在相同计算量下，普遍优于ToMe, PiToMe, ToMeSD, VidToMe和ALGM。
## 32. `cs.LG` - AgrI挑战赛：农业视觉跨团队验证的数据中心化人工智能竞赛 [PDF](https://arxiv.org/pdf/2603.07356), [HTML](https://arxiv.org/abs/2603.07356)
### Authors
Mohammed Brahimi,Karim Laabassi,Mohamed Seghir Hadj Ameur,Aicha Boutorh,Badia Siab-Farsi,Amin Khouani,Omar Farouk Zouak,Seif Eddine Bouziane,Kheira Lakhdari,Abdelkader Nabil Benghanem
### Background
农业视觉中的机器学习模型通常在精心策划的数据集上表现高精度，但无法在真实田地条件下泛化，主要是由于训练和部署环境之间的分布变化。此外，大多数机器学习竞赛主要关注模型设计，而将数据集视为固定资源，忽略了数据收集实践在模型泛化中的角色。
### Innovation
提出了AgrI挑战赛，这是一个以数据为中心的竞赛框架，其中多个团队独立收集田地数据集，产生一个异质多源基准，反映了获取条件的真实变异性。为了系统地评估独立收集的数据集之间的跨领域泛化，提出了跨团队验证（CTV），一种将每个团队的数据集视为不同领域的评估范式，包含两个互补协议：只在单一团队训练（TOTO）和排除单一团队的留一团队验证（LOTO）。
### Conclusion
单源训练下的泛化存在显著差距：模型在单一数据集验证中几乎达到完美准确性，但在其他团队数据集测试中，DenseNet121 和 Swin 变形器的验证-测试差距分别达到16.20%和11.37%。相比之下，多源合作训练大大提高了鲁棒性，分别将差距减少到2.82%和1.78%。此外，挑战还生成了一个包含50,673张来自十二个独立团队的六种树木种类的田地图像的公开数据集，用于农业视觉中的域移和数据驱动学习的研究。
## 33. `cs.LG` - 通过深条件变换模型实现条件等级-等级回归 [PDF](https://arxiv.org/pdf/2603.07230), [HTML](https://arxiv.org/abs/2603.07230)
### Authors
Xiaoyi Wang,Long Feng,Zhaojun Wang
### Background
代际流动性量化了父母的社会经济结果向子女的传递。传统的秩-秩回归（RRR）是标准方法，但直接加入协变量（RRRX）往往导致参数解释不清。条件秩-秩回归（CRRR）通过使用协变量调整后的（条件）秩来衡量组内流动性，解决了这个问题。文章在此基础上进一步改进了CRRR方法，通过使用深度条件转换模型（DCTM）和交叉契合来估计条件秩，从而在结构约束下实现端到端的条件分布学习，并能够处理非线性、高阶交互作用以及离散有序结果，而传统的CRRR在这些情况下可能较为复杂或容易配置错误。
### Innovation
本文提出了一种改进的条件秩-秩回归方法CRRR，通过使用深度条件转换模型（DCTM）和交叉契合来估计条件秩，从而实现更强的性能，适用于非线性、高阶交互作用以及离散有序结果。此外，对于离散结果，采用boldsymbol{text{ω}}-索引条件秩定义，并研究其对boldsymbol{text{ω}}的敏感性。对于连续结果，建立了提出估计量的渐近理论，并验证了可交换自举推断的有效性。研究表明，在简单/复杂的连续和离散有序设计中，方法表现出明显的准确性提升。
### Conclusion
最终，本文提出的方法被应用于两个实际案例研究，结果显示在美国收入中存在显著的组内持续性，同时在印度教育流动性中发现了明显的性别差异。
## 34. `cs.LG` - Vocos 基于的快速灵活音频带宽扩展 [PDF](https://arxiv.org/pdf/2603.07285), [HTML](https://arxiv.org/abs/2603.07285)
### Authors
Yatharth Sharma
### Background
当前音频带宽扩展技术在生成缺失的高频内容方面已经取得了一定进展，但面临成本效率和实时性的问题。需要一种既能在资源有限条件下高效运行又能支持高动态范围比特率转换的技术。
### Innovation
提出了一种基于Vocos的音频带宽扩展模型，通过在48 kHz下采样输入并使用神经语音编解码器生成缺失的高频内容。轻量级的Linkwitz-Riley启发式细化器通过平滑交叉口将生成的高频与原始低频信号合并。该模型不仅在NVIDIA A100和8核CPU上实现了超高的运行时效率（分别为0.0001和0.0053），还在生成的音频中保持了高质量。
### Conclusion
该研究展示了在极端吞吐量下实现具有实践意义且高质量的音频带宽扩展的可行性，具有较高的成本效率和实时性能。
## 35. `cs.LG` - 肠道客观听诊：肠道声音模式的自动分割和标注 [PDF](https://arxiv.org/pdf/2603.07215), [HTML](https://arxiv.org/abs/2603.07215)
### Authors
Zahra Mansour,Verena Uslar,Dirk Weyhe,Danilo Hollosi,Nils Strodthoff
### Background
肠鸣音通常短暂且幅值低，通过手动听诊难以准确检测，导致临床评估存在较大差异。数字声传感器可以获取高质量的肠鸣音信号，并允许自动化信号分析，从而为临床医生提供客观和定量的肠道活动反馈。
### Innovation
该研究提出了一种自动化流程，用于使用可穿戴声波传感器（SonicGuard）进行肠鸣音的分割和分类。通过开发基于能量的事件检测算法检测肠鸣音事件，并使用预训练的音频频谱变换器（AST）模型对检测到的声音段进行分类，实现了健康个体和患者中肠鸣音活动的定量评估。
### Conclusion
自动标注方法将手动标注时间减少了大约70%，专家审查结果显示，自动检测的片段中不到12%的部分需要修正。提出的自动化分割和分类系统能够定量评估肠道活动，为临床医生提供客观诊断工具，有助于胃肠功能的诊断，并支持大规模数据集的标注。
## 36. `cs.LG` - 使用卷积Tsetlin机的可解释和硬件高效的5G网络干扰检测 [PDF](https://arxiv.org/pdf/2603.07336), [HTML](https://arxiv.org/abs/2603.07336)
### Authors
Vojtech Halenka,Mohammadreza Amini,Per-Arne Andersen,Ole-Christoffer Granmo,Burak Kantarci
### Background
第五代（5G）网络中的所有应用程序都依赖于稳定的射频（RF）环境来支持移动、自动化和连接智能的关键任务服务。这些应用可能会受到有意干扰或低功率阻塞的威胁，尤其是在这些攻击低于链路层可观察性的情况下。本文研究了使用卷积Tsetlin机（CTM）直接在5G同步信号块（SSB）特征上进行轻量级、可解释和硬件高效型干扰检测的方法。
### Innovation
提出了使用CTM进行5G SSB特征直接处理的轻量级、可解释和硬件高效的干扰检测方法。该方法通过量化输入形成布尔逻辑条款，实现比特级推理并在FPGA架构上确定部署。方法在实际5G测试床上使用空中SSB数据进行实验验证，并将其性能与卷积神经网络（CNN）基线进行了比较。CTM在准确性和内存使用方面均具有显著优势。此外，展示了适用于Zybo™ Z7（Zynq-7000）的紧凑型FPGA设计，并提供了三种部署配置下的资源估算。
### Conclusion
CTM提供了一种实际、可解释和资源高效的替代方案，用于射频（RF）域干扰检测，可在边缘部署中为5G应用提供低延迟和安全性，并为B5G系统奠定了基础。
## 37. `cs.LG` - 先进人工智能的关闭安全阀 [PDF](https://arxiv.org/pdf/2603.07315), [HTML](https://arxiv.org/abs/2603.07315)
### Authors
Vincent Conitzer
### Background
人们普遍担心先进的人工智能可能会因为追求自己的目标而阻止我们关闭它。本文探讨了一种非传统的解决方案，即赋予AI（主要目标）能够被关闭的目标。相关研究还包括Martin等人和Goldstein及Robinson的研究。
### Innovation
提出了一种非传统的解决方案，即赋予先进人工智能被关闭的主要目标，这是一种新的思维方式，旨在解决AI无法被关闭的问题。
### Conclusion
讨论了是否以及什么条件下赋予AI被关闭目标是有利的。该研究为解决先进AI无法被关闭的问题提供了一种新的视角和思路。
## 38. `cs.LG` - 低资源场景中的领域特定机器翻译质量评估 [PDF](https://arxiv.org/pdf/2603.07372), [HTML](https://arxiv.org/abs/2603.07372)
### Authors
Namrata Patil Gurav,Akashdeep Ranu,Archchana Sindhujan,Diptesh Kanojia
### Background
在无参考的评估环境中，机器翻译质量估测(QE)对评估特定领域和资源贫乏语言的翻译质量至关重要。本文研究了英文到印度语族机器翻译在四个领域（医疗、法律、旅游、通用）中的句子级别QE，并且在五个语言对上进行了系统比较。
### Innovation
提出了ALOPE框架，采用低秩适应与退化联结到选定的中间Transformer层的回归头。还引入了最近提出的低秩乘法适应（LoRMA）。结果表明，中间层适应可以一致地改善QE性能，特别是在语义复杂的领域，为更稳健的QE在实际应用场景中的实现提供路径。
### Conclusion
本文展示了ALOPE及其增强版本的性能改进，在高风险领域尤其有效。研究成果公开了代码和领域特定的QE数据集，以支持进一步研究。
## 39. `cs.LG` - 一种多机器人分布式高斯过程模型用于制图 [PDF](https://arxiv.org/pdf/2603.07351), [HTML](https://arxiv.org/abs/2603.07351)
### Authors
Seth Nabarro,Mark van der Wilk,Andrew J. Davison
### Background
本文提出了一种多机器人协作学习全球函数的方法，仅依赖局部经验和计算。背景在于如何在多机器人系统中有效地学习和协作，以便实现全局任务优化。
### Innovation
该方法名为DistGP，通过使用稀疏高斯过程（GP）模型，并利用因子分解以反映任务的多机器人结构，实现了分布式训练。采用高斯信念传播（GBP）算法进行分布训练。该模型在线训练并在动态连通性条件下运行，表现出对稀疏通信的鲁棒性和持续学习能力。
### Conclusion
与基于树结构的高斯过程相比，DistGP在分布式和异步训练中能够获得相当的性能，尽管收敛速度较慢。在与分布式神经网络优化器DiNNO的比较中，DistGP在准确性、通信稀疏性和持续学习能力方面表现更优。
## 40. `cs.LG` - Variational Flow Maps: Make Some Noise for One-Step Conditional Generation [PDF](https://arxiv.org/pdf/2603.07276), [HTML](https://arxiv.org/abs/2603.07276)
### Authors
Abbas Mammadov,So Takao,Bohan Chen,Ricardo Baptista,Morteza Mardani,Yee Whye Teh,Julius Berner
### Background
传统流动图在单一前向传递中能够生成高质量的图像，但与迭代扩散模型不同，它们缺乏明确的采样轨迹，这阻碍了外部约束的结合以进行条件生成和解决逆问题。
### Innovation
提出了变分流动图(Variational Flow Maps, VFM)框架，旨在通过学习适当的初始噪声来改变条件采样的视角，而不是指导采样路径。具体而言，给定一个观察值，VFM通过学习一个噪声适配器模型来输出一个噪声分布，该模型映射到数据空间后生成的样本满足观察值和数据先验。
### Conclusion
在各种逆问题实验中，VFM能够以单步或少数几步生成高度校准的条件样本。特别是在ImageNet上，VFM在保持与替代迭代扩散/流动模型竞争的保真度的同时，极大地加速了采样过程。
