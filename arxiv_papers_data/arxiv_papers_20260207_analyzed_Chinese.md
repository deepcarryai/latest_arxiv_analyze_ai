# 20260207
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - VERA-MH：心理健康领域开源AI安全性评估的可靠性和效度 [PDF](https://arxiv.org/pdf/2602.05088), [HTML](https://arxiv.org/abs/2602.05088)
### Authors
Kate H. Bentley,Luca Belli,Adam M. Chekroud,Emily J. Ward,Emily R. Dworkin,Emily Van Ark,Kelly M. Johnston,Will Alexander,Millard Brown,Matt Hawrilenko
### Background
目前，大量用户使用生成式AI聊天机器人寻求心理支持。尽管这些工具在可用性和规模方面具有优势，但在心理健康AI领域的最大关切点在于这些工具的安全性。为了应对这种需求，提出了VeriEthics and Responsible AI in Mental Health (VERA-MH)评估方法，以提供基于证据的自动化安全性基准。这项研究旨在验证VERA-MH评估在自杀风险检测和应对中的临床效度和可靠性。
### Innovation
该研究创新性地模拟了大规模的对话，使用正式的评分指南来独立评价AI聊天机器人和模拟用户代理的安全行为，以及用户代理的真实性。这种方法通过跨不同临床人员、临床共识和基于语言模型的评估者的评分一致性，验证了VERA-MH评估的有效性和可靠性。
### Conclusion
研究结果表明，VERA-MH评估方法具有临床效度和可靠性，是一个开源的自动化AI安全性评估工具，适用于心理健康领域。未来的研究将重点验证VERA-MH的适用性和稳健性。
## 2. `cs.AI` - MINT: Minimal Information Neuro-Symbolic Tree for Objective-Driven Knowledge-Gap Reasoning and Active Elicitation [PDF](https://arxiv.org/pdf/2602.05048), [HTML](https://arxiv.org/abs/2602.05048)
### Authors
Zeyu Fang,Tian Lan,Mahdi Imani
### Background
联合规划通过语言交互是人类与AI团队合作的关键领域。在开放世界中的规划问题通常涉及到不完整信息和未知因素，如涉及的对象、人类的目标/意图等，这些都会导致在联合规划中存在知识缺口。本文旨在解决AI代理在物体驱动规划中积极寻求人类输入的最优交互策略问题。
### Innovation
提出了Minimal Information Neuro-Symbolic Tree（MINT），用于处理知识缺口的问题，并通过自我对抗学习来优化AI代理的交互策略和查询。具体来说，MINT构建了一个符号树，通过提出可能的人机交互命题，并咨询神经规划策略来估计剩余知识缺口对规划结果的不确定性。最后，利用LLM搜索和总结MINT的推理过程，并定制一组查询以最佳地获取人类输入以实现最佳规划性能。通过考虑具有知识缺口的扩展马尔可夫决策过程族，分析了与主动人类引发相关联的MINT的回报保证。
### Conclusion
在三个现实程度逐步提高的基准测试中，基于MINT的计划能够通过在每个任务中仅提出少量问题来接近专家级回报，同时显著提高了奖励和成功率。
## 3. `cs.AI` - 使用加油和自适应避碰的强化学习优化多碎片对接任务计划 [PDF](https://arxiv.org/pdf/2602.05075), [HTML](https://arxiv.org/abs/2602.05075)
### Authors
Agni Bandyopadhyay,Gunther Waxenegger-Wilfing
### Background
随着地球轨道上碎片越来越多，主动去除碎片（ADR）任务在确保操作安全的同时，需要最大限度地减少轨道碰撞的危险。小型卫星由于灵活性、低成本和机动性而被广泛采用，成为动态任务（如ADR）的理想选择。基于现有的多碎片对接研究，本文提出了一种基于强化学习（RL）的框架，通过整合加油策略、高效的任务规划和自适应轨迹避碰，来优化卫星对接操作，使小型卫星能够有效地进行多碎片对接和清除。
### Innovation
本文提出了一种使用掩码Proximal Policy Optimization (PPO)算法的强化学习框架，该框架能够使RL代理根据实时轨道条件动态调整机动，能够在多碎片对接任务中学习确定高效的对接顺序，优化燃料使用和任务时间，并整合必要的加油停靠。与传统的启发式方法相比，这种方法展示了在减少碰撞风险和提高任务效率方面的优越性能。
### Conclusion
该工作提供了一个可扩展的解决方案，用于规划复杂的多碎片ADR任务，并且这种方法也可应用于其他多目标对接问题的自主航天任务规划。
## 4. `cs.AI` - 在图论已解决问题和未解决问题上的大型语言模型评估：对计算教育的启示 [PDF](https://arxiv.org/pdf/2602.05059), [HTML](https://arxiv.org/abs/2602.05059)
### Authors
Adithya Kulkarni,Mohna Chakraborty,Jay Bagga
### Background
大型语言模型（LLMs）越来越被学生用于探索计算机科学中的高级材料，包括图论。随着这些工具在本科和研究生课程中的集成，理解它们如何可靠地支持数学严谨性变得至关重要。研究者考察了一种LLM在这两类图论问题上的表现：一个已解决且有关线图的优雅性问题，以及一个目前尚未找到解决方法的开放性问题。研究采用了包含八个阶段的真实数学研究流程，包括解释、探索、策略形成和证明构建。
### Innovation
研究使用了包含八个阶段的真实数学研究流程来评估LLM的表现，涵盖了解释、探索、策略形成和证明构建等环节。对于已解决问题，LLM表现良好，提供了正确的定义，识别了相关结构，并回想起了适当的结果。对于未解决问题，LLM生成了连贯的解释和可信的探索策略，但未能推进到找到解决方案的地步。研究结果表明，LLMs在探索既有知识方面表现出色，但在需要新颖数学洞察或关键结构推理的任务上仍有限制。
### Conclusion
这些发现表明，LLMs在探索已确立的内容时可以提供帮助，但在要求新颖的数学洞察或关键结构推理的任务上仍有限制。对于计算教育而言，这一区分强调了指导学生使用LLMs进行概念性探索的重要性，同时依赖独立验证和严谨论据来解决正式问题的重要性。
## 5. `cs.AI` - DeepRead: 具有文档结构感知的推理以增强自主搜索 [PDF](https://arxiv.org/pdf/2602.05014), [HTML](https://arxiv.org/abs/2602.05014)
### Authors
Zhanli Li,Huiwen Tian,Lvzhou Luo,Yixuan Cao,Ping Luo
### Background
随着工具使用和自主大型语言模型（LLMs）的快速进步，检索增强生成（RAG）正在从单次被动检索演变到多轮次、基于决策的证据获取。尽管在开放领域设置中取得了优异的结果，但现有的自主搜索框架通常将长文档视为扁平的片段集合，未能充分利用文档固有的先验信息，如层级组织和序列话语结构。已有研究未充分利用这些先验信息，特别是在处理长文档时造成了信息拆分与重组的困难。
### Innovation
本文提出了一个称为 DeepRead 的结构感知多轮次文档推理代理 DeepRead。该代理利用基于 LLM 的 OCR 模型将 PDF 转换为保持标题和段落边界的结构化 Markdown，然后在段落级别对文档进行索引，并为每个段落分配一个坐标式元数据键，编码其部分身份和部分内的顺序。DeepRead 通过赋予 LLM 两种互补工具（一个用于定位相关段落并显示其结构坐标（轻量级扫描上下文）的检索工具，以及一个用于在指定部分和段落范围内连续、按序读取的读取部分工具），增强了自主搜索效果。实验结果表明，DeepRead 在文档问题回答中的表现优于 Search-o1 类型的自主搜索框架，检索与阅读工具之间存在协同效应。
### Conclusion
细粒度的行为分析表明，DeepRead 的行为模式类似于人样的“先定位再阅读”行为。研究验证了 DeepRead 通过结合检索和阅读能力，增强了自主搜索的效能，提供了更高效的文档内容理解和处理方法。
## 6. `cs.AI` - 朝向可减少的不确定性建模以实现可靠的大型语言模型代理 [PDF](https://arxiv.org/pdf/2602.05073), [HTML](https://arxiv.org/abs/2602.05073)
### Authors
Changdae Oh,Seongheon Park,To Eun Kim,Jiatong Li,Wendi Li,Samuel Yeh,Xuefeng Du,Hamed Hassani,Paul Bogdan,Dawn Song,Sharon Li
### Background
大型语言模型（LLM）的安全保障是日常LLM应用的关键组成部分。目前大多数不确定性量化（UQ）研究仍集中在单一回合的问题回答上。然而，随着LLM代理在高度复杂任务中的部署增加，UQ研究需要转向互动性更强的真实应用场景。
### Innovation
论文提出了首个通用的代理不确定性量化（UQ）的总体框架，该框架涵盖了广泛现有的UQ设置。提出了“条件下的不确定性减少过程”这一新颖观点，通过强调代理行为的互动性，明确建模代理轨迹中的可减少不确定性。
### Conclusion
该研究提出了可减少不确定性建模的概念框架，提供了设计LLM代理UQ的实用指导。最后，讨论了在前沿LLM开发和特定领域应用中的实际意义，并指出了未解决的问题。
## 7. `cs.AI` - 人工智能的奇异智能：反对线性智力模型 [PDF](https://arxiv.org/pdf/2602.04986), [HTML](https://arxiv.org/abs/2602.04986)
### Authors
Kendra Chilson,Eric Schwitzgebel
### Background
本文背景在于对苏珊·施耐德对人工智能进步线性模型的批判，强调人工智能的智力表现可能不符合我们熟悉的模式，甚至在某些领域可以展示出超乎常人的能力，而在另一些领域则可能表现得不如普通人。
### Innovation
本文提出了两个创新概念——“熟悉智能”和“奇异智能”。作者发展并捍卫了一个非线性的智能模型，认为“通用智能”不是一个统一的能力，而是能够在多种环境和领域中实现多种目标的能力，这种能力不能简单地归结为单一的线性量度。此外，文章还探讨了这些观点对对手测试评估人工智能能力的影响。
### Conclusion
如果人工智能表现出奇异智能，那么即使是最强大的系统有时也可能会在看似明显的问题上失败。在非线性的智能模型下，单靠看似优秀的表现并不能表明系统在广泛领域内的出色能力。同样，单一领域中的出色表现也不能推断出系统在其他领域的广泛能力。
## 8. `cs.AI` - GAMMS: 基于图的对抗性多智能体建模模拟器 [PDF](https://arxiv.org/pdf/2602.05105), [HTML](https://arxiv.org/abs/2602.05105)
### Authors
Rohan Patil,Jai Malegaonkar,Xiao Jiang,Andre Dion,Gaurav S. Sukhatme,Henrik I. Christensen
### Background
随着智能系统和多智能体协调在实际应用中的重要性不断增加，对可扩展且易于访问的仿真工具的需求也在增长。现有的高保真模拟器虽然功能强大，但计算成本昂贵，并不适合快速原型设计或大规模智能体部署。
### Innovation
GAMMS (Graph based Adversarial Multiagent Modeling Simulator) 提供了一个轻量级但具有扩展性的仿真框架，旨在支持在可图表示的环境中快速开发和评估智能体行为，特别强调了五个核心目标：可扩展性、易用性、集成优先架构、快速可视化反馈以及现实世界连接。它能够高效模拟复杂领域（如城市道路网络和通信系统），支持与外部工具（例如机器学习库、规划求解器）的集成，并提供内置可视化功能，配置简便。GAMMS 支持不同类型策略的智能体，包括启发式、优化和基于学习的智能体，甚至包括使用大型语言模型的智能体。
### Conclusion
通过降低研究人员的入门门槛并在标准硬件上实现高性能模拟，GAMMS 促进多智能体系统、自主规划和对抗性建模领域的实验和创新。GAMMS 是开源的，并可通过此链接 <this https URL> 获取。
## 9. `cs.AI` - 理解大规模语言模型评估者行为：一种针对商户风险评估的结构化多评估者框架 [PDF](https://arxiv.org/pdf/2602.05110), [HTML](https://arxiv.org/abs/2602.05110)
### Authors
Liang Wang,Junpeng Wang,Chin-chia Michael Yeh,Yan Zheng,Jiarui Sun,Xiran Fan,Xin Dai,Yujie Fan,Yiwei Cai
### Background
大规模语言模型（LLMs）在评估推理质量方面越来越普遍，但它们在支付风险评估等情境下的可靠性和偏见尚未得到充分理解。
### Innovation
本文提出了一种结构化的多评估者框架，用于评估LLM在基于商户类别代码（MCC）的商户风险评估中的推理。该框架结合了五项准则评分表和蒙特卡洛评分法来评估推理质量与评估者稳定性。此外，引入了一个共识偏差度量方法，该方法通过将每个评估者的评分与所有其他评估者的平均分进行比较，以消除循环性，从而提供一个理论支撑的自我评估和跨模型偏差的度量。
### Conclusion
研究成果揭示了显著的异质性：GPT-5.1和Claude 4.5 Sonnet显示出负的自我评估偏差，而Gemini-2.5 Pro和Grok 4则显示正的偏差。这些偏差在匿名化条件下有所减轻。26名支付行业专家的评价结果显示，LLM评估者对人类共识的评分平均高出0.46分，并且GPT-5.1和Claude 4.5 Sonnet的负偏差反映了与人类判断的更接近性。基于支付网络数据的真实验证表明，四个模型在统计意义上表现出显著的一致性，证实了该框架能够捕获真正的质量。总体而言，框架为评估LLM作为判断系统的支付风险工作流程提供了一个可重复的基础，并强调了操作金融环境中需要具备偏见意识的协议。
## 10. `cs.AI` - 基于学习的任务规划在主动太空碎片清除中的鲁棒性和适应性评估 [PDF](https://arxiv.org/pdf/2602.05091), [HTML](https://arxiv.org/abs/2602.05091)
### Authors
Agni Bandyopadhyay,Günther Waxenegger-Wilfing
### Background
在低地轨道进行主动太空碎片清除（ADR）任务规划，需要平衡效率、适应性和燃料以及任务持续时间等严格约束。研究对比了三种针对受限多碎片会合问题的规划者，分别是针对固定任务参数训练的标准蒙特卡洛树搜索（MCTS），在不同任务约束下训练提高鲁棒性的域随机化蒙特卡洛树搜索（MCTS），以及名义上的掩码强化学习（PPO）策略。
### Innovation
提出了一种针对受限多碎片会合问题的综合评估方法，通过对比三种不同类型的规划者，分析了它们的鲁棒性和适应性。这种方法不仅考虑了传统搜索方法（MCTS）在约束变化下的鲁棒性，还研究了学习型方法（PPO）在不同条件下的性能变化。
### Conclusion
研究发现，名义上的PPO在条件匹配训练时表现最佳，但在分布变化时性能急剧下降；而域随机化的PPO表现出更好的适应性，仅在名义性能上有适度损失。MCTS在处理约束变化方面表现出色，但由于在线重规划导致的计算时间增加，使其在性能上略逊于域随机化的PPO。研究强调了学习策略的速度与基于搜索方法的适应性的权衡，并建议结合训练时的多样性与在线规划可能是未来鲁棒性ADR任务规划的一种有前景的方法。
## 11. `cs.CL` - 当共享知识有害：模型融合中的谱过积累 [PDF](https://arxiv.org/pdf/2602.05536), [HTML](https://arxiv.org/abs/2602.05536)
### Authors
Yayuan Li,Ze Peng,Jian Zhang,Jintao Guo,Yue Duan,Yinghuan Shi
### Background
现有的模型合并方法主要集中在解决任务更新间的冲突，但未能解决共享知识过量累积的问题。当任务共享对齐的方向(即，有重叠的奇异向量)时，简单的线性组合会不断积累这些方向，导致奇异值膨胀，从而使合并模型偏向共享子空间。
### Innovation
提出了Singular Value Calibration (SVC)，这是一种无需训练和数据的后处理方法，用于量化子空间重叠并重新调整膨胀的奇异值，以恢复平衡的特征谱。SVC不仅在视觉和语言基准测试中提高了完成稳健的合并基线，还通过仅调整奇异值使任务算术的表现提高了13.0%。
### Conclusion
SVC在整个模型合并基准测试中一致地改进了强劲的合并基线，并达到了最先进的性能。此外，通过仅调整奇异值，SVC还提高了任务算术的性能。
## 12. `cs.CL` - AgentXRay: 通过工作流重构使代理系统透明化 [PDF](https://arxiv.org/pdf/2602.05353), [HTML](https://arxiv.org/abs/2602.05353)
### Authors
Ruijie Shi,Houbin Zhang,Yuecheng Han,Yuheng Wang,Jingru Fan,Runde Yang,Yufan Dang,Huatao Li,Dewen Liu,Yuan Cheng,Chen Qian
### Background
大型语言模型在复杂问题解决方面表现出强大的能力，但许多代理系统仍然难以解释和控制，因为它们内部的工作流程不透明。尽管一些框架提供了明确的协作架构，但许多部署的代理系统依然对用户作为黑箱操作。我们通过引入代理工作流重构（AWR）任务来解决这个问题，该任务旨在仅使用输入-输出访问重构一个解释型的替代工作流来近似一个黑箱系统。
### Innovation
我们提出了AgentXRay，一种基于搜索的框架，将AWR问题表述为一个离散代理角色和工具调用在链式工作流空间中的组合优化问题。与模型蒸馏不同，AgentXRay生成可编辑的白盒工作流，同时保持可观察和基于输出的代理相似度指标，而无需访问模型参数。为了导航庞大的搜索空间，AgentXRay运用蒙特卡洛树搜索，并通过一个基于评分的红黑剪枝机制动态整合代理质量与搜索深度。
### Conclusion
在不同领域的实验表明，与未剪枝的搜索相比，AgentXRay在代理相似度方面表现更好，减少了令牌消耗，并在固定迭代预算下使工作流探索更为深入。
## 13. `cs.CL` - 通过流匹配引导大型推理模型实现简洁推理 [PDF](https://arxiv.org/pdf/2602.05539), [HTML](https://arxiv.org/abs/2602.05539)
### Authors
Yawei Li,Benjamin Bergner,Yinghan Zhao,Vihang Prakash Patil,Bei Chen,Cheng Wang
### Background
大型推理模型（LRMs）在复杂的推理任务中表现出色，但由于过度冗长的输出，其效率受到了阻碍。先前的指导方法试图通过应用单一的全局向量来解决这一问题，这种方法基于线性表示的假设。这种方法对于复杂的非线性关系的处理能力有限。
### Innovation
本文提出了FlowSteer，这是一种非线性指导方法，通过学习从冗长推理到简洁推理的完整变换来超越均匀的线性平移。这种变换是通过作为速度场的流匹配学习得到的，能够实现对模型推理过程的精准、输入依赖的控制，从而使得指导后的表示与简洁推理的激活分布对齐，产生更为紧凑的推理结果。
### Conclusion
FlowSteer在多种推理基准测试中展示了强大的任务性能和更高的令牌效率，相比先进的推理时基线模型更为有效。我们的工作表明，使用生成技术建模完整的分布传输提供了一种更有效的且更为原则的控制LRMs的基础。
## 14. `cs.CL` - BhashaSetu：从高资源语言到极端低资源语言的跨语言知识迁移 [PDF](https://arxiv.org/pdf/2602.05599), [HTML](https://arxiv.org/abs/2602.05599)
### Authors
Subhadip Maji,Arnab Bhattacharya
### Background
尽管自然语言处理取得了显著进展，但开发有效的低资源语言系统仍然是一个艰巨的挑战，因为数据稀缺和不足的资源导致其性能通常远低于高资源语言对应的系统。跨语言知识迁移作为一个有前景的方法，通过利用高资源语言的资源来解决这一挑战。
### Innovation
本文提出了一种新颖的方法GETR（Graph-Enhanced Token Representation）用于跨语言知识迁移，并且引入了两种基准方法，即在隐藏层中的增强和通过标记翻译进行的标记嵌入转移。实验结果表明，基于图的GETR方法显著优于现有的多语言和跨语言基准方法，分别在Mizo和Khasi等极度低资源语言的词性标注任务中实现13个百分点的改进，以及分别在Marathi、Bangla和Malayalam等模拟低资源语言的情感分类任务和命名实体识别任务中实现20和27个百分点的宏F1改进。
### Conclusion
本文详细分析了知识转移机制，并确定了成功知识转移的一些关键因素。
## 15. `cs.CL` - 多领域工具检索 [PDF](https://arxiv.org/pdf/2602.05366), [HTML](https://arxiv.org/abs/2602.05366)
### Authors
Yichen Tang,Weihang Su,Yiqun Liu,Qingyao Ai
### Background
随着可用工具的规模不断扩大，如何有效检索这些工具成为大型语言模型（LLMs）与现实环境互动和解决复杂任务的关键。现有方法通常将工具检索视为传统的即兴检索任务，通过匹配用户查询与整个工具文档进行。然而，这种传统方法存在三个主要限制：工具文档的不完整性及结构一致性问题；查询与技术文档之间显著的语义和粒度不匹配；以及工具功能的多维度特性，这包括功能、输入约束和输出格式，这些特性的格式和重要性各不相同。
### Innovation
本文提出了多领域工具检索框架，该框架采用细粒度、多领域建模方式，旨在将用户意图与工具表示对齐。实验结果显示，该框架在五个数据集和混合基准上表现出SOTA性能，并展示了更好的泛化能力和稳健性。
### Conclusion
多领域工具检索框架通过细粒度、多领域建模解决了传统方法的限制，并在实验中展示了优越的性能，增强了大型语言模型解决现实世界任务的能力。
## 16. `cs.CL` - H-AdminSim：带有FHIR集成的多智能体模拟器，用于现实的医院行政工作流 [PDF](https://arxiv.org/pdf/2602.05407), [HTML](https://arxiv.org/abs/2602.05407)
### Authors
Jun-Min Lee,Meong Hi Son,Edward Choi
### Background
医院行政管理部门处理一系列操作任务，在大型医院中，每天处理的请求数量超过10,000个，这导致了对基于LLM的自动化处理的技术需求增加。此前的研究主要集中在患者-医师互动或零碎的行政子任务上，未能完全捕捉到实际行政工作流程的复杂性。
### Innovation
提出了一种名为H-AdminSim的全面的端到端模拟框架，该框架结合了现实数据生成与基于多智能体的医院行政工作流程模拟。通过FHIR集成，H-AdminSim提供了一个统一且可互操作的环境，用于跨异构医疗机构测试行政工作流程，从而作为基于LLM的行政自动化标准试验平台，以评估其可行性和性能。
### Conclusion
该模拟框架通过详细的评判标准对LLM进行量化评估，使得系统性比较成为可能。H-AdminSim提供了一个标准化的测试平台，用于检验LLM驱动的行政自动化在不同医疗机构中的实用性和效能。
## 17. `cs.CL` - SciDef: 使用大型语言模型自动化学术文献中的定义提取 [PDF](https://arxiv.org/pdf/2602.05413), [HTML](https://arxiv.org/abs/2602.05413)
### Authors
Filip Kučera,Christoph Mandl,Isao Echizen,Radu Timofte,Timo Spinde
### Background
随着科学文献的大量增加，找到与关键词相关的定义变得愈加困难。现有的研究面临挑战。
### Innovation
SciDef 管道基于大型语言模型（LLM），用于自动提取定义。通过测试多种提示策略和使用 underwear-based 方法评估提取结果，展示了多层次提示和 DSPy-优化提示的有效性，证明了 LLM 能够从科学文献中提取大量定义。
### Conclusion
LLMs 可以从科学文献中提取大量定义，但未来需要关注提取相关定义的问题。研究代码和数据可在指定的 URL 获得。
## 18. `cs.CL` - ArkTS-CodeSearch: 开源ArkTS代码集用于代码检索 [PDF](https://arxiv.org/pdf/2602.05550), [HTML](https://arxiv.org/abs/2602.05550)
### Authors
Yulong He,Artem Ermakov,Sergey Kovalchuk,Artem Aliev,Dmitry Shalymov
### Background
ArkTS是OpenHarmony生态系统中的核心编程语言，但由于缺乏公开的数据集和评估基准，针对ArkTS代码智能的研究受到了限制。该研究基于开源代码库构建了一个大规模的ArkTS数据集，用于代码检索和代码评估任务。
### Innovation
设计了一个单一搜索任务，使用自然语言注释检索相应的ArkTS函数；评估了所有现有的开源代码嵌入模型，并在单个搜索任务上进行微调，以提高ArkTS代码理解能力；建立了第一个系统化的ArkTS代码检索基准。
### Conclusion
本工作建立了第一个系统化的ArkTS代码检索基准。数据集和微调后的模型将被公开发布，可以在指定的URL处获取。
## 19. `cs.CL` - 统一的多模态框架：基于数据集构建和模型诊断的成釉细胞瘤 [PDF](https://arxiv.org/pdf/2602.05515), [HTML](https://arxiv.org/abs/2602.05515)
### Authors
Ajo Babu George,Anna Mariam John,Athul Anoop,Balu Bhasuran
### Background
人工智能（AI）在面部和颌骨疾病诊断中的应用需要结构化和高质量的多模态数据集。已有的资源在成釉细胞瘤覆盖范围和格式一致性方面存在不足，不能直接用于模型训练。
### Innovation
本文提出了一个专为成釉细胞瘤设计的新多模态数据集，整合了注释的放射学、病理学和口腔临床图像，并从病历报告中提取结构化数据。利用自然语言处理技术提取临床相关特征，以及对图像数据进行特定领域预处理和增强，建立了多模态深度学习模型，用于成釉细胞瘤变体分类和行为模式评估，模型部署时可接受临床输入以增强个性化推理。
### Conclusion
定量评估显示，模型的分类准确率从46.2%提高到65.9%，检测异常组织的F1分数从43.0%提高到90.3%。与资源如MultiCaRe相比，本研究通过提供一个稳健的数据集和一个适应性强的多模态AI框架，提升了患者的个性化决策支持水平。
## 20. `cs.CL` - MerNav：一种适用于零样本对象目标导航的高可泛化记忆-执行-审查框架 [PDF](https://arxiv.org/pdf/2602.05467), [HTML](https://arxiv.org/abs/2602.05467)
### Authors
Dekang Qi,Shuang Zeng,Xinyuan Chang,Feng Xiong,Shichao Xie,Xiaolong Wu,Mu Xu
### Background
Visual Language Navigation (VLN) 是实现体态智能的基础能力之一，但目前的方法在成功率 (SR) 和泛化能力上仍有诸多不足。监督微调 (SFT) 方法通常能实现更高的 SR，而无需训练 (TF) 方法则在泛化能力上表现更佳，但很难两方面都做到出色。因此，提出了一个记忆-执行-审查框架 (MEF) 来解决这些问题。
### Innovation
该框架包括三部分：层级记忆模块以提供信息支持，执行模块用于常规决策和行动，审查模块用于处理异常情况并纠正行为。在零样本对象目标导航任务上，该框架在四个数据集中分别相对于所有基准方法实现了 7% 和 5% 的绝对 SR 提升，在 ZS 设置下，相较 HM3D_v0.1 和开放词汇数据集 HM3D_OVON 上实现了 8% 和 6% 的提升。此外，该方法在 MP3D 和 HM3D_OVON 数据集上不仅优于所有 TF 方法，还超过了所有 SFT 方法，在 SR 和泛化能力上均表现优异。
### Conclusion
MerNav 框架在零样本对象目标导航任务上表现优异，成功实现了同时提升成功率和泛化能力的目标，特别是在 HM3D_OVON 和 MP3D 数据集上，该方法在这一关键问题上实现了引领性的突破。
## 21. `cs.SE` - 大型语言模型在软件文档与建模中的应用：文献综述与发现 [PDF](https://arxiv.org/pdf/2602.04938), [HTML](https://arxiv.org/abs/2602.04938)
### Authors
Lukas Radosky,Ivan Polasek
### Background
生成式人工智能引起了广泛关注，尤其是在大型语言模型引入后。这种模型被用于解决多种软件工程任务，尤其是处理软件文档和理解编程语言方面表现出色。
### Innovation
本文综述了大型语言模型在软件工程文档和建模任务中的应用，对来自四大主要领域的相关文章进行了分析，并组织归类，提供了关于所使用提示技术、评估指标、人类评估方法和主要数据集的概述。
### Conclusion
本文总结了大型语言模型在软件工程中的文档和建模任务应用现状，指出了当前研究的主要趋势和发现，并为未来研究提供了参考。
## 22. `cs.SE` - TestMigrationsInPy: 一个从Unittest到Pytest的测试迁移数据集 [PDF](https://arxiv.org/pdf/2602.05122), [HTML](https://arxiv.org/abs/2602.05122)
### Authors
Altino Alves,Andre Hora
### Background
Unittest和pytest是Python中最受欢迎的测试框架，其中pytest提供了更简单的断言、可重用的固定装置以及更好的互操作性。由于这些优势，Python生态系统中的多个项目已经从unittest迁移到了pytest。虽然pytest可以运行unittest测试，使得迁移过程可以逐渐进行，但整个迁移过程依然耗时较长。为了支持这一过程，需要自动化的解决方案来帮助完成迁移动作。
### Innovation
本文提出了TestMigrationsInPy数据集，这是一个包含923个开发人员实际执行的从Unittest到pytest迁移的测试集合。此数据集不仅作为未来Python框架迁移研究中解决方案验证的真实依据，还提供了关于迁移类型的详细信息（例如断言或固定装置的变化），这有助于研究者验证和测试各种迁移解决方案的有效性。
### Conclusion
TestMigrationsInPy数据集以公开的形式提供，可供研究人员使用，以支持和评估Python框架迁移过程中的自动化解决方案。
## 23. `cs.SE` - 全面评估基于AI的自动化功能的必要性 [PDF](https://arxiv.org/pdf/2602.05157), [HTML](https://arxiv.org/abs/2602.05157)
### Authors
Alireza Abbaspour,Shabin Mahadevan,Kilian Zwirglmaier,Jeff Stafford
### Background
传统的安全分析方法将质量管理（QM）组件排除在安全性评估之外，而这类组件并非通常认为的安全相关部分。然而，随着人工智能（AI）技术的进步，特别是其在自动驾驶功能中的应用，这些质量管理组件可能成为SOTIF（安全旨在功能）相关风险的因素。因此，顺应新兴的AI安全标准，如ISO/PAS 8800的规定，需要重新评估这些组件的安全考虑。案例研究表明，即使是归类为质量管理的组件，其缺陷也可能导致危险的功能行为，进而对驾驶安全构成重大影响。
### Innovation
本文提出的创新在于强调了全面的FuSa（功能安全）、SOTIF（安全旨在功能）和AI标准驱动的方法的重要性，以识别并缓解AI组件中的风险。这填补了现有安全评估框架中的空白，特别是在AI组件哪些情况下会对驾驶安全性构成重大影响的识别上。
### Conclusion
该研究得出结论，当前的安全框架需要进行修订，以应对由AI带来的不断变化的挑战，从而在整个组件分类中保证全面的安全性。这将有助于确保在多标准下的综合安全性验证。
## 24. `cs.SE` - 应用于机器学习驱动系统的以需求为中心的敏捷管理方法 [PDF](https://arxiv.org/pdf/2602.05042), [HTML](https://arxiv.org/abs/2602.05042)
### Authors
Lucas Romao,Luiz Xavier,Júlia Condé Araújo,Marina Condé Araújo,Ariane Rodrigues,Marcos Kalinowski
### Background
机器学习（ML）驱动的系统挑战了传统的需求工程（RE）和敏捷管理，因为这些系统依赖于数据、实验和不确定性模型的行为。现有的RE和敏捷实践仍然没有很好地集成，也未能充分适应这些特点。背景强调了这种情况对未来系统开发的影响。
### Innovation
本文介绍了RefineML方法，这是一种专注于需求的连续和敏捷方法，用于ML驱动系统的持续细化。该方法整合了定制的ML规范和敏捷管理方法，并结合了系统综述中得出的最佳实践。此外，它在一个行业-学术合作项目中得到了应用，该项目涉及里约热内卢天主教大学（PUC-Rio）和巴西网络安全公司EXA。
### Conclusion
应用问卷调查和半结构化访谈进行评估，结果显示RefineML具有很高的可用性和整体接受度。受访者认为RefineML能够提高沟通和早期可行性评估，并使ML和软件工作并行治理，使模型在项目整体软件演进过程中不断细化。然而，该方法仍存在一些限制，特别是在将ML关注点转化为敏捷需求以及估计ML工作量方面存在困难。
## 25. `cs.SE` - 异常行为：它们的测试频率有多高？ [PDF](https://arxiv.org/pdf/2602.05123), [HTML](https://arxiv.org/abs/2602.05123)
### Authors
Andre Hora,Gordon Fraser
### Background
异常允许开发人员处理预期会发生但不频繁的错误情况。理想情况下，优质的测试套件应同时测试常规和异常行为，以捕捉更多错误并防止回归。当前的研究分析了传播到测试的异常，但尚未探讨那些未达到测试的异常。本研究旨在通过实证研究探索现实世界系统中异常行为的测试频率，考察那些传播到测试的异常和未达到测试的异常。
### Innovation
研究通过运行测试套件的仪器化版本，监测其执行情况并收集运行时异常信息，分析了25个Python系统的测试套件，涵盖5,372个执行方法、17.9兆次调用和140万个引发的异常。研究发现21.4%的方法在运行时会引发异常。在引发异常的方法中，平均每10次调用中，有1次会涉及异常行为。近80%的引发异常的方法是偶尔发生的，但约20%的方法是频繁发生的。
### Conclusion
研究提出了对未来研究和实际操作的建议。建议开发新型工具来支持异常行为的测试和优化昂贵的try/except语句块。研究也强调了异常产生行为不一定代表异常或罕见的情况。
## 26. `cs.SE` - 编程语言重要吗？ fuzzing 漏洞检测的实证研究 [PDF](https://arxiv.org/pdf/2602.05312), [HTML](https://arxiv.org/abs/2602.05312)
### Authors
Tatsuya Shirai,Olivier Nourry,Yutaro Kashiwa,Kenji Fujiwara,Hajimu Iida
### Background
 fuzzing 已经成为一种流行的自动化检测漏洞和bug的技术，通过生成意外输入。近年来，fuzzing 过程被集成到持续集成的工作流中（即持续 fuzzing），这是短小、频繁的测试周期的重要组成部分。尽管这种技术得到了广泛应用，但现有的研究并没有探讨不同编程语言之间持续 fuzzing 效果的变化情况。
### Innovation
本研究进行了大规模的跨语言分析，旨在研究 fuzzing bug 的特性和检测效率在不同编程语言之间的差异。研究分析了559个开源项目的61,444个 fuzzing bug 和999,248个构建，通过将这些项目按照主要编程语言进行分类。结果揭示了C++和Rust在 fuzzing bug 检测频率上的优势，以及Rust和Python较低的漏洞比例但更容易揭露关键漏洞的现象。此外，研究还发现了不同语言之间崩溃类型的不同，并且在Go中频繁出现无法复现的 bug，但在Rust中极为罕见。同时，Python取得了更高的补丁覆盖率，但在时间检测方面表现较弱。
### Conclusion
本研究证明了 fuzzing 行为和效果受到语言设计的强烈影响，为编程语言感知的 fuzzing 策略和工具开发提供了重要的见解。
## 27. `cs.SE` - PatchGuru: 从自然语言文件推断补丁或acles的大型语言模型 [PDF](https://arxiv.org/pdf/2602.05270), [HTML](https://arxiv.org/abs/2602.05270)
### Authors
Thanh Le-Cong,Bach Le,Toby Murray,Michael Pradel,Cristian Cadar
### Background
随着软件系统的演进，补丁可能会意外地改变程序的行为。验证补丁的预期语义具有挑战性，因为回归测试不完整且补丁意图的描述大多是非正式的、非执行的自然语言形式。当前主要依赖代码审查和回归测试，这些方法不总是能准确地验证补丁的语义。因此，需要一种自动化方法从自然语言描述中推断出补丁或acles，以辅助验证补丁的意图。
### Innovation
PatchGuru 是第一个从实际提交请求（PRs）中自动推断出可执行补丁规格的技术。该方法利用大型语言模型（LLMs）从自然语言文件中提取开发者的意图，并生成补丁或acles。这些或acles通过蕴含抽象的、实用的规格（以运行时断言形式表达的对比程序）重点强调补丁相关的行为，能够实现自动验证并支持跨版本属性。此外，PatchGuru 能够迭代改进推断出的或acles，识别违反预期的行为，通过自我审查过滤不一致内容，并生成缺陷报告。
### Conclusion
PatchGuru 在对四个广泛使用的开源 Python 项目中的近期 PR 进行评估后，报告了 39 个警告，其中 24 个确实是真阳性，包括 12 个先前未知的缺陷，其中有 11 个缺陷已被开发者修复。相比 Testora，PatchGuru 能检测到更多的缺陷（24 vs. 7），并提高了精度从 0.32 到 0.62。PatchGuru 的平均成本为每 PR 8.9 分钟和 0.07 美元。这些结果表明，PatchGuru 可以补充代码审查和回归测试，提供可执行的文档并实现补丁意图的自动化验证。
## 28. `cs.SE` - ASA: 基于激活调整的工具调用领域适应 [PDF](https://arxiv.org/pdf/2602.04935), [HTML](https://arxiv.org/abs/2602.04935)
### Authors
Youjin Wang,Run Zhou,Rong Fu,Shuaishuai Cao,Hongwei Zeng,Jiaxuan Lu,Sicheng Fan,Jiaqiao Zhao,Liangming Pan
### Background
为大规模语言模型（LLM）在实际场景中的部署，核心挑战并非工具本身的应用，而是高效适应迅速变化的工具集、API和协议。针对不同领域频繁跨域训练和维护的重复使用LoRA或SFT会带来指数级增长的成本，而提示或模式方法则在分布变化和复杂接口下表现脆弱。
### Innovation
提出了轻量级、推理时、无需训练的机制——激活调整适配器（ASA），该机制通过读取中间激活的路由信号，使用超轻量级路由器产生适应性控制强度，以实现精确的领域对齐。实验证明，ASA 在多个模型规模和领域内实现了与LoRA可比拟的适应性，同时具有显著更低的开销和跨模型的强可迁移性，使得它适合用于频繁接口变化的多领域工具生态系统，具有较强的稳健性、可扩展性和效率。
### Conclusion
ASA 作为一种轻量级机制，在保持高效适应性的前提下，显著降低了开销和提升了跨模型的可迁移性，使其成为应对快速变化的工具环境的理想方案。
## 29. `cs.SE` - EGSS: Entropy-guided Stepwise Scaling for Reliable Software Engineering [PDF](https://arxiv.org/pdf/2602.05242), [HTML](https://arxiv.org/abs/2602.05242)
### Authors
Chenhui Mao,Yuanting Lei,Zhixiang Wei,Ming Liang,Zhixiang Wang,Jingxuan Xu,Dajun Chen,Wei Jiang,Yong Li
### Background
Agentic Test-Time Scaling (TTS) 已在复杂的软件工程任务，如代码生成和错误修复中展现了卓越的性能（SOTA），但由于显著的计算开销，其实际应用受到了限制。主要挑战包括：（1）部署大型模型需要高昂的成本；（2）缺乏可靠的机制来选择最佳候选解决方案，最终限制了性能的提升。
### Innovation
我们提出了Entropy-Guided Stepwise Scaling (EGSS)，这是一种新型的TTS框架，通过熵导向的自适应搜索和稳健的测试套件扩充动态平衡效率和效果。实验结果显示，EGSS 在所有评估模型中的一致提升了5-10％的性能，特别是在与GLM-4.6结合使用时，EGSS 达到了开源大模型的SOTA。此外，EGSS 比现有TTS方法减少超过28%的推理时间标记使用量，同时提高了计算效率和效果。
### Conclusion
在SWE-Bench-Verified上的大量实验表明，EGSS 有助于提升性能并减少推理时间的标记使用，同时在多个方面实现了双重改进，是可靠软件工程的有效方法。
## 30. `cs.SE` - 机器学习组件的质量模型 [PDF](https://arxiv.org/pdf/2602.05043), [HTML](https://arxiv.org/abs/2602.05043)
### Authors
Grace A. Lewis,Rachel Brower-Sinning,Robert Edman,Ipek Ozkaya,Sebastián Echeverría,Alex Derr,Collin Beaudoin,Katherine R. Maffey
### Background
尽管机器学习（ML）采用率和算法有所进步，但仍有很多原型未进入生产阶段，测试主要集中在模型性能等模型属性上，而忽略了系统要求，如吞吐量、资源消耗或健壮性。这种有限的测试观点导致了模型整合、部署和运行时的失败。传统软件开发中，ISO 25010等质量模型为评估软件质量、定义质量要求和提供利益相关者沟通的共同语言提供了一种广泛使用的结构化框架。ISO 25059则定义了专门针对AI系统的质量模型。但此标准的问题在于它将系统属性与ML组件属性相结合，这对模型开发者来说并无帮助，因为许多系统属性在组件层面难以评估。
### Innovation
本文提出了一种针对ML组件的质量模型，旨在引导需求提取和谈判，为ML组件开发者和系统利益相关者提供共同的词汇，使其能够在系统衍生要求的基础上达成一致并集中测试努力。该质量模型通过一项调查得到了肯定与认可，并成功集成到一个开源工具中，用于ML组件的测试和评估，展示了其实用性。
### Conclusion
所提出的质量模型已被验证有效，并成功应用于开源工具，以实现ML组件的测试和评估，突显了其实用价值。
