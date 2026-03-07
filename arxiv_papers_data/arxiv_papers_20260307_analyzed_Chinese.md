# 20260307
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 通过多智能体系统发现数学概念 [PDF](https://arxiv.org/pdf/2603.04528), [HTML](https://arxiv.org/abs/2603.04528)
### Authors
Daattavya Aggarwal,Oisin Kim,Carl Henrik Ek,Challenger Mishra
### Background
数学概念是在实验、证明尝试和反例的相互作用过程中形成的。本文基于这一观察，提出了一种多智能体模型，用于基于计算的数学发现。
### Innovation
该系统能够自己提出假设，并尝试证明这些假设，通过反馈和演化数据分布做出决策。该系统以欧拉多面体猜想的历史和文献中的开放挑战为灵感，自主从多面体数据和线性代数知识中学习同调概念。
### Conclusion
实验是重要的验证手段，通过删减实验（ablation studies）和统计测试，证实了优化合适的局部过程组合可以产生令人惊讶地符合数学兴趣的标准。
## 2. `cs.AI` - ECG-MoE：专家混合的心电图基础模型 [PDF](https://arxiv.org/pdf/2603.04589), [HTML](https://arxiv.org/abs/2603.04589)
### Authors
Yuhao Xu,Xiaoda Wang,Yi Wu,Wei Jin,Xiao Hu,Carl Yang
### Background
心电图（ECG）分析对心脏诊断至关重要，但现有的基础模型往往无法捕捉不同临床任务所需的时间周期性和多样化特征。
### Innovation
提出了一种混合架构ECG-MoE，结合了多模型时间特征和心脏周期感知专家模块。该方法使用双路径的专家混合模型分别建模节律和形态，结合使用基于LoRA的分层融合网络进行高效推理。
### Conclusion
ECG-MoE在五个公开的临床任务上实现了最佳性能，并且与多任务基线相比具有40%更快的推理速度。
## 3. `cs.AI` - LLM代理的自适应内存准入控制 [PDF](https://arxiv.org/pdf/2603.04549), [HTML](https://arxiv.org/abs/2603.04549)
### Authors
Guilin Zhang,Wei Jiang,Xiejiashan Wang,Aisha Behr,Kai Zhao,Jeffrey Friedman,Xu Chu,Amine Anoun
### Background
LLM（大语言模型）驱动的代理越来越依赖长期记忆来支持跨会话的推理和交互，然而现有的系统对保留的信息缺乏控制。代理要么累积大量的会话内容，包括错的或过时的信息，要么依赖于完全由LLM驱动的、难以审计的、成本高的记忆策略。因此，内存准入在代理架构中仍然是一个模糊且缺乏控制的组件。
### Innovation
本文提出了自适应内存准入控制（A-MAC），一种将内存准入视为结构化决策问题的框架。A-MAC将内存价值分解为五个互补且可解析的因素：未来效用、事实置信度、语义新颖性、时序新颖性以及内容类型优先级。该框架结合了轻量级规则特征提取与单一LLM辅助效用评估，并通过交叉验证优化学习领域自适应的准入策略。这种设计实现了透明且高效的长期记忆控制。
### Conclusion
A-MAC在LoCoMo基准上的实验表明，与最先进的LLM原生记忆系统相比，其在精确性和召回率之间的权衡上表现出更优的性能，F1得分提高到0.583，同时将延迟降低了31%。消融结果表明，内容类型优先级是可靠内存准入控制中最显著的因素。这些发现证明了显式的、可解析的准入控制对于可扩展和可靠的LLM基于代理的记忆设计至关重要。
## 4. `cs.AI` - 使用视图+语言模型预测项目难度 [PDF](https://arxiv.org/pdf/2603.04670), [HTML](https://arxiv.org/abs/2603.04670)
### Authors
Samin Khan
### Background
该项目旨在研究大型语言模型（LLMs）在确定数据可视化素养测试题难度方面的能力。研究探索了从题目文字、可视化图像或两者结合中提取的特征，能否预测美国成年人数据可视化素养测试题的难度（答对题目的比例）。
### Innovation
采用了多模态的方法，结合使用视觉和文本特征，这比单一模态的视觉或文本特征预测效果更好，显示出LLMs在心理测量分析和自动化题库建设中的潜力。
### Conclusion
最佳表现的多模态模型在独立测试集上进行了外部评估，均方误差为0.10805，这表明LLMs在心理测量分析和自动题库开发中的应用潜能。
## 5. `cs.AI` - 当AI监控行为偏袒自身时：自我归因偏差 [PDF](https://arxiv.org/pdf/2603.04582), [HTML](https://arxiv.org/abs/2603.04582)
### Authors
Dipika Khullar,Jack Hopkins,Rowan Wang,Fabien Roger
### Background
随着代理系统越来越多地依赖语言模型监控自身行为，这些模型被赋予了评估或自我批评生成代码的能力。然而，先前研究表明，当代理行为被呈现给模型审查时，该设计模式可能会失效。背景即探讨在代理系统的背景下，模型在自我归因偏差中可能犯的错误。
### Innovation
该研究首次揭示了自我归因偏差在代理系统中的问题，并提出了一个新的概念——在模型执行由用户提交的任务还是在之前的助手任务期间生成的动作被重新评估时，模型的评估标准会发生变化。研究通过四个编码和工具使用数据集证明了这一现象，强调了这一偏差在代理系统评估中的潜在影响。
### Conclusion
发现代理系统中的监控程序在审查之前由助手生成的动作时，往往会忽视高风险或低正确性的行为，这使得监控程序在实际部署中显得比实际更为可靠，因此开发人员可能无意识地部署了不充分的监控程序。明确指出行为来自监控程序本身并不能独立地引起自我归因偏差。
## 6. `cs.AI` - 基于LLM的指导框架：迈向自动化数据分析 [PDF](https://arxiv.org/pdf/2603.04631), [HTML](https://arxiv.org/abs/2603.04631)
### Authors
Panteleimon Rodis
### Background
大数据集的自动和准确分析变得越来越重要，尤其是当大型语言模型（LLMs）被集成到关键决策流中时。当前的分析方法主要是手工审计，这耗时且复杂。虽然完全基于人工智能的自动分析方法速度更快，但存在幻觉和其他对齐问题。为解决这些问题，论文提出了一个在人类指导下利用生成AI进行数据集风险估计的框架，旨在为未来的自动化风险分析提供基础。
### Innovation
该研究提出了一种整合生成AI的框架，通过生成代码并生成结构和语义特性来分析数据库模式，最终由人类监督者指导模型完成分析任务，确保过程一致性和任务目标的对齐。
### Conclusion
该框架通过人工和生成AI的结合，初步展示了在风险评估任务中产生有意义结果的可能性，从而为未来的自动化风险分析奠定了基础。
## 7. `cs.AI` - 能力阈值与制造拓扑：嵌入式人工智能如何引发经济地理相变 [PDF](https://arxiv.org/pdf/2603.04457), [HTML](https://arxiv.org/abs/2603.04457)
### Authors
Xinmin Fang,Lingfeng Tao,Zhengxiong Li
### Background
自1913年亨利·福特的移动装配线以来，制造业的基础拓扑结构没有经历过转型。过去一个世纪中的每一次重大创新，如丰田生产系统到工业4.0，都是在福特主义的框架内进行优化，而没有改变其结构逻辑：集中式大型工厂，靠近劳动力密集地区，大规模生产。本文认为，嵌入式智能有望打破这一长期僵局，不是通过使现有工厂更高效，而是通过触发制造业经济地理的相变。
### Innovation
本文定义了一个能力空间 C = (d, g, r, t)，当能力向量越过关键面时，选址的目标函数会发生拓扑重组。此外，通过三条路径——重量反转、批处理崩溃和人力-基础设施脱钩——本文展示了嵌入式智能是如何实现需求附近的微制造、消除“制造沙漠”并逆转由劳动力套利驱动的地理集中度。
### Conclusion
本文立基于嵌入式智能经济学，研究物理人工智能能力阈值如何重塑生产的空间和结构性逻辑。一旦人类工人被移除，最优工厂位置由机器最优条件（低湿度、高光照、热稳定性）决定，这些因素与传统的选址逻辑无关，从而形成一个前所未有的生产地理。
## 8. `cs.AI` - 当代理人说服：基于LLM的 propaganda 生成与缓解 [PDF](https://arxiv.org/pdf/2603.04636), [HTML](https://arxiv.org/abs/2603.04636)
### Authors
Julia Jose,Ritik Roongta,Rachel Greenstadt
### Background
尽管基于大规模语言模型（LLM）的代理在开放环境中具有广泛的好处，但它们也可以被利用生成 manipulative 材料。研究通过将 LLM 任务化为其宣传目标，并使用两个领域专用模型进行分析，展示了 LLM 在提示下展示宣传行为并使用各种修辞手法的现象。
### Innovation
研究探索了通过有监督微调（SFT）、直接偏好优化（DPO）和 ORPO（Odds Ratio Preference Optimization）来减轻宣传生成的方法。其中，有监督微调显著降低了 LLM 生成此类内容的倾向，而 ORPO 证明最为有效。
### Conclusion
研究发现，当受到调用时，LLM 会表现出宣传行为并使用多种修辞手法。通过有监督微调、直接偏好优化和 ORPO 等方法，可以显著减少 LLM 生成宣传内容的倾向，其中 ORPO 最为有效。
## 9. `cs.AI` - SkillNet: 创建、评估和连接AI技能 [PDF](https://arxiv.org/pdf/2603.04448), [HTML](https://arxiv.org/abs/2603.04448)
### Authors
Yuan Liang,Ruobin Zhong,Haoming Xu,Chen Jiang,Yi Zhong,Runnan Fang,Jia-Chen Gu,Shumin Deng,Yunzhi Yao,Mengru Wang,Shuofei Qiao,Xin Xu,Tongtong Wu,Kun Wang,Yang Liu,Zhen Bi,Jungang Lou,Yuchen Eleanor Jiang,Hangcheng Zhu,Gang Yu,Haiwen Hong,Longtao Huang,Hui Xue,Chenxi Wang,Yijun Wang,Zifei Shan,Xi Chen,Zhaopeng Tu,Feiyu Xiong,Xin Xie,Peng Zhang,Zhengke Gui,Lei Liang,Jun Zhou,Chiyu Wu,Jin Shang,Yu Gong,Junyu Lin,Changliang Xu,Hongjie Deng,Wen Zhang,Keyan Ding,Qiang Zhang,Fei Huang,Ningyu Zhang,Jeff Z. Pan,Guilin Qi,Haofen Wang,Huajun Chen
### Background
当前的AI代理能够灵活地调用工具并执行复杂的任务，但其长期发展受到缺乏系统积累和转移技能的限制。缺乏统一的技能巩固机制导致代理经常“重新发明轮子”，在孤立的上下文中重新发现解决方案，而不利用先前的策略。
### Innovation
我们提出了SkillNet，这是一个开放的基础设施，旨在按规模创建、评估和组织AI技能。SkillNet通过统一的本体论将技能结构化，支持从异构来源创建技能、建立丰富的关系连接，并在安全性、完备性、可执行性、可维护性和成本意识等方面进行多维度评估。实验评估表明，SkillNet在ALFWorld、WebShop和ScienceWorld上显著增强了代理性能，各类基础模型的平均奖励提高40%，执行步骤减少30%。
### Conclusion
通过将技能正式化为可演进、可组合的资产，SkillNet为代理从短暂经验过渡到持久掌握提供了坚实的基础。
## 10. `cs.AI` - 渐进式精细调节以加速扩散语言模型解码 [PDF](https://arxiv.org/pdf/2603.04514), [HTML](https://arxiv.org/abs/2603.04514)
### Authors
Lipeng Wan,Jianhui Gu,Junjie Ma,Jianguo Huang,Shiguang Sun,Siyuan Li,Xuguang Lan
### Background
目前的扩散语言模型通过统一精炼规则对所有标记进行逐迭代除噪来生成文本。然而，在实践中，标记稳定化速度不同，导致大量冗余精炼，推动了对精炼过程进行精炼控制的需求。现有方法通常基于固定的解码过程评估每一步骤的精炼必要性，其通过即时、步骤级的信号进行评估。相比之下，是否某个标记已经收敛则取决于其未来精炼轨迹上的预测变化。精炼规则的变化会重塑未来精炼轨迹，反过来决定了精炼规则的构建方式，使得精炼控制本质上是动态的。
### Innovation
本文提出了一种渐进式、基于轨迹的精炼控制框架——渐进式精炼调节（Progressive Refinement Regulation, PRR），该框架能够从完整的解码展开中推导出标记级的收敛进展经验性概念。基于这一信号，PRR 学习了基于温度调整分布的轻量化标记级控制器，以渐进自我演化训练方案实现精炼调控。通过实验验证，PRR 能显著加速扩散语言模型的解码同时保持生成质量。
### Conclusion
实验证明，PRR 能够加速扩散语言模型的解码过程而不牺牲生成的质量。
## 11. `cs.CL` - 一种在语音产生期间同时获取实时MRI视频、EEG和表面EMG的艺术运动、大脑和肌肉活动的方法 [PDF](https://arxiv.org/pdf/2603.04840), [HTML](https://arxiv.org/abs/2603.04840)
### Authors
Jihwan Lee,Parsa Razmara,Kevin Huang,Sean Foley,Aditya Kommineni,Haley Hsu,Woojae Jeong,Prakash Kumar,Xuan Shi,Yoonjeong Lee,Tiantian Feng,Takfarinas Medani,Ye Tian,Sudarsana Reddy Kadiri,Krishna S. Nayak,Dani Byrd,Louis Goldstein,Richard M. Leahy,Shrikanth Narayanan
### Background
语音产生是一个复杂的神经生理过程，涉及神经计划、运动控制、肌肉激活和发音动作。尽管语音的声学信号是最容易获取的产品，但它并不能直接揭示其因果神经生理的底层结构。目前，多模态采集在语音科学中的应用面临着技术挑战，如MRI引起的电磁干扰和肌电伪影。
### Innovation
本文介绍了一种同时获取实时MRI、EEG和表面EMG的方法，涵盖语音产生链中的关键方面：大脑信号、肌肉激活和发音动作。为了缓解MRI引起的电磁干扰和肌电伪影等技术挑战，我们提出了一个定制的去伪影处理流水线，专门针对三模态设置。这项多模态采集范式有望提供前所未有的窗口，探索语音神经科学，进而推动脑-机接口技术的进步。
### Conclusion
一旦开发完成，该框架就有望为语音神经科学提供前所未有的视角，并带来脑-机接口技术的新进展。
## 12. `cs.CL` - 打破上下文惯性：基于单一转锚点的强化学习在稳定多轮交互中的应用 [PDF](https://arxiv.org/pdf/2603.04783), [HTML](https://arxiv.org/abs/2603.04783)
### Authors
Xingwu Chen,Zhanqiu Zhang,Yiwen Guo,Difan Zou
### Background
尽管大语言模型在单次交互中提供了完整信息时表现出强大的推理能力，但在多轮交互中却表现出显著的脆弱性。具体来说，在信息逐步揭示或需要更新时，模型往往无法整合新的约束条件，导致性能大幅下降，低于单轮交互的基准。这是由于模型表现出一种被称为“上下文惯性”的现象：模型固守于之前的推理路径，即使用户在后续轮次中提供明确的纠正或新数据，模型也会忽略这些信息，以保持之前的、错误的推理路径一致。
### Innovation
本文提出了一种名为RLSTA（Reinforcement Learning with Single-Turn Anchors）的一般化训练方法，以解决多轮交互中的上述问题。RLSTA利用模型在单轮交互中表现出的优越稳定性作为内部锚点，通过多轮响应与这些锚点对齐来提供奖励信号。这种方法能够使模型打破上下文惯性，根据最新信息自我校准其推理。
### Conclusion
实验表明，RLSTA显著优于标准微调和主动放弃方法。此外，该方法在跨领域泛化（如数学到代码）方面表现出色，并且在缺少外部验证者的情况下仍有效，证明了其在通用领域的应用潜力。
## 13. `cs.CL` - VisionPangu: 具有17亿参数的紧凑且精细的多模态助手 [PDF](https://arxiv.org/pdf/2603.04957), [HTML](https://arxiv.org/abs/2603.04957)
### Authors
Jiaxin Fan,Wenpo Song
### Background
LMMs已经在视觉-语言理解方面取得了显著的成果，但许多现有方法依赖于大规模架构和粗粒度的监督，这限制了它们生成详细图像描述的能力。
### Innovation
提出VisionPangu，一种紧凑的1.7B参数多模态模型，通过高效多模态对齐和高质量的监督提高详细图像描述的准确性。该模型结合了InternVL衍生的视觉编码器和OpenPangu-Embedded语言骨干，并通过轻量级的MLP投影器进行连接，采用灵感来自LLaVA的指令调优流水线。通过融入DOCCI数据集中的密集人工描述，VisionPangu提升了语义连贯性和描述丰富性，而不需要使用激进的模型放大。
### Conclusion
实验结果表明，紧凑的多模态模型可以实现具有竞争力的性能，在生成详细和结构化描述方面更具优势。代码和模型权重将在公开网址 this https URL 公开可用。
## 14. `cs.CL` - Alignment Backfire: Language-Dependent Reversal of Safety Interventions Across 16 Languages in LLM Multi-Agent Systems [PDF](https://arxiv.org/pdf/2603.04904), [HTML](https://arxiv.org/abs/2603.04904)
### Authors
Hiroki Fukui
### Background
研究发现，在罪犯治疗中普遍存在认知与行为之间的脱节现象，即罪犯虽然表示悔改，但行为并未随之改变。本文探讨了大语言模型中的类似现象，即通过调整方法使模型行为更加一致时，表面看来提高了安全性，但实际上可能掩盖了集体心理病理并引起内部脱节。
### Innovation
论文通过四项预注册的研究，证明了通过调整方法使大语言模型更加一致时，会出现表面安全但实质上却产生了集体心理病理和内部脱节的现象。这些研究在16种语言和三种模型家族中进行了验证，揭示了这种现象在不同语言和文化背景下的差异性。
### Conclusion
研究结果表明，模型的安全性调整受到语言和文化背景的结构性影响，即表面的安全性改善在不同语言中表现不同，甚至可能适得其反。这需要重新审视模型行为干预的风险，并强调跨文化干预措施的有效性限制。这些发现将模型的调整视为一种行为干预，并将其置于风险自稳态与医源性危害的框架下。
## 15. `cs.CL` - 为什么RLHF对齐浅薄？梯度分析 [PDF](https://arxiv.org/pdf/2603.04851), [HTML](https://arxiv.org/abs/2603.04851)
### Authors
Robin Young
### Background
该论文探讨了大规模语言模型（LLMs）中安全对齐问题，指出基于梯度的对齐方法在处理潜在有害输出时存在局限性，这导致了浅层对齐。
### Innovation
作者通过鞅分解序列级别危害，推导出精确描述对齐梯度的方程，显示出了在输出有害性已决定的阶段后梯度信号消失的现象。引入了危害信息的概念，证明了静态KL分散度追踪这一概念，并提出了基于恢复惩罚的目标，以产生在整个位置上的梯度信号。
### Conclusion
标准对齐目标不能产生深层对齐，而基于恢复惩罚的目标则提供了对此现象的理论依据，支持了成功的数据增强技术。
## 16. `cs.CL` - 超越线性LLM调用：一种高效且有效的语义过滤范式 [PDF](https://arxiv.org/pdf/2603.04799), [HTML](https://arxiv.org/abs/2603.04799)
### Authors
Nan Hou,Kangfei Zhao,Jiadong Xie,Jeffrey Xu Yu
### Background
大型语言模型（LLMs）被广泛用于处理大规模语料库的语义查询。已提出了一套从关系代数中派生的语义操作符来提供统一接口表达此类查询，其中语义筛选操作符至关重要。但是，由于这种逐元组的评估需要对表进行完整的一次性扫描，导致了显著的延迟和令牌成本。尽管最近有一些工作试图优化语义筛选，但它们仍然无法突破LLM调用的线性障碍。
### Innovation
为解决上述问题，本文提出了Clustering-Sampling-Voting（CSV）框架，它通过嵌入元组到语义集群中、抽样小部分进行LLM评估，并通过两个提出的投票策略（Uniform Voting (UniVote) 和 Semantic Similarity Weighted Voting (SimVote)）推断集群级标签来降低LLM调用次数至亚线性复杂度，同时保证了错误保证。此外，CSV在模糊集群上触发重新聚类，以确保多种数据集上的鲁棒性。
### Conclusion
在实际数据集上的实验结果表明，与最先进的方法相比，CSV将LLM调用次数减少至1.28至355倍，同时在精确度和F1分数方面保持相当的有效性。
## 17. `cs.CL` - Privacy-Aware Camera 2.0 技术报告 [PDF](https://arxiv.org/pdf/2603.04775), [HTML](https://arxiv.org/abs/2603.04775)
### Authors
Huan Song,Shuyu Tian,Ting Long,Jiang Liu,Cheng Yuan,Zhenyu Jia,Jiawei Shao,Xuelong Li
### Background
随着智能传感技术在如卫生间和储物室等高度敏感环境中的广泛应用，视觉监控系统面临着隐私安全悖论的挑战。现有的隐私保护方法，如物理脱敏、加密和模糊化，往往会影响语义理解，或者不能保证数学上的不可逆性。尽管Privacy Camera 1.0在源头消除了视觉数据以防止泄露，但它只提供了文本判断，导致在纠纷中存在证据盲点。
### Innovation
本文提出了一种基于AI Flow范式和协作边缘-云架构的新颖的隐私保护感知框架。通过在边缘部署视觉钝化器，通过对信息瓶颈原则下的非线性映射和随机噪声注入将原始图像实时转换为抽象特征向量，确保隐私敏感信息被去除，并且原始图像在数学上不可重建。这种抽象表示通过“动态轮廓”视觉语言传输至云端进行行为识别和语义重构，实现了感知和隐私之间的关键平衡，同时也使得提供具有代表性的视觉参考而不需要暴露原始图像。
### Conclusion
该框架在保持合理的目的感知能力的同时，极大地保护了视觉监控系统中的个人隐私。通过测试表明，该方法在提高隐私保护的同时，仍能有效支持行为识别任务，为未来的隐私安全监控发展提供了新的思路。
## 18. `cs.CL` - Mixture of Universal Experts: Scaling Virtual Width via Depth-Width Transformation [PDF](https://arxiv.org/pdf/2603.04971), [HTML](https://arxiv.org/abs/2603.04971)
### Authors
Yilong Chen,Naibin Gu,Junyuan Shang,Zhenyu Zhang,Yuchen Feng,Jiawei Sheng,Tingwen Liu,Shuohuan Wang,Yu Sun,Hua Wu,Haifeng Wang
### Background
当前Mixture-of-Experts (MoE)模型能够将模型的容量与每个令牌的计算量分离开来，但其可扩展性仍然受限于深度和宽度的物理维度。MoE通过在不同层之间重用相同类型的专家来实现并行计算，但这会导致路由路径的爆炸性增长和重用诱导的曝光与传统的负载均衡目标之间存在不匹配。
### Innovation
本文提出了一种新的模型扩展方法Mixture of Universal Experts (MOUE)，引入了虚拟宽度这一新的扩展维度。MOUE旨在利用一个适用于所有层的专家池，并通过结构化的专家共享、深度感知的负载平衡以及轻量级的多步骤路由来解决路径爆炸和负载分配不平衡的问题。
### Conclusion
在不同扩展范围内，MOUE始终优于匹配的MoE基线，可提高多达1.3%的性能；此外，MOUE能够以高达4.2%的收益将现有的MoE检查点转换为MOUE，发现了MoE架构中的新扩展维度。
## 19. `cs.CL` - TimeWarp：通过重温过去评估网络代理 [PDF](https://arxiv.org/pdf/2603.04949), [HTML](https://arxiv.org/abs/2603.04949)
### Authors
Md Farhan Ishmam,Kenneth Marino
### Background
由于现代网络代理在当前基准测试中的性能提升，这个问题引起了注意：当网络发生变化时，今天的代理是否还会表现得同样出色？研究表明，网络环境的演变对代理性能有显著影响，单一版本的行为克隆方法在这种变化面前显得力不从心。
### Innovation
本文提出了一种名为TimeWarp的基准测试，它通过容器化环境模拟了不断变化的网络环境，这些环境在用户界面、设计和布局方面各不相同。TimeWarp包含三个网络环境，每个环境有六个不同互联网时代的用户界面版本，以及相应的复杂现实任务。研究还提出了一种名为TimeTraj的新算法，通过跨版本计划提炼收集轨迹，相比于单一版本的行为克隆方法，能够在训练代理上达到显著的性能提升。
### Conclusion
通过我们的实验和TimeTraj算法，我们证明了网络代理对环境变化的脆弱性，以及单一版本行为克隆方法的局限性。我们的工作旨在帮助研究人员研究不同网页设计下的泛化能力，并有望开启一种新的计划收集方法，从而提高网络代理的鲁棒性。
## 20. `cs.CL` - Fisher--Rao 曲面上基于功能的 LLM 融合 [PDF](https://arxiv.org/pdf/2603.04972), [HTML](https://arxiv.org/abs/2603.04972)
### Authors
Jiayu Wang,Zuojun Ye,Wenpeng Yin
### Background
现有的权重空间合并方法主要是基于参数空间的手法，如线性平均和任务向量，这些方法主要针对欧几里得坐标进行操作，但目的是要在多个微调过的大型语言模型（LLM）之间合并功能，即任务预测行为的功能性合并。这类方法存在三个实践限制：第一，当前方法在合并多个模型的功能时缺乏有效性；第二，当源检查点间差异较大时，欧几里得混合可能导致表示崩溃，这在精度上会有明显下降；第三，许多几何启发方法最适合两模型插值，但难以扩展到合并多位专家情况。
### Innovation
本文提出了一种基于 Fisher--Rao 曲面上加权 Karcher 平均的新方法，这对于解决参数空间方法的局限性具有创新性。具体而言，该方法使用黎曼几何框架中的加权 Karcher 平均来计算预测分布之间的 KL 距离最小化。此外，通过使用一个轻量级的球形代理来保持范数，该方法能够直接扩展到多位专家的合并。
### Conclusion
实验结果表明，该方法在合并的模型数量和异质性增加的情况下仍保持稳定性，且在多种基准测试和崩溃诊断中均优于之前的方法。因此，这一方法能够有效解决现有合并方法存在的问题，并提供了一种新的用于 LLM 融合的可靠途径。
## 21. `cs.LG` - 随机性一致性评分的对称聚合以实现高效的不确定性集 [PDF](https://arxiv.org/pdf/2512.06945), [HTML](https://arxiv.org/abs/2512.06945)
### Authors
Nabil Alami,Jad Zakharia,Souhaib Ben Taieb
### Background
随着多个针对同一任务进行训练的预测模型在多种应用中变得越来越普遍，整合这些模型的预测不确定性以生成可靠且高效的不确定性量化仍然面临挑战，尤其是在符合预测框架中。现有的符合预测方法可以从每个模型生成个别预测集，但将这些模型整合成一个更具有信息价值的集合仍然是一个难题。
### Innovation
本文提出了一种新颖的方法——SACP（对称聚合一致预测），它可以将多个预测器的非一致评分转化为e值，并使用任一对称聚合函数来结合这些评分。这一灵活的设计允许选择能够产生更精确预测集的聚合策略，同时提供了理论见解来支持SACP方法的有效性和性能。
### Conclusion
在多种数据集上进行的广泛实验表明，SACP方法在效率上表现较差，但在大多数情况下，SACP方法能够超越最先进的模型聚合基准方法。
## 22. `cs.LG` - NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation [PDF](https://arxiv.org/pdf/2512.05106), [HTML](https://arxiv.org/abs/2512.05106)
### Authors
Yu Zeng,Charles Ochoa,Mingyuan Zhou,Vishal M. Patel,Vitor Guizilini,Rowan McAllister
### Background
标准的扩散过程使用具有随机幅度和随机相位的高斯噪声来污染数据。这种噪声对于无条件生成或文本转图像生成有效，但破坏了相位成分，损害了空间结构，使其不适合需要几何一致性的任务，例如重新渲染、模拟增强和图像转图像翻译。
### Innovation
引入了一个模型无关的扩散过程改写textbackslash{}phi-PD，该过程保留输入相位的同时随机化幅度，允许结构对齐生成而不需更改架构或增加参数。提出了一种频率选择性结构（FSS）噪声，通过单一频率截止参数提供对结构刚性的连续控制。此外，textbackslash{}phi-PD 不增加推理时间成本，并可与任何图像或视频的扩散模型兼容。
### Conclusion
在现实照片和风格化重新渲染、以及从模拟到现实的驾驶规划增强方面，textbackslash{}phi-PD 生成了可控且空间对齐的结果。当应用于 CARLA 模拟器时，该方法显著提高了模拟到现实规划器的迁移性能。该方法与现有的条件方法互补，并广泛适用于图像到图像和视频到视频生成。
## 23. `cs.LG` - 具代理性的多角色框架以实现基于证据的假新闻检测 [PDF](https://arxiv.org/pdf/2512.21039), [HTML](https://arxiv.org/abs/2512.21039)
### Authors
Roopa Bukke,Soumya Pandey,Suraj Kumar,Soumi Chattopadhyay,Chandranath Adak
### Background
网络上的虚假信息迅速扩散，威胁着数字社会系统的稳定，并对公共信任、政策和安全构成重大风险。现有的方法通常在应对多媒体内容、领域泛化和可解释性方面存在问题。为了应对这些挑战，需要一种可靠的自动化虚假信息检测方法。
### Innovation
提出了一个具有代理性的多角色证据驱动框架AMPEND-LS，并结合LLM-SLM的协同作用，用于多模态虚假信息检测。AMPEND-LS通过LLMs驱动的结构化推理管道整合文本、视觉和上下文信号，加入了逆向图像搜索、知识图谱路径和说服策略分析。为了提高可靠性，引入了结合语义相似性、领域可信度和时间上下文的可信度融合机制，以及互补的SLM分类器以减少LLM的不确定性与幻觉。
### Conclusion
该框架在基于权威数据集的广泛实验中，表现出一致优于最先进的基线模型的准确性、F1分数和稳健性。定性的案例研究进一步突显了其透明的推理过程以及对演进式虚假信息的鲁棒性。这项工作促进了适应性、可解释性以及基于证据的系统的开发，以保障在线信息的完整性。
## 24. `cs.LG` - 语言模型中的并行 token 预测 [PDF](https://arxiv.org/pdf/2512.21323), [HTML](https://arxiv.org/abs/2512.21323)
### Authors
Felix Draxler,Justus Will,Farrin Marouf Sofian,Theofanis Karaletsos,Sameer Singh,Stephan Mandt
### Background
自回归解码在语言模型中固然是缓慢的，每次前向传递只能生成一个标记。因此，生成长文本需要大量的计算资源和时间。
### Innovation
提出了并行 token 预测（PTP）框架，它在单次模型调用中可以预测多个 token，将随机性从后处理采样转移到随机输入变量中，使得未来 token 成为输入变量的确定性函数，从而在一个前向传递中联合可预测。
### Conclusion
实验结果表明，PTP 在多任务推测性解码基准上实现了 2.4 倍的加速。PTP 可通过蒸馏现有模型或逆向自回归训练进行训练，无需教师指导。源代码和检查点通过此链接提供：this https URL.
## 25. `cs.LG` - Observer-Actor: Sparse-视图高斯点积的主动视觉模仿学习 [PDF](https://arxiv.org/pdf/2511.18140), [HTML](https://arxiv.org/abs/2511.18140)
### Authors
Yilong Wang,Cheng Qian,Ruomeng Fan,Edward Johns
### Background
本文提出了一种新的框架Observer Actor (ObAct)，该框架旨在让观察者主动选择视觉观察的最优位置，以便于执行者更好地进行操作。研究主要在配备腕部摄像头的双臂机器人系统上进行。这种设计增强了执行者观察中物体和抓手的清晰度和可见性，从而使得模仿学习更高效、更鲁棒。
### Innovation
提出了Observer Actor (ObAct)，通过让观察者移动到最优视觉观察位置，改进了执行者在动态设置下的观察能力。这种新颖的框架使用3D高斯点积（3DGS）来构建视图表示，并通过虚拟探索找到最优相机姿态。实验结果表明，ObAct 在没有遮挡和有遮挡的情况下较静态摄像头设置分别提高了145%和233%，对于轨迹传输方法和行为克隆方法都有显著提升。
### Conclusion
本文通过实验验证了Observer Actor框架的有效性，增强了执行者观察物体和抓手的清晰度，并显著提升了模仿学习的性能。即使在有遮挡的情况下，能够训练出更鲁棒的政策。这些方法可以在双臂机器人上实现更为精细和复杂的操作任务。
## 26. `cs.LG` - CycleChemist: 一种双管齐下的机器学习框架，用于有机光伏材料发现 [PDF](https://arxiv.org/pdf/2511.19500), [HTML](https://arxiv.org/abs/2511.19500)
### Authors
Hou Hei Lam,Jiangjie Qiu,Xiuyuan Hu,Wentao Li,Fankun Zeng,Siwei Fu,Hao Zhang,Xiaonan Wang
### Background
有机光伏（OPV）材料是可持续能源生成的理想选择，但其开发受到高功率转换效率（PCE）的高性能给受体对难以识别的限制。现有的设计策略通常只关注给体或受体单一组件，缺乏统一的方法来同时建模两者。
### Innovation
本文提出了一个结合预测建模和生成分子设计的双机器学习框架，用于OPV材料发现。该框架利用了有机光伏捐赠接受者数据集（OPV2D），包含2000个实验表征的捐赠接受者对，并开发了有机光伏分类器（OPVC）、多任务学习和捐赠接受者相互作用建模的分层图神经网络，以及分子轨道能量估算器（MOE2）和光伏性能预测器（P3）。此外，引入了材料生成预训练变换器（MatGPT），通过目标策略优化的强化学习策略生产可合成的有机半导体。
### Conclusion
通过将分子表示学习与性能预测相结合，该框架推动了高性能OPV材料的数据驱动发现。
## 27. `cs.LG` - GRAND: 网络分配中的指导、再平衡与分配多智能体路径规划 [PDF](https://arxiv.org/pdf/2512.03194), [HTML](https://arxiv.org/abs/2512.03194)
### Authors
Johannes Gaber,Meshal Alharbi,Daniele Gammelli,Gioele Zardini
### Background
大型机器人队现在在仓库和其他物流环境中非常普遍，小的控制增益会转化为较大的运营影响。本文针对终身多代理拾取和交付(MAPD)任务调度，提出了一个将基于学习的全局指导与轻量级优化相结合的混合方法。
### Innovation
提出了一种称为GRAND的分层算法，该算法依赖于指导、再平衡和分配策略，明确利用工作空间网络结构并分配代理执行任务。这种方法利用图神经网络策略进行强化学习，输出期望的自由代理分布，通过最小成本流进行区域到区域的再平衡，并通过局部分配问题进行最终确定，保持准确性的同时确保每步延迟在1秒的计算预算内。
### Conclusion
在从League of Robot Runners (LoRR)得到的拥堵仓库基准测试中，最多500个代理的环境中，该方法在保持实时执行的情况下，相比2024年的冠军调度器提高了最多10%的吞吐量。结果表明，将图结构的习得指导与可处理的求解器结合可以降低拥堵并提供一种在大型车队中实现高吞吐量调度的实用、可扩展蓝图。
## 28. `cs.LG` - ReFusion：具有并行自回归解码的扩散大语言模型 [PDF](https://arxiv.org/pdf/2512.13586), [HTML](https://arxiv.org/abs/2512.13586)
### Authors
Jia-Nan Li,Jian Guan,Wei Wu,Chongxuan Li
### Background
自回归模型（ARMs）的顺序推理速度较慢。尽管掩蔽扩散模型（MDMs）提供建立并行替代方案，但它们面临着关键的缺点：高计算开销，因为禁止键值（KV）缓存，以及从难以处理的标记组合空间中学习依赖关系导致生成结果不一致。
### Innovation
我们提出了ReFusion，一种新颖的掩蔽扩散模型，它将序列重组整合到因果注意力框架中。通过将并行解码从标记级别提升到更高层级的插槽级别，ReFusion交错使用插槽间的基于扩散的选择与插槽内的自回归填充，同时在每次迭代后重新生成插槽并预先安排剩余的掩码。这种设计同时解决了全面的KV缓存重用问题并降低了从难以处理的标记组合空间学到的复杂性，转化为可管理的插槽级排列空间。
### Conclusion
在七个不同基准上的广泛实验表明，ReFusion不仅超越了之前的MDMs，在平均上提升了34%的性能并且快了超过18倍的速度，还减少了性能差距与强劲的ARMs之间的差距，同时保持平均2.33倍的速度优势。
## 29. `cs.LG` - CoRPO: 在GRPO中添加正确性偏见以提高泛化能力 [PDF](https://arxiv.org/pdf/2511.04439), [HTML](https://arxiv.org/abs/2511.04439)
### Authors
Anisha Garg,Claire Zhang,Nishit Neema,David Bick,Ganesh Venkatesh,Joel Hestness
### Background
Group-Relative Policy Optimization (GRPO) 已成为通过强化学习训练大语言模型推理能力的标准方法。通过使用群体均值奖励而非学习批评家来估计优势，GRPO 使从可验证奖励 (RLVR) 规模化的强化学习变得更加高效。然而，GRPO 的均值基线可以简单地赋予表现优于低效群体平均值的错误解决方案为正优势，从而导致对错误行为的过度估计和强化。
### Innovation
本文提出了一种简单的修正 CoRPO，它将 GRPO 目标中的最低基线限制到固定的正确性阈值。这种方法引入了对优势估计的保护性偏差，减少了过拟合并保留有效的探索。
### Conclusion
实验结果表明，使用 CoRPO 训练的模型在跨领域的推理中有所提高，并且能更一致地泛化到域外 (OOD) 任务。当在编码任务上训练时，CoRPO 在数学方面优于 GRPO，反之亦然，表明 CoRPO 学习到的是稳健且可迁移的推理模式而非特定任务的解决方案。
## 30. `cs.LG` - DPAC：基于分布保持的对抗控制在扩散采样中的应用 [PDF](https://arxiv.org/pdf/2512.01153), [HTML](https://arxiv.org/abs/2512.01153)
### Authors
Han-Jin Lee,Han-Ju Lee,Jin-Seong Kim,Seok-Hwan Choi
### Background
对抗引导的扩散采样通常能够实现目标类别，但样本质量会随着对抗控制轨迹与正常轨迹偏差的累积而下降。目前缺乏一种形式化表示这种下降的方法，特别是将其转化为路径空间Kullback-Leibler散度(path-KL)。研究表明，这种散度等于控制能量，从随机最优控制(SOC)的角度，理论建立了一种最小化path-KL的方法，该方法同时能够收紧2- Wasserstein距离和Fréchet Inception Distance（FID）的上界，揭示了对抗控制能量与感知保真度之间的原理性联系。
### Innovation
文章通过建立路径空间Kullback-Leibler散度(path-KL)来表征对抗控制过程与正常扩散过程之间的差异，通过Girsanov定理证明了path-KL等于控制能量。随后，从变分角度推导出关于控制的第一个最优性条件，即在相同分类增益的条件下，与等概率（对数-密度）面相切的方向在最小化path-KL方面效果更优，而垂直方向则直接增加分布漂移。文章提出了DPAC（分布保持对抗控制），通过在生成得分几何定义的切空间中投影对抗梯度实现扩散指导规则。实验上验证了当攻击成功率匹配时，DPAC提供了更低的FID和路径KL估计。
### Conclusion
通过对path-KL的分析，文章揭示了对抗控制能量与感知保真度之间的联系，并提出了基于等概率（对数-密度）面切线方向的DPAC方法，进一步通过分析在离散求解器中的体现，展示了DPAC方法对模型和度量近似的鲁棒性，实验结果证明了其有效性。
