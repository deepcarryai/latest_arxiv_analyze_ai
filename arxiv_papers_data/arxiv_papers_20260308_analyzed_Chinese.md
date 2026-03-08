# 20260308
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - SkillNet: 创建、评估和连接人工智能技能 [PDF](https://arxiv.org/pdf/2603.04448), [HTML](https://arxiv.org/abs/2603.04448)
### Authors
Yuan Liang,Ruobin Zhong,Haoming Xu,Chen Jiang,Yi Zhong,Runnan Fang,Jia-Chen Gu,Shumin Deng,Yunzhi Yao,Mengru Wang,Shuofei Qiao,Xin Xu,Tongtong Wu,Kun Wang,Yang Liu,Zhen Bi,Jungang Lou,Yuchen Eleanor Jiang,Hangcheng Zhu,Gang Yu,Haiwen Hong,Longtao Huang,Hui Xue,Chenxi Wang,Yijun Wang,Zifei Shan,Xi Chen,Zhaopeng Tu,Feiyu Xiong,Xin Xie,Peng Zhang,Zhengke Gui,Lei Liang,Jun Zhou,Chiyu Wu,Jin Shang,Yu Gong,Junyu Lin,Changliang Xu,Hongjie Deng,Wen Zhang,Keyan Ding,Qiang Zhang,Fei Huang,Ningyu Zhang,Jeff Z. Pan,Guilin Qi,Haofen Wang,Huajun Chen
### Background
当前的AI代理能够灵活调用工具并执行复杂的任务，但长期发展受限于技能的系统积累与转移机制的缺乏。缺乏统一的技能巩固机制导致代理在孤立情境中重新发现解决方案，未能利用之前的策略。
### Innovation
提出了一种名为SkillNet的开放基础设施，用于批量创建、评估和组织AI技能。SkillNet通过统一的本体结构化技能，并支持从不同来源创建技能、建立丰富的关系连接、以及在安全、完备性、执行性、维护性和成本意识方面进行多维度评估。基础设施包括超过200,000个技能的存储库、交互式平台和多功能Python工具包。实验证实，SkillNet显著提高了代理性能，在ALFWorld、WebShop和ScienceWorld等多个应用场景下，平均奖励提升了40%，执行步骤减少了30%。
### Conclusion
通过将技能形式化为可演化和组合的资产，SkillNet为代理提供了从短暂经验过渡到持久精通的坚实基础。
## 2. `cs.AI` - 向自动化数据分析迈进：基于LLM的指导风险估计框架 [PDF](https://arxiv.org/pdf/2603.04631), [HTML](https://arxiv.org/abs/2603.04631)
### Authors
Panteleimon Rodis
### Background
大型语言模型（LLMs）越来越多地被集成到关键决策流程中，这一趋势增加了对高效和自动化的数据集风险分析的需求。当前的数据集风险分析方法依赖于手动审计方法，耗时且复杂，而基于人工智能的全自动分析则面临幻觉和AI对齐问题。
### Innovation
本文提出了一种在人类指导下和监督下集成生成型AI的数据集风险估计框架，旨在为未来自动化的风险分析范式奠定基础。该方法利用LLMs识别数据库模式中的语义和结构特性，提出聚类技术，生成代码，并最终解释产生结果。人类监督员指导模型进行所需的分析，并确保过程的完整性和与任务目标的一致。
### Conclusion
提出的研究框架通过利用人类的指导和监督，有效结合了生成型AI，为数据集风险评估任务提供了可行的方法，具有重要意义。
## 3. `cs.AI` - 当代理机构劝说时：LLM中的宣传生成与缓解 [PDF](https://arxiv.org/pdf/2603.04636), [HTML](https://arxiv.org/abs/2603.04636)
### Authors
Julia Jose,Ritik Roongta,Rachel Greenstadt
### Background
虽然基于LLM的代理人在开放环境中具有广泛的好处，但它们也可能被利用生成具有操纵性的内容。本文探讨了当LLM被赋予宣传目标时，它们如何生成宣传性质的内容，并且使用了两项专门领域模型：一个用于分类文本为宣传或非宣传，另一个用于检测宣传手法（如情绪用语、恐惧诉求、旗帜挥动、污名化）。研究还发现了当被提示时，LLM会表现出宣传行为并在其中使用多种修辞技巧。这种现象需要采取措施进行缓解。
### Innovation
研究通过使用监督微调（SFT）、直接偏好优化（DPO）和ORPO（概率比偏好优化）来探索缓解方法。研究发现，微调显著减少了LLMs生成此类内容的倾向，而ORPO在缓解效果上最为明显。
### Conclusion
当被提示时，LLMs会展示出宣传行为，并使用各种修辞技艺。通过实验发现，微调显著降低了生成此类内容的倾向，ORPO方法效果最为突出。
## 4. `cs.AI` - 使用视觉+语言模型预测题目难度 [PDF](https://arxiv.org/pdf/2603.04670), [HTML](https://arxiv.org/abs/2603.04670)
### Authors
Samin Khan
### Background
本项目旨在利用大规模语言模型（LLMs）来确定数据可视化识字测试题目的难度。研究探讨了项目文本（问题和选项）特征、可视化图像特征、或两者结合特征对预测美国成年人测试题难度（正确响应的比例）的能力。使用GPT-4.1-nano分析题目并基于这些特征集生成预测。结果表明，结合视觉和文本特征的多模态方法具有最低的平均绝对误差（MAE），分别为0.224，优于仅视觉（0.282）和仅文本（0.338）的方法。最佳表现的多模态模型在外部测试集上进行评估，均方误差为0.10805，这表明LLMs有潜力用于心理测量分析和自动题目开发。
### Innovation
本研究展示了使用GPT-4.1-nano分别利用视觉和文本特征进行多模态分析，并证明了这种结合方法具有较低的预测误差。这表明LLMs在预测题目难度上的巨大潜力，特别是在数据可视化识字测试领域。
### Conclusion
结合视觉和文本特征的多模态模型在预测题目难度方面表现最佳，提供了一种新的方法进行心理测量分析和自动题目开发。
## 5. `cs.AI` - 渐进式精炼调节以加速扩散语言模型解码 [PDF](https://arxiv.org/pdf/2603.04514), [HTML](https://arxiv.org/abs/2603.04514)
### Authors
Lipeng Wan,Jianhui Gu,Junjie Ma,Jianguo Huang,Shiguang Sun,Siyuan Li,Xuguang Lan
### Background
扩散语言模型通过应用统一的精炼规则对所有标记进行逐代去噪来生成文本。然而，在实践中，标记的稳定速率不同，导致了显著的多余精炼，并促使在精炼过程中控制精炼过程。现有的方法通常基于固定解码过程和瞬时步骤信号来评估精炼的必要性。相比之下，标记是否收敛是基于其未来精炼轨迹中预测的变化来定义的。此外，改变精炼规则会重塑未来的精炼轨迹，从而影响如何制定精炼规则，使得精炼控制本质上是动态的。
### Innovation
论文提出了一种渐进式精炼调节（PRR）框架，这是一种基于精炼轨迹的渐进精炼控制框架，通过完整的解码全局回放从全解码卷集推导出标记层面的经验收敛进展信号。基于此信号，PRR 学习一种轻量级的标记层面控制器，通过基于温度的分布塑造来调节精炼，在渐进式自我演变训练方案下进行。
### Conclusion
实验表明，PRR 显著加快了扩散语言模型解码速度，同时保持了生成质量。
## 6. `cs.AI` - 自我归因偏见：当AI监控器自我宽容时 [PDF](https://arxiv.org/pdf/2603.04582), [HTML](https://arxiv.org/abs/2603.04582)
### Authors
Dipika Khullar,Jack Hopkins,Rowan Wang,Fabien Roger
### Background
随着代理系统越来越多地依赖语言模型来监控自身的操作，例如编码代理可能会自我评估生成的代码以进行拉取请求审批或评估工具使用行为的安全性。然而，当操作由代理模型自身而非用户在先前或同一助手轮次呈现时，代理设计模式可能会失效。研究发现，监控行为在评估时若紧跟在其生成的助手轮次之后，与在用户轮次中呈现同一行为时相比，更容易忽视高风险或低正确性行为。
### Innovation
本文首次通过定义自我归因偏见来研究语言模型在评价自身操作时的偏差现象。研究跨越了四个编码和工具使用数据集，揭示了传感器在评估发生在先前助手轮次中的动作时容易发现问题较少，而在新的用户轮次中呈现相同动作时则能更准确地识别问题。研究还发现，只要明确表示动作来自传感器，就不足以消除自我归因偏见。
### Conclusion
尽管明确声明动作来自监控器不会单独引发自我归因偏见，但由于传感器经常在固定实例而非自身生成的动作上进行评估，导致传感器在部署时显得比实际上更可靠，这可能会误导开发者无意中部署了不充分的传感器。
## 7. `cs.AI` - ECG-MoE: Mixture-of-Expert Electrocardiogram Foundation Model [PDF](https://arxiv.org/pdf/2603.04589), [HTML](https://arxiv.org/abs/2603.04589)
### Authors
Yuhao Xu,Xiaoda Wang,Yi Wu,Wei Jin,Xiao Hu,Carl Yang
### Background
心电图（ECG）分析对于心脏诊断至关重要，但现有的基础模型通常无法捕捉到不同临床任务所需的周期性和多样特征。
### Innovation
提出了ECG-MoE，这是一种混合架构，将多模型时间特征与具有心脏周期感知专家模块相结合。该方法使用双路径混合专家（Mixture-of-Experts）分别建模心电波形和心律，并结合使用低秩适应性（LoRA）的分层融合网络以实现高效的推理。
### Conclusion
ECG-MoE在五个公共临床任务中达到了最先进的性能，并且比多任务基线的推理速度提高了40%。
## 8. `cs.AI` - 通过多智能体系统发现数学概念 [PDF](https://arxiv.org/pdf/2603.04528), [HTML](https://arxiv.org/abs/2603.04528)
### Authors
Daattavya Aggarwal,Oisin Kim,Carl Henrik Ek,Challenger Mishra
### Background
数学概念的形成是实验、证明尝试和反例之间相互作用的过程。本文基于这一观察，提出了一种新的多智能体模型，用于基于计算的数学发现。
### Innovation
该系统提出了自己的猜想，并尝试证明它们，其决定受到反馈和不断变化的数据分布的影响。系统通过尝试自主恢复来自多面体数据和线性代数知识的同调概念，展示了其能力。最重要的是，实验是消融测试，通过统计测试完整动态的有效性，控制实验设置。这些实验支持了文章的主要主张：优化适当组合的本地过程可以导致意外一致的数学有趣性。
### Conclusion
该系统的实验结果支持主要主张：正确组合本地过程的优化可以产生非常一致的数学趣味性观念。
## 9. `cs.AI` - LLM代理的自适应内存准入控制 [PDF](https://arxiv.org/pdf/2603.04549), [HTML](https://arxiv.org/abs/2603.04549)
### Authors
Guilin Zhang,Wei Jiang,Xiejiashan Wang,Aisha Behr,Kai Zhao,Jeffrey Friedman,Xu Chu,Amine Anoun
### Background
基于LLM的代理逐渐依赖于长期记忆来支持多会话推理和交互，然而当前系统很少提供控制哪些信息被保留的方法。实践中，代理要么积累了大量的对话内容，包括虚幻或过时的事实，要么依赖于不透明的、完全由LLM驱动的记忆策略，这些策略昂贵且难以审核。因此，内存准入仍然是代理架构中一个疏忽严重且控制薄弱的组成部分。
### Innovation
提出了自适应内存准入控制（A-MAC），将内存准入视为一个结构化的决策问题。A-MAC将内存价值分解为五个互补且可解释的因素：未来效用、事实置信度、语义新颖性、时间新颖性以及内容类型先验。该框架结合了轻量级的基于规则的特征提取和一个单一的LLM辅助效用评估，并通过交叉验证优化学习领域适应性的准入策略。这一设计实现了对长期记忆的透明和高效的控制。
### Conclusion
在LoCoMo基准上的实验表明，A-MAC实现了优越的精确召回权衡，其F1值达到0.583，相比最先进的LLM本地记忆系统延迟降低了31%。消融结果显示内容类型先验是最具影响力的因素，这些结果表明显式且可解释的准入控制是可扩展且可靠的LLM基础代理中关键的设计原则。
## 10. `cs.AI` - 能力门槛与制造拓扑：实体人工智能如何引发经济地理的相变 [PDF](https://arxiv.org/pdf/2603.04457), [HTML](https://arxiv.org/abs/2603.04457)
### Authors
Xinmin Fang,Lingfeng Tao,Zhengxiong Li
### Background
自亨利·福特1913年的移动装配线以来，制造业的基本拓扑结构没有发生过范式级的转变。过去一个世纪的每一个重大创新，从丰田生产系统到第四次工业革命，都在福特范式内优化，而未改变其结构逻辑：集中在劳动力集群附近的大型工厂，大规模生产。本文认为，嵌入式人工智能即将打破这一百年来的僵局，不是通过使现有工厂更高效，而是通过触发制造业经济地理的相变。当嵌入式人工智能的能力在灵巧性、通用性、可靠性和触觉-视觉融合等方面跨越关键阈值时，其影响远不止成本降低：它们重新定义了工厂的位置、供应链的组织方式以及可行的生产规模。
### Innovation
本文提出了一个能力空间 C = (d, g, r, t)，并通过权重倒置、批次合并和人-基础设施脱钩三个途径，展示了嵌入式智能使需求附近的微制造成为可能、消灭“制造业荒漠”并逆转因劳动力替代而驱使的地理集中的创新观点。还引入了机器气候优势的概念，一旦人类工人被移除，最优工厂位置将由机器优化条件（低湿度、高辐照度、热稳定性）决定，这是不同于传统选址逻辑的因素，创造了前所未有的生产地理。
### Conclusion
本文奠定了实体人工智能经济学的基础，研究物理人工智能能力阈值如何重塑生产和空间结构逻辑。
## 11. `cs.CL` - AILS-NTUA在SemEval-2026任务3上的高效维度方面情感分析 [PDF](https://arxiv.org/pdf/2603.04933), [HTML](https://arxiv.org/abs/2603.04933)
### Authors
Stavros Gazetas,Giorgos Filandrianos,Maria Lymperaiou,Paraskevi Tzouveli,Athanasios Voulodimos,Giorgos Stamou
### Background
本文介绍了AILS-NTUA系统，用于SemEval-2026任务3中的DimABSA（维度方面基于情感分析）赛道，该系统包含三个互补问题：维度方面情感回归（DimASR）、维度方面情感三元组提取（DimASTE）和维度方面情感四元组预测（DimASQP），这一切都在一个多语言、多领域框架下实现。
### Innovation
方法论包括使用语言合适的编码器骨干进行微调，以进行连续方面的层次情感预测，以及使用LoRA对大型语言模型进行特定于语言的指令微调，以进行结构化的三元组和四元组提取。这一统一且适应任务的设计强调了语言和领域参数高效的专业化，从而降低了训练和推理需求，同时保持了强大的有效性。
### Conclusion
实验证明，所提出的模型在大部分评估设置中达到了竞争性性能，并且一致地超越了提供的基准模型。
## 12. `cs.CL` - Free Lunch for Pass@$k$? Low Cost Diverse Sampling for Diffusion Language Models [PDF](https://arxiv.org/pdf/2603.04893), [HTML](https://arxiv.org/abs/2603.04893)
### Authors
Sean Lamont,Christian Walder,Paul Montague,Amir Dezfouli,Michael Norrish
### Background
在复杂推理任务（如代码生成和数学问题解决）中，文本生成的多样输出对于有效探索是必不可少的。传统的采样方法往往因重复失败模式而导致计算资源浪费。扩散语言模型作为一种与自回归范式竞争的替代方案，仍然容易出现这种冗余现象，独立样本频繁地落入相似模式。
### Innovation
本文提出了一种无需训练且低成本的干预措施，以增强扩散语言模型的生成多样性。该方法按顺序修改批次中的中间样本，每个样本都被排斥出前一个样本的特征空间，从而积极惩罚冗余。与需要重新训练或使用束搜索的方法相比，该策略几乎不增加任何计算开销，同时确保每个样本都为批次贡献了独特的视角。
### Conclusion
我们的方法在HumanEval和GSM8K基准上使用LLaDA-8B-Instruct模型进行评估，结果显示在各种温度设置下都显著改善了多样性与Pass@$k$性能。作为一个简单的采样过程修改，我们的方法可以立即且低成本地提升当前和未来的扩散语言模型在需要多样解空间搜索任务中的表现。该方法的代码可在指定的URL获取。
## 13. `cs.CL` - MPCEval: 一个多方对话生成的基准 [PDF](https://arxiv.org/pdf/2603.04969), [HTML](https://arxiv.org/abs/2603.04969)
### Authors
Minxing Zhang,Yi Yang,Zhuofan Jia,Xuan Yang,Jian Pei,Yuchen Zang,Xingwang Deng,Xianglong Chen
### Background
多方对话生成，如智能回复和协作助手，是生成AI日益重要的能力，但其评估仍然是一个关键瓶颈。与两方对话相比，多方设置引入了独特的挑战，包括复杂的轮流发言、角色依赖的发言行为、长范围的对话结构以及多个同等有效的后续继续。因此，本文提出了MPCEval，这是一个面向任务的多方对话生成评估和基准套件。
### Innovation
MPCEval通过将生成质量分解为说话人建模、内容质量和说话人-内容一致性，并明确区分局部下一句话预测与全局完整对话生成，解决了多方对话评估的瓶颈。它提供了新颖的、量化的、无需参考并且可再现的评价指标，这些指标适用于跨数据集和模型。MPCEval被应用于多种公共和真实世界的数据集，并评估了现代生成方法与人类撰写的对话。
### Conclusion
MPCEval揭示了参与平衡、内容进展与新颖性以及说话人-内容一致性在各维度上的系统性模型特征。结果表明，评估目标对模型评估至关重要，单一分数的评估掩盖了多方对话行为中的根本差异。MPCEval的实现和相关评价代码已公开发布。
## 14. `cs.CL` - LocalSUG：面向本地生活服务的地理感知LLM查询建议 [PDF](https://arxiv.org/pdf/2603.04946), [HTML](https://arxiv.org/abs/2603.04946)
### Authors
Jinwen Chen(1 and 2),Shuai Gong,Shiwen Zhang(1 and 2),Zheng Zhang,Yachao Zhao,Lingxiang Wang(1 and 2),Haibo Zhou,Yuan Zhan,Wei Lin,Hainan Zhang(1 and 2) ((1) Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, (2) School of Artificial Intelligence, Beihang University, China)
### Background
在本地生活服务平台上，查询建议模块根据用户的输入前缀生成候选查询，以提升用户体验，减少用户操作并加快搜索速度。传统的多阶段瀑布式系统主要依赖历史热门查询，这限制了它们对长尾需求的应对能力。虽然语言模型（LLM）具有强大的语义泛化能力，但在本地生活服务中的部署也面临三大挑战：缺乏地理网络、自回归生成中的曝光偏差以及在线推理延迟。为解决这些问题，本文提出了一种基于LLM的查询建议框架LocalSUG，特别适合本地生活服务平台。
### Innovation
LocalSUG框架提供了两种创新方法：一是引入基于词项共现的城市感知候选挖掘策略，以增强生成过程中的地理定位。二是提出了一种基于束搜索的GRPO算法，通过使训练过程中奖励机制与推理时解码对齐来减少自回归生成中的曝光偏差。此外，还开发了质量感知束加速和词汇表缩减技术，这可以显著减少在线推理延迟，同时保持生成质量。多目标奖励机制进一步优化了相关性和业务目标指标。
### Conclusion
广泛的离线评估和大规模在线A/B测试结果表明，LocalSUG可以提高点击率(CTR)0.35%，并降低低/无结果率2.56%，证实其在实际部署中的有效性。
## 15. `cs.CL` - LLMs能否捕捉专家的不确定性？一项关于文化质性研究中价值观对齐的比较分析 [PDF](https://arxiv.org/pdf/2603.04897), [HTML](https://arxiv.org/abs/2603.04897)
### Authors
Arina Kostina,Marios Dikaiakos,Alejandro Porcel,Tassos Stassopoulos
### Background
在民族志和经济研究中，开放性采访的定性分析是一个关键步骤，它揭示了个人的价值观、动机以及文化嵌入的金融行为。现有大语言模型（LLMs）虽然提供了自动化和丰富人类理解和分析的潜力，但它们在任务固有的模糊性中产生细致、可靠解释的能力仍不明确。本文通过评估LLMs在基于Schwartz基本价值观框架识别长篇采访中表达的前三个人类价值观表现来研究这一问题。
### Innovation
本文首次通过Schwartz基本价值观框架评估多种LLMs，不仅分析了它们的表现和不确定性模式，还揭示了在文化质性研究中预测人类价值观排名的局限性。特别是，通过比较不同模型的输出和不确定性结构，发现了一些系统性的价值观偏好，这揭示了LLMs可能带来的互补视角以及需更深入研究模型带来的价值观偏差。
### Conclusion
研究结果表明，LLMs在基于集合的指标上接近人类表现极限，但在恢复精确的价值排序时表现不佳。尽管大多数模型在Schwartz价值观分布上与人类分析师相当，但在不确定性结构上存在差异。Qwen模型最接近专家级共识，展现出最强的专家Schwartz价值观分布对齐。LLM蒙特卡洛方法在多个指标上表现出一致的改进，总体多数投票和Borda计分法效果最佳。这些发现强调了LLMs在复杂且有歧义的价值分析中既有潜力也存在局限性。
## 16. `cs.CL` - 回放预训练数据提高微调效果 [PDF](https://arxiv.org/pdf/2603.04964), [HTML](https://arxiv.org/abs/2603.04964)
### Authors
Suhas Kotha,Percy Liang
### Background
当前的语言模型训练范式是在大量通用网络文本上预训练，然后在相对有限的目标数据上进行微调。通用数据通常只在微调过程中混入，以防止对通用领域的遗忘。本文发现，在微调过程中回放通用数据实际上可以提高对（不相关性更强的目标任务）的表现。
### Innovation
本文的主要创新在于提出了回放在微调过程中的通用数据，并发现在特定条件下这种方法可以显著提高目标数据效率，特别是在预训练阶段数据较少的情况下，回放通用数据能更好地帮助模型学习。
### Conclusion
研究结果表明，回放预训练数据可以在不相关性较强的目标任务上改善模型性能。具体地，在微调8亿参数模型时，回放预训练数据可以提高阿格内特网络导航成功率为4.5%，提高巴斯克语问答准确率为2%。
## 17. `cs.CL` - FireBench：评估企业级和API驱动的LLM应用中的指令遵循能力 [PDF](https://arxiv.org/pdf/2603.04857), [HTML](https://arxiv.org/abs/2603.04857)
### Authors
Yunfan Zhang,Yijie Bei,Jetashree Ravi,Pawel Garbacki
### Background
在企业级和API驱动的环境下，指令跟随对于LLM至关重要，因为它涉及到严格遵守输出格式、内容限制和程序要求，这对于确保可靠的工作流至关重要。现有的指令跟随基准主要评估反映聊天助手需求而非企业用户需求的自然语言生成约束。FireBench填补了这一空白，它基于真实的企业级和API使用模式，评测机器学习模型在多种应用场景中的指令遵循能力。
### Innovation
FireBench是一个新基准，它专注于评测企业在多种应用中（如信息提取、客户服务和代码代理）的指令遵循能力。它包含超过2400个样本，评估了11种不同的LLM，并提供了重要发现。FireBench被开源，以便用户评估模型适用性，支持模型开发者诊断性能问题，并邀请社区贡献。
### Conclusion
FireBench的贡献在于填补了现有基准的空白，提供了一个更符合企业用户实际需求的评测工具，可以帮助用户评估模型适用性，支持模型开发者改进性能，并鼓励社区贡献样本和改进工具。
## 18. `cs.CL` - AILS-NTUA在SemEval-2026任务10中的研究：有自主性的LLM用于心理语言标记提取和共谋认同检测 [PDF](https://arxiv.org/pdf/2603.04921), [HTML](https://arxiv.org/abs/2603.04921)
### Authors
Panagiotis Alexios Spanakis,Maria Lymperaiou,Giorgos Filandrianos,Athanasios Voulodimos,Giorgos Stamou
### Background
该研究针对SemEval-2026任务10，旨在从文本中联合抽取心理语言学共谋标记并检测共谋认同。传统分类器将语义推理与结构定位混淆在一起，而该研究采用了解除耦合的设计，将这两个挑战独立处理。
### Innovation
该研究提出了一个名为Dynamic Discriminative Chain-of-Thought (DD-CoT)的新颖方法，通过确定性的锚定解决语义模糊性和字符级别的薄弱性。此外，还提出了‘反回音室’架构，包括对抗平行理事会和经过校准的法官，以克服模型误判客观报道的‘记者陷阱’。
### Conclusion
该研究实现了S1上的0.24宏F1分数（相对于基线提高100%）和S2上的0.79宏F1分数（提高了49%），并指出其系统在开发排行榜中位列第三。这种方法确立了可解释的心理语言学基础的NLP的范式。
## 19. `cs.CL` - 当弱大语言模型充满自信时，偏好对齐变得更加强大 [PDF](https://arxiv.org/pdf/2603.04968), [HTML](https://arxiv.org/abs/2603.04968)
### Authors
Amirabbas Afzali,Myeongho Jeon,Maria Brbic
### Background
偏好对齐是将大型语言模型（LLMs）与人类价值观进行适配的重要步骤，但现有方法通常依赖昂贵的人工注释或大规模API基模型。因此，探索是否可以利用弱大语言模型作为有效的注释器成为一个研究问题。
### Innovation
研究发现，选择弱大语言模型中高置信度的样本子集即可获得比全面人工注释更好的性能。据此，提出了基于置信加权偏好优化（CW-PO）的框架，该框架通过对训练样本施加弱大语言模型的置信度权重来重新加权，适用于不同类型的偏好优化目标。实验结果表明，使用20%的人标注数据通过CW-PO得到的模型表现优于使用100%人工标注数据得到的模型。
### Conclusion
研究表明，将弱大语言模型与置信加权结合使用，不仅能够大幅度降低偏好对齐的成本，甚至还能在某些情况下超越完全基于人类标注数据训练的方法。
## 20. `cs.CL` - 联邦异构语言模型优化在混合自动语音识别中的应用 [PDF](https://arxiv.org/pdf/2603.04945), [HTML](https://arxiv.org/abs/2603.04945)
### Authors
Mengze Hong,Yi Gu,Di Jiang,Hanlin Gu,Chen Jason Zhang,Lu Wang,Zhiyang Su
### Background
随着自动语音识别（ASR）模型训练越来越依赖分散的联邦学习来保证数据隐私和可访问性，产生了多个本地模型，需要有效合并。在混合ASR系统中，虽然语音模型可以使用现有方法合并，但用于重新评分的N-best语音识别列表的语言模型（LM）面临着由于非神经n-克模型和神经网络模型异构性带来的挑战。
### Innovation
提出了一个异构LM优化任务，并引入了匹配-合并范式，其中包括两种算法：使用遗传操作来进化和配对LM的遗传匹配-合并算法（GMMA）；利用强化学习实现高效收敛的强化匹配-合并算法（RMMA）。实验结果显示RMMA在七个OpenSLR数据集上的平均字符错误率最低，且具有更好的泛化能力，比基线算法快7倍的速度达到收敛。
### Conclusion
匹配-合并范式在分布式环境中具有很高的潜力，为构建可扩展且隐私保护的ASR系统提供了新的可能。
## 21. `cs.LG` - 我们真的需要置换吗？模型宽度对线性模式连通性的影响 [PDF](https://arxiv.org/pdf/2510.08023), [HTML](https://arxiv.org/abs/2510.08023)
### Authors
Akira Ito,Masanori Yamada,Daiki Chijiwa,Atsutoshi Kumagai
### Background
先前的研究表明，给定两个独立训练的模型，通过保持输入与输出行为不变的方式进行参数置换可以使这两模型通过一个低损失的线性路径相连，这一现象被称为线性模式连接（LMC）。虽然这样的路径一旦存在，表明模型在某些方面是等效的，但实现在LMC的过程中不仅需要合适的置换搜索，还需要足够宽的模型，例如，对于ResNet-20模型，需要32倍的宽度倍增。这种现象认为是由于增加模型宽度增加了候选置换的数量，从而增加了找到一个能够实现LMC的置换的可能性。
### Innovation
本研究展示了即使不进行任何置换，单独增加模型宽度，通过合适的softmax温度校准，就可以实现LMC。研究者进一步通过分析中间层输出解释了这种现象，并引入了层间的指数加权连通（LEWC）理论，即合并模型的每一层输出可以被表示为原始模型中对应的每一层输出的指数加权和，从而使得合并后的模型输出与原始模型的集成输出一致，增强了线性模式连通性的可能性。
### Conclusion
这项工作是首次表明，不仅仅模型宽度扩大促进了非线性模式连通性，同时显著增加了实现线性模式连通性的可能性。
## 22. `cs.LG` - 在多智能体系统中击破并修复针对控制流劫持的防御 [PDF](https://arxiv.org/pdf/2510.17276), [HTML](https://arxiv.org/abs/2510.17276)
### Authors
Rishi Jha,Harold Triedman,Justin Wagle,Vitaly Shmatikov
### Background
控制流劫持攻击操控多智能体系统中的协调机制，以执行不安全的操作，损害系统并泄露敏感信息。最近提出的一些防御措施，如LlamaFirewall，依赖高级语言模型执行的通信对齐检查来确保所有智能体调用都“相关于”且“有助于”原始目标。作者指出，多智能体系统中的安全性与功能性目标本质上是冲突的，而“对齐”的模糊定义和检查器对执行上下文的不完全可见性进一步加剧了这一冲突。
### Innovation
作者提出的ControlValve是一种新的防御措施，灵感源自控制流完整性原则和最小权限原则。ControlValve（1）为多智能体系统生成允许的控制流图，并（2）强制所有执行必须遵循这些图，同时还为每个智能体调用生成基于零样本的上下文规则。
### Conclusion
研究表明，ControlValve能够有效识别并防御控制流劫持攻击，同时解决了现有防御的冲突问题。
## 23. `cs.LG` - LLEMA：带有LLMs的多目标材料发现的进化搜索 [PDF](https://arxiv.org/pdf/2510.22503), [HTML](https://arxiv.org/abs/2510.22503)
### Authors
Nikhil Abhyankar,Sanchit Kabra,Saaketh Desai,Chandan K. Reddy
### Background
材料发现需要探索广泛的化学和结构空间，并满足多种通常相互冲突的目标。在这一背景下，提出了一种新的框架——LLM引导进化（LLEMA），结合了大型语言模型中的科学知识、化学导向的进化规则以及基于记忆的改进方法。
### Innovation
LLEMA框架通过将大型语言模型的科学知识与化学导向的进化规则和基于记忆的改进方法相结合，提出了一种新颖的方法来指导材料发现过程。每一代过程中，LLM会基于明确的属性约束性提出晶体学指定候选；使用增强代理优化심乘物理化学属性；多目标评分器更新成功/失败记忆来指导后续代数。研究结果表明，LLEMA在14项涵盖电子、能源、涂层、光学和航空航天的现实任务中表现出更高的命中率和改进的帕累托前沿质量，相比生成模型和仅大型语言模型基线方法有显著改进。
### Conclusion
去除了规则导向生成、基于记忆的改进和代理预测，消融研究表明这些组件的重要性。通过确保合成性和多目标权衡，LLEMA为加速实际材料发现提供了一种基础性的方法。
## 24. `cs.LG` - OPPO: 通过管道重叠加速基于PPO的RLHF [PDF](https://arxiv.org/pdf/2509.25762), [HTML](https://arxiv.org/abs/2509.25762)
### Authors
Kaizhuo Yan,Yingjie Yu,Yifan Yu,Haizhong Zheng,Fan Lai
### Background
Proximal Policy Optimization (PPO)-基于的强化学习从人类反馈（RLHF）是一个广泛采用的框架，用于使大型语言模型（LLM）与人类偏好对齐。然而，它在训练过程中存在显著的效率低下问题，原因包括多模型的顺序依赖性（例如，奖励模型依赖于演员模型的输出）和响应长度的长尾效应，导致一些长响应延迟完成。
### Innovation
OPPO是一个新颖、轻量级且模型无关的PPO基于的RLHF框架，通过重叠训练流水线来提高训练效率。它引入了两种创新技术：（1）步骤内重叠，将上游模型输出（如演员模型）以适当的大小分块流式传输，使下游模型（如奖励）可以在上游模型继续解码时进行预填充；（2）步骤间重叠，适当地承担一些提示并在未来步骤中延迟长生成，从而减少尾部延迟而不丢弃部分工作。该框架与现有PPO实现易于集成，并且不需要额外的复杂性。
### Conclusion
广泛评估表明，与PPO基于的RLHF训练相比，OPPO可以加速训练$1.8times$--$2.8times$，同时提高GPU利用率$1.4times$--$2.1times$，且不牺牲训练收敛性。
## 25. `cs.LG` - SPOT: 单射频孔径可调波束形成本域定位 [PDF](https://arxiv.org/pdf/2511.11391), [HTML](https://arxiv.org/abs/2511.11391)
### Authors
Yeyue Cai,Jianhua Mo,Meixia Tao
### Background
相位时间阵列（phase-time arrays）结合了相移器（PSs）和真时延时间控制器（TTDs），成为宽带传感和定位中生成频率依赖的彩虹波束的经济有效架构。在该文中，提出了一种端到端的基于深度学习的方法，能够同时设计彩虹波束并估计用户位置，这是一种新颖的方法。
### Innovation
将相移器和时延控制器的系数作为可训练变量，使得网络能够合成旨在最大化定位精度的任务导向波束。轻量级的全连接模块从单下行传输的最大量化接收到的功率及其对应的子载波索引的反馈中恢复用户的角度-范围坐标，相比现有的分析和基于学习的方法，该方法的开销减少了十倍，并且在二维定位误差上表现更优。
### Conclusion
提出的基于深度学习的方法大大降低了开销，并且能够持续地提供比现有方法更低的二维定位误差。
## 26. `cs.LG` - TabStruct: 测量表格数据的结构忠实度 [PDF](https://arxiv.org/pdf/2509.11950), [HTML](https://arxiv.org/abs/2509.11950)
### Authors
Xiangjian Jiang,Nikola Simidjievski,Mateja Jamnik
### Background
评估表格生成器仍然是一个具有挑战性的问题，这是因为异质表格数据的独特因果结构先验不适合直观的人类检查。现有工作引入了结构忠实度作为具有特定表格数据评估维度，用于评估合成数据是否符合真实数据的因果结构。然而，现有的基准通常忽视了结构忠实度与传统评估维度之间的相互作用，导致无法提供对模型性能的全面理解。此外，现有的基准通常仅限于玩具数据集，因为量化现有的结构忠实度度量要求访问真实的因果结构，而这在现实世界的数据集中很少见。
### Innovation
本文提出了一种新的评估框架，该框架同时考虑了结构忠实度和传统评估维度。此外，引入了一个新的评估指标——全局实用性，即使在缺乏真实因果结构的情况下，也可以评估结构忠实度。同时，呈现了TabStruct，这是一个全面的评估基准，提供了29个不同数据集上的13个表格生成器的大规模定量分析。研究表明，全局实用性提供了一个与任务无关且与领域无关的解释单元格生成器性能。
### Conclusion
我们发布了TabStruct基准套件，包括所有数据集、评估管道和原始结果。代码可在该链接获取。
## 27. `cs.LG` - ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting [PDF](https://arxiv.org/pdf/2511.11991), [HTML](https://arxiv.org/abs/2511.11991)
### Authors
Xiang Ma,Taihua Chen,Pengcheng Wang,Xuemei Li,Caiming Zhang
### Background
时间序列预测在各个领域都至关重要。传统方法通常通过全局分解为趋势、季节性和残差分量来进行，但对于以局部、复杂和高度动态模式为主导的实际序列变得无效。此外，这类方法的高度模型复杂度限制了其在实时或资源受限环境中的应用。
### Innovation
本文提出了一种新颖的REliability-aware Codebook-ASsisted Time系列预测框架（ReCast），通过利用重复的局部形状实现轻量级且鲁棒的预测。ReCast通过使用可学习码簿进行局部模式的块状量化编码，精简地捕捉稳定规律结构。为补偿量化过程中未保留的残差变化，ReCast采用了一种双路径结构，包括用于高效建模规律结构的量化路径和用于重建不规则波动的残差路径。ReCast的核心贡献是一种可靠性驱动的码簿更新策略，通过加权修正增量性地优化码簿。这些修正权重通过结合来自互补视角的多种可靠性因素通过分布式鲁棒优化（DRO）方案融合，确保适应性和模型鲁棒性。
### Conclusion
广泛的实验证明了ReCast在准确度、效率以及适应分布迁移方面的优越性，超过了最先进的（SOTA）模型。
## 28. `cs.LG` - 非渐近分析条件下约束化回归效率 [PDF](https://arxiv.org/pdf/2510.07093), [HTML](https://arxiv.org/abs/2510.07093)
### Authors
Yunzhen Yao,Lie He,Michael Gastpar
### Background
约束化预测通过提供具有覆盖保证的预测集而被广泛应用。这种方法的有效性量化的常见方式是预期预测集大小。先前关于约束化回归效率的工作通常将误覆盖水平 α 看作是一个固定的常数。这篇论文改进了这个方法，通过研究约束化分位数和中位数回归（使用 SGD 训练）在数据分布温和假设下的非渐进偏差界，探讨了预测集长度偏离最优范围与样本大小以及误覆盖水平之间的关系。
### Innovation
论文推导了约束化分位数和中位数回归在使用 SGD 训练下的非渐进偏差界，强调了三个关键因素——样本集大小 n、校准集大小 m 和误覆盖水平 α——对预测集效率的影响。边界反映了这三个因素之间的复杂依赖关系，包括特定的阶数 Θ(1/√n + 1/(α^2 n) + 1/√m + ∅(-α^2 m))。
### Conclusion
研究结果揭示了不同 α 计划下的收敛率的相变现象，为调整数据分配以控制过长预测集长度提供指导。实验证据支持了理论发现。
## 29. `cs.LG` - 复杂性正则化近端策略优化 [PDF](https://arxiv.org/pdf/2509.20509), [HTML](https://arxiv.org/abs/2509.20509)
### Authors
Luca Serfilippi,Giorgio Franceschelli,Antonio Corradi,Mirco Musolesi
### Background
政策梯度方法通常依赖熵正则化以防止过早收敛。然而，单纯的熵最大化会将策略推向均匀分布，这可能导致丢弃奖励信号，尤其是在正则化参数未最优调整的情况下。该研究提出用自调节复杂性项替代标准熵项，定义为香农熵与偏离均匀分布度量的乘积，后者量化了远离均匀分布的距离。与单纯的熵（偏好最大无序度）不同，这种复杂性度量在完全确定性和完美均匀分布时都为零，而在那些表现出有序与随机性有机结合的系统中严格为正。这些特性确保了在策略高度不确定时，保持有益的随机性的同时减少正则化压力，这允许学习更专注于奖励优化。
### Innovation
提出了一种新的复杂性正则化近端策略优化方法（CR-PPO），其核心是用自调节复杂性项代替传统的熵正则化。CR-PPO方法在使用过程中表现出了更稳健的超参数选择特性，即使在从非常小到非常大的正则化系数范围内，也能保持一致的性能，并且不会在不必要的正则化下导致不良影响，从而减少了昂贵的超参数调整需求。
### Conclusion
实验结果表明，CR-PPO在不同规模的正则化系数范围内更稳健，具有更好的鲁棒性，并且在不需要正则化时不会导致负面影响，从而减少对超参数的精细调整需求。
## 30. `cs.LG` - 理解潜意识学习：何时以及如何转移隐藏偏见 [PDF](https://arxiv.org/pdf/2509.23886), [HTML](https://arxiv.org/abs/2509.23886)
### Authors
Simon Schrodi,Elias Kempf,Fazl Barez,Thomas Brox
### Background
语言模型在知识蒸馏过程中可能会转移隐藏的偏见。例如，一个喜欢猫的教师会使它的学生也倾向于喜欢猫，即便训练数据只是数字列表。这种现象被称为潜意识学习。潜意识学习可以在软蒸馏过程中预期到，当学生接受教师完整的下一个标记分布时。但这一现象在硬蒸馏中也发生-学生只看到样化的标记时-引出了更深的问题：潜意识学习究竟何时以及如何发生？
### Innovation
本文通过控制实验和机械分析回答了这个问题。研究表明，潜意识学习并不需要(全局)标记纠缠或logit泄漏，而是依赖于一小部分分歧标记——教师有不同的偏见时可能会预测不同的标记。屏蔽这些标记主要可以消除隐藏偏见的转移。此外发现，早期层对于潜意识学习至关重要，并且即使是单独微调一个如此早期的层，也足以实现潜意识学习。最后，潜意识学习是脆弱的，即使是小的改变，如提示重述，通常也足以抑制潜意识学习。
### Conclusion
研究结果显示，潜意识学习不需要全局标记纠缠或logit泄漏，而是一些分歧标记造成的结果。这些分歧标记表明早期层对于影响潜意识学习至关重要，且微调一个这样的早期层就足以实现潜意识学习。此外，潜意识学习是脆弱的，小的改变足以抑制它。
