# 20260216
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 关于决策值映射和表示依赖性 [PDF](https://arxiv.org/pdf/2602.11295), [HTML](https://arxiv.org/abs/2602.11295)
### Authors
Gil Raitses
### Background
不同的数据表示方式可以导致计算引擎产出不同的离散结果。某些表示方式保留结果，而某些则完全改变结果。决策值映射记录了哪些表示方式能够保留结果以及哪些会改变它，关联每个声明的表示族与它所产生的离散结果。
### Innovation
本文将决策值映射形式化，并描述了DecisionDB基础设施，该基础设施使用从内容和以写一次形式存储的 artefacts计算的标识符来记录、重放和审计这些关系。这种确定性重放能够精确地从存储的 artefacts中恢复每个记录的决策标识符，所有三个标识字段与持久化值完全匹配。贡献包括将表示空间划分为持久性区域与边界，并将决策重用视为可机械检验的条件。
### Conclusion
本文通过形式化决策值映射和构建DecisionDB基础设施，提供了一种方法来管理和验证不同表示之间的依赖关系，这对于记录、重放和审计计算结果尤其重要。
## 2. `cs.AI` - 剖析数据注释中的主观性与‘真实’事实幻觉 [PDF](https://arxiv.org/pdf/2602.11318), [HTML](https://arxiv.org/abs/2602.11318)
### Authors
Sheza Munir,Benjamin Mah,Krisha Kalsi,Shivani Kapania,Julian Posada,Edith Law,Ding Wang,Syed Ishtiaque Ahmed
### Background
本文探讨了机器学习中“真实”事实及其相关研究，指出传统的“真实”事实假设将人类分歧视为技术噪音，忽视了其作为社会技术信号的重要性。通过对2020年至2025年间发表在七家顶级期刊上的文献进行系统综述，研究发现，数据注释实践中的机制导致了“共识陷阱”，并揭示了结构化失败和最近的人机协作转变对人类观点的影响。
### Innovation
研究创新在于揭示了地理霸权如何将西方规范强加为普遍标准，以及其对脆弱的数据工人造成的影响。进一步批评了“噪音传感器”谬误，指出应重新评估文化多元性在统计模型中的重要性。并提出了一条多元注释基础设施的蓝图，旨在从寻找单一“正确”答案转向映射人类经验的多样性。
### Conclusion
本文通过系统文献综述和反思性主题分析，指出需要调整数据注释基础，从发现单一“正确”答案转向理解多元人类经验，并提出了建设文化敏感型模型的方法。
## 3. `cs.AI` - AgentNoiseBench: 面对噪声条件下的工具使用大语言模型代理稳健性评估 [PDF](https://arxiv.org/pdf/2602.11348), [HTML](https://arxiv.org/abs/2602.11348)
### Authors
Ruipeng Wang,Yuxin Chen,Yukai Wang,Chang Wu,Junfeng Fang,Xiaodong Cai,Qi Gu,Hui Su,An Zhang,Xiang Wang,Xunliang Cai,Tat-Seng Chua
### Background
近年来，大规模语言模型（LLM）的进步使得基于LLM的代理在各种基准测试中表现出强大的性能。然而，在实际部署中，这些代理的表现往往低于基准测试中的表现，尤其是在复杂和不完美的环境中。这种差异主要是由于当前的训练和评估范式大多基于理想的假设，忽略了真实交互中存在的固有随机性和噪音。
### Innovation
我们提出了AgentNoiseBench框架，该框架系统性地评估代理模型在嘈杂环境下的稳健性。我们首先对真实世界场景中的偏差和不确定性进行了深入分析，并将环境噪音分为两类：用户噪音和工具噪音。基于此分析，我们开发了一种自动流水线，在保持任务可解的同时向现有的以代理为中心的基准测试注入可控性噪音。利用该流水线，我们在具有不同架构和参数规模的广泛模型上进行了广泛的评估。结果显示，在不同噪声条件下的表现存在一致的变化，突显了当前代理模型对现实环境扰动的敏感度。
### Conclusion
我们的研究结果表明，代理模型的表现受现实环境扰动的影响显著，不同类型的噪音会对模型性能产生不同影响。这为改进代理模型的稳健性提供了重要指导。
## 4. `cs.AI` - 推动前瞻性代理最优行为前沿 [PDF](https://arxiv.org/pdf/2602.11351), [HTML](https://arxiv.org/abs/2602.11351)
### Authors
Yihang Yao,Zhepeng Cen,Haohong Lin,Shiqi Liu,Zuxin Liu,Jiacheng Zhu,Zhang-Wei Hong,Laixi Shi,Ding Zhao
### Background
现有的大型语言模型（LLM）代理旨在积极规划、查询和多轮次交互，以高效地完成任务，超越被动的指令遵循。这些代理对于实现以用户为中心的真实世界应用至关重要。代理强化学习（RL）作为一种在多轮交互设置中训练这类代理的潜在解决方案已经出现，它允许从反馈中学习交互策略。然而，现有的工作流程面临一个关键挑战，即平衡任务性能与用户参与度，因为被动代理无法高效地适应用户的意图，而过度依赖人类反馈则会降低用户的满意度。
### Innovation
为了解决这一权衡，本文提出了BAO，这是一种结合行为增强和行为正则化相结合的代理RL框架。BAO通过增强代理的前瞻性推理和信息收集能力，抑制了低效或冗余交互，并使代理行为与用户期望保持一致。在用户RL基准套件的多项任务上评估BAO，结果显示它在性能上明显优于前瞻性代理RL基线，甚至与商用的LLM代理具有同等或更优的表现，表明它在复杂多轮交互场景中训练前瞻性、用户对齐的LLM代理的有效性。
### Conclusion
BAO显著提高了前瞻性代理的表现，同时在复杂多轮交互场景中保持了良好的用户参与度，适用于现实世界的以用户为中心的应用。
## 5. `cs.AI` - ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences [PDF](https://arxiv.org/pdf/2602.11354), [HTML](https://arxiv.org/abs/2602.11354)
### Authors
Bang Nguyen,Dominik Soós,Qian Ma,Rochana R. Obadage,Zack Ranjan,Sai Koneru,Timothy M. Errington,Shakhlo Nematova,Sarah Rajtmajer,Jian Wu,Meng Jiang
### Background
现有文献对AI代理在科学研究论文的自动化评估方面产生了浓厚的兴趣。现有的基准主要集中在计算层面，测试了代理在能够访问代码和数据时重现或复制研究结果的能力。这种设定虽然基础，但（1）未能捕捉到在复制与再现之间的新数据可用性的不一致性，（2）缺乏通过仅关注可复制的论文来确保真实性而生成的真实性的多样性。此外，大多数基准仅评估结果而未考虑复制过程。
### Innovation
本文介绍了ReplicatorBench，一个端到端的基准测试，包括由人类验证的可复制和不可复制的研究声明，旨在评估AI代理的社会和行为科学中的研究复制能力，分为三个阶段：（1）复制数据的提取和检索；（2）设计和执行计算实验；（3）结果解释。为了设定AI代理的能力基线，本文开发了ReplicatorAgent框架，配备了必要的工具（如网络搜索和迭代与沙盒环境的交互）来完成ReplicatorBench中的任务，并在四个大型语言模型（LLMs）之上进行了评估，同时考虑了编程语言和代码访问级别的不同选择。
### Conclusion
我们的发现表明，现有的LLM代理能够有效地设计和执行计算实验，但在检索如新数据等资源时存在问题，这些资源是复制声明所必需的。所有代码和数据都已公开。
## 6. `cs.AI` - Latent生成求解器用于具有长期物理模拟的可泛化的模拟 [PDF](https://arxiv.org/pdf/2602.11229), [HTML](https://arxiv.org/abs/2602.11229)
### Authors
Zituo Chen,Haixu Wu,Sili Deng
### Background
该研究聚焦于跨异质偏微分方程（PDE）系统的长期代理模拟。传统方法难以在不同PDE系统之间进行有效的模拟和预测，尤其在长时间尺度下表现不佳。研究提出了一种新的方法，旨在解决这些问题。
### Innovation
该论文引入了一种名为Latent生成求解器（Latent Generative Solvers，LGS）的双阶段框架。该框架包括两个步骤：首先，使用预训练的VAE将各种PDE状态映射到共享的潜在物理空间；其次，通过流匹配训练的变换器学习潜在动力学。LGS的关键机制是在训练和推断过程中扰动潜在输入，以纠正轨迹偏离流形的现象，并稳定自回归预测。此外，流强迫机制用于从模型生成的轨迹更新系统描述符（上下文），从而提高长期稳定性和训练/测试条件一致性。
### Conclusion
LGS在短时间尺度上与强确定性神经算子基线相当，但长期轨迹中的漂移显著减少。潜在空间学习和高效的架构选择使得LGS相比非生成性基线减少高达70倍的FLOPs，从而支持了可扩展的预训练。此外，即使在有限的微调预算下，LGS也可高效适应异质的数据集。总体而言，LGS提供了一种实用途径，以实现具有长期可预测性与下游科学工作流适用性的、具有不确定性感知能力的神经PDE求解器。
## 7. `cs.AI` - 无需代码解释AI：无需代码的可解释人工智能用户研究 [PDF](https://arxiv.org/pdf/2602.11159), [HTML](https://arxiv.org/abs/2602.11159)
### Authors
Natalia Abarca,Andrés Carvallo,Claudia López Moncada,Felipe Bravo-Marquez
### Background
随着机器学习（ML）在敏感领域（如医疗、金融和公共政策）中的应用增加，人们对自动化决策透明度的担忧也随之上升。可解释人工智能（XAI）通过阐明模型预测生成过程来解决这一问题，但大多数方法都需要专业技术知识，这限制了其对初学者的价值。特别是在无代码ML平台中，这些平台旨在使人工智能民主化但很少包含解释性，这一缺口尤为关键。本文在开源无代码ML平台DashAI中提出了一个以用户为中心的XAI模块，该模块将部分依赖图、置换特征重要性和KernelSHAP三种互补技术集成到DashAI的工作流程中，用于处理表格分类问题。
### Innovation
该研究在开源无代码ML平台DashAI中引入了一种以人为中心的XAI模块，该模块将三种互补的技术——部分依赖图、置换特征重要性和KernelSHAP——整合到DashAI的工作流程中。这些技术帮助用户理解和信任基于ML模型的预测过程。
### Conclusion
用户研究证明，该XAI模块在初学者和专家中都能成功执行解释任务，初学者认为解释有用、准确和可信，而专家则更关注解释的充分性和完整性。解释可以提高用户对自动化的信任感，而初学者的信任感高于专家。这些发现突显了无代码ML中XAI的核心挑战：既要使解释易于初学者理解，又要让专家感到足够的详细和完整。
## 8. `cs.AI` - GHOST：通过分组隐态输出感知的选择与截断在Mamba2中揭示幽灵状态 [PDF](https://arxiv.org/pdf/2602.11408), [HTML](https://arxiv.org/abs/2602.11408)
### Authors
Michael Menezes,Anastasios Kyrillidis
### Background
Mamba2的扩展状态维度虽然提高了时间建模能力，但在自回归生成过程中产生了大量的推理开销，导致带宽饱和。标准的剪枝方法无法解决这一瓶颈：无结构稀疏度使激活保持稠密，基于幅度的选择忽略了运行时动态，而基于梯度的方法则带来了过高的成本。
### Innovation
提出了GHOST（分组隐态输出感知的选择与截断），这是一种结构化剪枝框架，仅使用前向传递统计量近似控制理论平衡截断。通过联合测量可控性和可观测性，GHOST在无需反向传播的情况下，与基于梯度的方法相比具有相同的精度。
### Conclusion
在从130M到2.7B参数的不同模型中，该方法实现了约50%的状态维度减少，同时在WikiText-2数据集上的困惑度增加了约1个单位。相关代码可从这里获得。
## 9. `cs.AI` - 恰如其分的荣誉：跨模态连接性推动精确的多模态大语言模型推理强化学习 [PDF](https://arxiv.org/pdf/2602.11455), [HTML](https://arxiv.org/abs/2602.11455)
### Authors
Zhengbo Jiao,Shaobo Wang,Zifan Zhang,Wei Wang,Bing Zhao,Hu Wei,Linfeng Zhang
### Background
Reinforcement Learning with Verifiable Rewards (RLVR)大幅提升了多模态大语言模型（MLLMs）的推理能力，但视觉证据如何在推理过程中集成仍不清楚。研究表明，只有少量（约15%）的tokens表现出强烈的视觉-文本耦合，这些高度连通的tokens在图像中提供了推理的锚点，而大多数tokens遵循语言模式。
### Innovation
我们提出了基于图聚类注意力拓扑的Anchor-Token RL（AT-RL），这是一种轻量级框架，通过选择性强化高连通性tokens来优化多模态RLVR。AT-RL仅引入了1.2%的开销，但显著提升了32B模型在MathVista上的表现，达到了80.2的分数，并在STEM、视频和一般任务上持续提高性能。低连通性tokens的单独训练会导致严重性能下降，强调了视觉锚精确激励对于多模态RL的重要性。
### Conclusion
我们的研究揭示，推理质量并不取决于tokens的数量，而是取决于跨模态锚定点的准确性，这意味着精确的credit assignment对于多模态RL至关重要。
## 10. `cs.AI` - 预算约束下的代理大型语言模型：基于意图的计划以应对昂贵工具的使用 [PDF](https://arxiv.org/pdf/2602.11541), [HTML](https://arxiv.org/abs/2602.11541)
### Authors
Hanbing Liu,Chunhao Tian,Nan An,Ziyuan Wang,Pinyan Lu,Changyuan Yu,Qi Qi
### Background
研究需要在严格预算约束下使用外部工具的工具增强型代理，特别是大型语言模型解决多步骤任务的情形。任务涉及的瞬时状态和动作空间巨大，每一步工具执行的结果具有高变异性，而探索所有可能的决策路径是成本高昂且不切实际的。
### Innovation
提出了INTENT，一种推理时间的规划框架，通过一个意图感知的层级世界模型来预测未来的工具使用情况、风险校准的成本，并在线引导决策。
### Conclusion
在成本增强的StableToolBench上，INTENT严格实现了预算约束的可行性，显著提高了任务的成功率，并在工具价格变化等动态市场变动下保持了 robust 性。
## 11. `cs.AI` - 基于人类启发式连续学习的内部推理过程学习：适应性人工智能系统的思考学习 [PDF](https://arxiv.org/pdf/2602.11516), [HTML](https://arxiv.org/abs/2602.11516)
### Authors
Hong Su
### Background
开发能够在动态现实环境中持续适应的AI系统时，了解其内部推理过程至关重要。然而，大多数现有方法主要关注学习特定任务的输出或静态知识表示，忽视了内部推理结构、行为调度策略和学习机制的持续优化。
### Innovation
本文提出了一种基于人类启发式的连续学习框架，该框架在增强的序列推理模型中统一了推理、行动、反思和验证。该框架将内部思考过程作为主要学习对象，并系统地记录内部推理过程和环境交互，构建结构化学习材料，以优化任务级内容、推理活动的组织、调度和进化。此外，框架支持预定义逻辑的可控替换和分层学习机制，该机制可以联合适应任务级参数和学习策略。
### Conclusion
实验结果表明，在温度传感器异常检测任务中，纳入内部过程学习可以将平均运行时间减少23.9%。系统在保持操作稳定性的前提下，其内部认知架构逐渐进化。
## 12. `cs.AI` - Causal-JEPA: Learn World Models through Object-Level Latent Interventions [PDF](https://arxiv.org/pdf/2602.11389), [HTML](https://arxiv.org/abs/2602.11389)
### Authors
Heejeong Nam,Quentin Le Lidec,Lucas Maes,Yann LeCun,Randall Balestriero
### Background
世界模型需要稳健的关系理解以支持预测、推理和控制。尽管基于对象的表示提供了一种有用的抽象，但它们不能捕捉到依赖于交互的动力学。因此，本文介绍了一种简单灵活的对象中心世界模型C-JEPA，它将掩蔽联合嵌入预测从图像片段扩展到了基于对象的表示。
### Innovation
C-JEPA通过在对象级上应用掩码，强制对象的状态需要从其他对象中推断，进而产生了潜在的干预，具有类似反事实的效果，并防止捷径解决方案，使得交互推理成为必须。实验上，与没有对象级掩码的相同架构相比，C-JEPA在反事实推理方面绝对提高了大约20%。对于代理控制任务，C-JEPA仅需基于图像片段的世界模型所需总潜在输入特征的1%，即可实现更加高效的规划，同时性能相当。
### Conclusion
我们通过潜在干预的形式证明了对象级掩码产生了因果归纳偏置。我们已经开源了代码。
## 13. `cs.AI` - Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge [PDF](https://arxiv.org/pdf/2602.11340), [HTML](https://arxiv.org/abs/2602.11340)
### Authors
Bo Pan,Xuan Kan,Kaitai Zhang,Yan Yan,Shunwen Tan,Zihao He,Zixin Ding,Junjie Wu,Liang Zhao
### Background
大型语言模型（LLMs）已被广泛用作评估AI生成内容的自动化裁判员。尽管取得成功，但将LLM的评估与人类判断对齐仍然具有挑战性。监督微调虽然可以改进对齐，但成本高且不灵活，需要为新任务或数据集重新训练。最近的自动提示优化（APO）进展提供了一种更有效的替代方案，可以自动改进指导LLM裁判的指令。然而，现有的APO方法主要针对文本评估，并在多模态环境中未被充分探索。本文研究了多模态LLM作为裁判员中的自动提示优化问题，特别是在评估AI生成的图像方面。
### Innovation
提出了一个多级提示优化框架BLPO，该框架将图像转换为文本表示，同时保留与评估相关的视觉线索，从而协同改进裁判提示和I2T提示，以在有限上下文预算下保持保真度。
### Conclusion
在四个数据集和三种LLM裁判员上进行的实验结果证明了该方法的有效性。
## 14. `cs.AI` - CausalAgent: 一个端到端因果推理的对话式多智能体系统 [PDF](https://arxiv.org/pdf/2602.11527), [HTML](https://arxiv.org/abs/2602.11527)
### Authors
Jiawei Zhu,Wei Chen,Ruichu Cai
### Background
因果推理在医疗保健、经济学和社会科学等领域具有巨大价值。然而，传统的因果分析工作流程带来了重大的技术障碍，要求研究人员同时具备统计和计算机科学的背景，在手动选择算法、处理数据质量问题和解释复杂结果方面投入大量精力。
### Innovation
我们提出了CausalAgent，一个对话式多智能体系统，用于端到端的因果推理。该系统创新性地结合了多智能体系统（MAS）、检索增强生成（RAG）和模型上下文协议（MCP），实现了从数据清理、因果结构学习到偏差修正和报告生成的自动化过程，通过自然语言交互进行。用户只需上传数据集并用自然语言提出问题，即可获得严谨且互动的分析报告。CausalAgent提供了一个新颖的以人为本的人机协作范式，通过使用交互式可视化，显著降低了因果分析的门槛，同时确保过程的严谨性和可解释性。
### Conclusion
CausalAgent 显著降低了因果分析的门槛，通过与用户的自然语言交互，实现自动化流程，同时保持了过程的严谨性和可解释性。
## 15. `cs.AI` - 通过稳健的价值因式分解实现分布鲁棒的合作多智能体强化学习 [PDF](https://arxiv.org/pdf/2602.11437), [HTML](https://arxiv.org/abs/2602.11437)
### Authors
Chengrui Qu,Christopher Yeh,Kishan Panaganti,Eric Mazumdar,Adam Wierman
### Background
多智能体强化学习（MARL）通常采用集中训练分散执行的框架，使用价值分解方法确保个体-整体最大原则（IGM），从而使得分散的贪婪动作恢复团队最优联合动作。然而，由于来自仿真到现实的差距、模型不匹配和系统噪声的环境不确定性，这种配方在实际应用场景中的可靠性不足。
### Innovation
提出了一种分布鲁棒的个体-全局最大原则（DrIGM），要求每个智能体的稳健贪婪动作与稳健的团队最优联合动作相一致。通过这一原则，重新定义了稳健的个体行动价值，实现分散贪婪执行并为整个系统提供可验证的稳健性保证。基于此基础，提出了遵守DrIGM稳健的价值分解架构（如VDN/QMIX/QTRAN的稳健版本），具有训练稳健的Q目标、保持可扩展性和与现有代码无缝集成等特点。
### Conclusion
在高保真SustainGym模拟器和StarCraft游戏环境中，该方法在分布外性能上表现更优。相关代码和数据可访问此链接：this https URL.
## 16. `cs.AI` - Voxtral 实时 [PDF](https://arxiv.org/pdf/2602.11298), [HTML](https://arxiv.org/abs/2602.11298)
### Authors
Alexander H. Liu,Andy Ehrenberg,Andy Lo,Chen-Yo Sun,Guillaume Lample,Jean-Malo Delignon,Khyathi Raghavi Chandu,Patrick von Platen,Pavankumar Reddy Muddireddy,Rohin Arora,Sanchit Gandhi,Sandeep Subramanian,Soham Ghosh,Srijan Mishra,Abhinav Rastogi,Alan Jeffares,Albert Jiang,Alexandre Sablayrolles,Amélie Héliou,Andrew Bai,Angele Lenglemetz,Anmol Agarwal,Anton Eliseev,Antonia Calvi,Arjun Majumdar,Baptiste Bout,Baptiste Rozière,Baudouin De Monicault,Benjamin Tibi,Clémence Lanfranchi,Connor Chen,Corentin Barreau,Corentin Sautier,Cyprien Courtot,Darius Dabert,Diego de las Casas,Elliot Chane-Sane,Enguerrand Paquin,Faruk Ahmed,Federico Baldassarre,Gabrielle Berrada,Gaëtan Ecrepont,Gauthier Guinet,Genevieve Hayes,Georgii Novikov,Giada Pistilli,Guillaume Martin,Gunjan Dhanuka,Gunshi Gupta,Han Zhou,Indraneel Mukherjee,Irene Zhang,Jaeyoung Kim,Jan Ludziejewski,Jason Rute,Joachim Studnia,John Harvill,Jonas Amar,Josselin Somerville Roberts,Julien Tauran,Karmesh Yadav,Kartik Khandelwal,Kush Jain,Laurence Aitchison,Léonard Blier,Lingxiao Zhao,Louis Martin,Lucile Saulnier,Luyu Gao,Maarten Buyl,Manan Sharma,Margaret Jennings,Marie Pellat,Mark Prins,Mathieu Poirée,Mathilde Guillaumin,Matthieu Dinot,Matthieu Futeral,Maxime Darrin,Maximilian Augustin,Mert Unsal,Mia Chiquier,Nathan Grinsztajn,Neha Gupta,Olivier Bousquet,Olivier Duchenne,Patricia Wang,Paul Jacob,Paul Wambergue,Paula Kurylowicz,Philomène Chagniot,Pierre Stock,Piotr Miłoś,Prateek Gupta,Pravesh Agrawal,Quentin Torroba,Ram Ramrakhya,Rishi Shah,Romain Sauvestre,Roman Soletskyi
### Background
该研究介绍了一种名为 Voxtral Realtime 的实时自动语音识别模型，该模型能够在低于一秒的延迟下达到离线转录的同等质量。与通过分块或滑动窗口来适应离线模型的方法不同，Voxtral Realtime 是为流式处理而训练的，并且音频流和文本流之间明确对齐。该模型基于 Delayed Streams Modeling 框架，引入了一种新的因果音频编码器和 Ada RMS-Norm，以改善延迟条件。预训练使用了跨13种语言的大规模数据集，实现在480毫秒延迟下与 Whisper (一种广泛部署的离线转录系统) 相当的性能。
### Innovation
Voxtral Realtime 的创新之处在于其针对实时流式处理的端到端训练方法，以及引入了新的因果音频编码器和 Ada RMS-Norm 来改善延迟条件。此外，该模型使用了大规模跨13种语言的预训练数据集。
### Conclusion
在480毫秒的延迟下，Voxtral Realtime 的性能与 Whisper 十分接近。该模型是在 Apache 2.0 许可证下发布的。
## 17. `cs.AI` - PBSAI治理生态：面向企业AI领地的多代理AI参考架构 [PDF](https://arxiv.org/pdf/2602.11301), [HTML](https://arxiv.org/abs/2602.11301)
### Authors
John M. Willis
### Background
企业正迅速将大规模语言模型、检索增强生成管道以及使用代理部署到生产环境中，这些系统通常运行在共享高性能计算集群和云加速平台上，这些平台还支持防御性数据分析。现有的治理和安全框架，例如NIST的人工智能风险管理框架和系统安全工程指南，虽然指出了原则和风险功能，但并未提供实施多代理、AI赋能的网络防御的可实施架构。本文背景在于探讨如何在复杂的AI系统环境下实施有效的治理。
### Innovation
本文提出了Practitioners Blueprint for Secure AI（PBSAI）治理生态，这是一种多代理参考架构，用于确保企业及超大规模AI领地的安全。PBSAI通过将责任组织到十二领域的分类法，并定义中介工具与政策之间的代理家庭，通过共享上下文包和结构化输出合同实现安全治理。该架构假定企业具备基本的安全能力，并嵌入了关键的安全技术，如分析监控、协同防御和适应性响应。此外，利用了一个轻量级形式模型来清晰界定各领域间的可追溯性、来源及人力介入保证。
### Conclusion
PBSAI被提议作为开放生态系统的结构化、证据导向的基础，并为进一步的经验验证奠定了基础。本文还展示了PBSAI与NIST人工智能风险管理框架功能的一致性，并在企业安全运营中心（SOC）及超大规模防护环境中进行了应用说明。
## 18. `cs.AI` - TRACER: 在代理推理中轨迹风险聚合对于关键事件 [PDF](https://arxiv.org/pdf/2602.11409), [HTML](https://arxiv.org/abs/2602.11409)
### Authors
Sina Tayebati,Divake Kumar,Nastaran Darabi,Davide Ettori,Ranganath Krishnan,Amit Ranjan Trivedi
### Background
在现实世界中，多轮次的人工智能代理与人类使用工具的交互中估算不确定性是非常困难的，因为失败往往由稀疏的关键事件触发（比如循环、工具使用不连贯或用户-代理协调错误），即便局部生成似乎很有信心。现有用于生成的不确定性的代理主要集中在单次文本生成上，因此错过了轨迹级别的信号中断。已有研究没有将代理和用户交互中的工具使用情景作为重点关注对象。
### Innovation
TRACER 是一个轨迹级别的不确定性度量方法，专门用于代理与用户双控制下工具使用交互。它通过结合内容感知突袭度、情境感知信号、语义和词汇的重复，以及工具依据的连贯缺口，使用尾部重点风险功能和MAX合成步骤风险来聚集这些因素，并发现关键性异常。TRACER 在$tau^2$sub-bench 上通过预测任务失败和选择性任务执行来评估。该方法相较基线提高了高达37.1%的AUROC 和55%的AUARC，使其能够更早、更准确地检测复杂对话工具使用场景中的不确定性。
### Conclusion
TRACER 通过汇聚轨迹级别的信号中断，如利用尾部重点风险功能和MAX合成步骤风险，有效地检测了关键性的代理推理中的不确定性，显著提高了不确定性检测的准确性和及时性，尤其是在复杂的交互式工具使用情景中。相关代码和基准数据已开放获取。
## 19. `cs.AI` - SemaPop: 具有语义人设条件的群体合成 [PDF](https://arxiv.org/pdf/2602.11569), [HTML](https://arxiv.org/abs/2602.11569)
### Authors
Zhenlin Qin,Yancheng Ling,Leizhen Wang,Francisco Câmara Pereira,Zhenliang Ma
### Background
群体合成是个体层次上社会经济模拟的关键组成部分，但由于需要同时代表统计结构和潜在行为语义，因此仍具有挑战性。现有的群体合成方法主要依赖于结构化属性和统计约束，没有充分捕捉调查数据中的隐含行为模式。因此，存在通过语义条件来生成群体的方法缺口。
### Innovation
本文提出了一种结合大型语言模型与生成性人口建模的语义-统计群体合成模型SemaPop。该模型从个人调查记录中提取高层次的人格表示，并将其作为人口生成的语义条件信号。与此同时，通过边际正则化确保与目标人群边缘分布的一致性。本研究使用带梯度惩罚的Wasserstein GAN基干实现了这种框架，称为SemaPop-GAN。大量实验表明，SemaPop-GAN在生成性能上有所改进，接近目标边际和联合分布的一致性，同时保证语义条件下的样本层面的可行性和多样性。消融研究表明，语义人设条件和架构设计选择为平衡边际一致性与结构现实性做出了贡献。
### Conclusion
这些结果表明，SemaPop-GAN能够通过有效融合语义-统计信息来实现可控和可解释的群体合成。SemaPop-GAN还为将个体行为语义与人口统计约束集成的生成性人口预测系统提供了有希望的模块化基础。
## 20. `cs.AI` - AgentLeak：多智能体大语言模型系统中隐私泄漏的全栈基准 [PDF](https://arxiv.org/pdf/2602.11510), [HTML](https://arxiv.org/abs/2602.11510)
### Authors
Faouzi El Yagoubi,Ranwa Al Mallah,Godwin Badu-Marfo
### Background
当前基准无法衡量多智能体大型语言模型系统中的隐私风险。当智能体在任务上协作时，敏感数据通过智能体之间的消息、共享内存和工具参数进行传递，而这些传递过程是现有的单一输出审计无法检测到的。
### Innovation
引入了AgentLeak，这是第一个涵盖内部通道的全栈隐私泄漏基准，跨越1000个场景，涵盖医疗、金融、法律和商业领域，并附带了一个32类的攻击分类和三级检测管道。测试GPT-4o、GPT-4o-mini、Claude 3.5 Sonnet、Mistral Large和Llama 3.3 70B四个模型在4,979条踪迹中的结果揭示了多智能体配置减少了每通道输出泄漏（C1：27.2% vs 单智能体中的43.2%），但增加了未监测的内部通道，从而使整个系统暴露增加到68.9%（C1与C2、C5跨域聚合）。内部通道是最主要的部分，其中智能体间的消息（C2）泄漏率高达68.8%，而输出通道（C1）仅为27.2%。这意味着单一输出审计会错过41.7%的违规情况。Claude 3.5 Sonnet通过其设计强调了安全性对齐，实现了外部（3.3%）和内部（28.1%）通道最低的泄漏率，这表明模型层面的安全训练可能有助于保护内部通道。
### Conclusion
所有五个模型和四个领域的研究结果均显示C2 > C1，确认了智能体间通信是主要的漏洞。这些发现突显了需要引入协调框架来整合内部通道的隐私保护，以及在智能体间通信上强制执行隐私控制的必要性。
## 21. `cs.AI` - MapReduce LoRA: 在生成模型多偏好优化中的帕累托前沿推进 [PDF](https://arxiv.org/pdf/2511.20629), [HTML](https://arxiv.org/abs/2511.20629)
### Authors
Chieh-Yun Chen,Zhonghao Wang,Qi Chen,Zhifan Ye,Min Shi,Yue Zhao,Yinan Zhao,Hui Qu,Wei-An Lin,Yiru Shen,Ajinkya Kale,Irfan Essa,Humphrey Shi
### Background
强化学习从人类反馈（RLHF）结合奖励模型已使生成模型对人类审美和感知偏好达到更好的对齐。然而，同时优化多个奖励往往会导致一个维度的提升而其他维度的下降，即‘对齐税’。
### Innovation
本文介绍了两个互补的方法：MapReduce LoRA 和 Reward-aware Token Embedding (RaTE)。MapReduce LoRA 在并行训练特定偏好的 LoRA 专家，并迭代合并它们以细化共享基础模型；RaTE 学习奖励特定的标记嵌入，可在推理时组合，提供灵活的偏好控制。实验证明，这些方法在文本到图像生成、文本到视频生成和语言任务等方面表现出色。
### Conclusion
我们的框架在多种模态下的多偏好对齐方面设置了一个新的最佳实践。
## 22. `cs.AI` - SUGAR: 一种去除多个身份生成遗忘的甜蜜解决方案 [PDF](https://arxiv.org/pdf/2512.06562), [HTML](https://arxiv.org/abs/2512.06562)
### Authors
Dung Thuy Nguyen,Quang Nguyen,Preston K. Robinette,Eli Jiang,Taylor T. Johnson,Kevin Leach
### Background
近期，3D 意识生成模型的发展已实现高质量的人类身份图像合成，但这也引发了用户同意以及从模型输出空间去除特定个体的迫切问题。为应对这一问题，本文引入了 SUGAR 框架，通过在不重新训练模型的情况下同时或顺序去除多个身份，实现身份的可逆删除。
### Innovation
SUGAR 框架能够为每个需要删除的身份学习一个个性化的替代潜在表示，从而导向视觉一致的替代重建，同时保持模型的质量和多样性。此外，还引入了一种持续的实用性保护目标，以防止在更多身份被遗忘时模型性能下降。经过实验验证，SUGAR 在去除多达 200 个身份方面达到了最先进的性能，相比现有基线在保留实用性方面提高了 700%。
### Conclusion
本文提出了 SUGAR 框架，能够快速有效地去除多个身份，同时保持模型的质量和多样性。FUR 表现出比现有基线更好的性能，并且具有持续的实用性保护机制。SUGAR 的代码已公开，将在促进去身份生成模型研究领域的发展方面发挥重要作用。
## 23. `cs.AI` - 揭开LLM充当裁判的面纱：一个可解析表示的推理时间扩展模型 [PDF](https://arxiv.org/pdf/2512.19905), [HTML](https://arxiv.org/abs/2512.19905)
### Authors
Indranil Halder,Cengiz Pehlevan
### Background
近年来，大型语言模型在训练时间和推理时间之间重新分配计算资源方面显示出优势。然而，推理时间扩展的基本原理尚不清晰。本文通过引入一类可解析的推理时间扩展模型，旨在深入理解这一现象。
### Innovation
本文提出了一种基于贝叶斯线性回归和奖励加权抽样的可解析的推理时间扩展模型，该模型可以优化自教师模型的训练数据在高维情况下的推理时间选择。理论分析表明，当奖励与教师模型接近时，随着推理时间样本数的增加，泛化误差会单调减少。本文通过额外的大型语言模型实验验证了这些结论，且在“最佳的k个样本”极限情况下，泛化误差的衰减可表示为Θ(1/k^2)。
### Conclusion
本文的实际数据和理论分析表明，推理时间的计算扩展在某些情况下比收集更多数据更具优势。然而，当任务难度增加时，推理时间计算的优势会逐渐减弱。
## 24. `cs.AI` - Hilbert-Guided Sparse Local Attention [PDF](https://arxiv.org/pdf/2511.05832), [HTML](https://arxiv.org/abs/2511.05832)
### Authors
Yunge Li,Lanyu Xu
### Background
全局自注意力计算和内存成本很高，限制了其在高分辨率图像中的应用。局部注意力通过限制注意力范围至局部街区来降低复杂度。虽然块稀疏核能够进一步提高局部注意力的有效性，但传统局部注意力模式经常未能显著加速处理，因为窗口内的token在1D序列中往往不是连续的。为此，作者提出了一种基于希尔伯特曲线的方法来构建窗口和邻域。
### Innovation
作者提出了一种基于希尔伯特曲线的方法，首先沿希尔伯特曲线重新排列图像token，然后基于重新排序后的1D序列形成窗口和邻域。从块稀疏角度来看，这种策略显著增加了块稀疏性，并可与现有的块稀疏核结合使用，以提高2D局部注意力的效率。实验结果显示，所提出的希尔伯特窗口注意力和滑动注意力分别加速了约4倍和18倍。
### Conclusion
作者通过希尔伯特曲线指导的局部注意力策略与块稀疏核结合的方式，提供了一种增强2D局部注意力效率的通用且实用的方法。该策略作为希尔伯特窗口变换器和希尔伯特邻域变换器实现，实现了端到端的速度提升，同时保持了微量的准确率损失。
## 25. `cs.AI` - Vision Transformers中的Block-Recurrent 动力学 [PDF](https://arxiv.org/pdf/2512.19941), [HTML](https://arxiv.org/abs/2512.19941)
### Authors
Mozes Jacobs,Thomas Fel,Richard Hakim,Alessandra Brondetta,Demba Ba,T. Andy Keller
### Background
随着Vision Transformers（ViTs）成为标准的视觉骨干网络，对其计算现象的机制解释变得至关重要。尽管ViTs的架构暗示了动态结构的存在，但目前尚无框架能够清晰地解释Transformer的深度作为特征流的形式。
### Innovation
提出了Block-Recurrent Hypothesis（BRH），即ViTs可以通过少量重用的区块运用循环结构进行准确计算。通过训练Block-Recurrent surrogates（Raptor）来验证这一假设，并发现了一系列特定于token的动力学，例如类依赖的向量收敛、token特定的动力学模式以及深度后期低秩更新等现象。实验表明，BRH能够解释ViT内部的复杂动力学过程。
### Conclusion
证明了ViT深度中存在一个紧凑的循环计划，这表明这些模型可通过原理性的动力学系统分析进行研究，呈现出低复杂度的规范性解，使得这些模型可以被简化理解和研究。
## 26. `cs.AI` - 和谐泛化与专业化：不确定性驱动的协作学习在半监督医学图像分割中的应用 [PDF](https://arxiv.org/pdf/2512.13101), [HTML](https://arxiv.org/abs/2512.13101)
### Authors
Wenjing Lu,Yi Hong,Yang Yang
### Background
视觉基础模型在医学图像分割中表现出强大的泛化能力，得益于大规模异质预训练。然而，它们在有限注释或稀有病理变异的情况下难以泛化到特定的临床任务中，因为通用先验与任务特定需求之间存在不匹配。
### Innovation
提出了一种双重教师框架Uncertainty-informed Collaborative Learning（UnCoL），该框架旨在在半监督医学图像分割中平衡泛化和专业化。具体来说，UnCoL通过冻结的基础模型提炼视觉和语义表示以转移通用知识，同时保持逐步适应的教师以捕捉细微和任务特定的表示。UnCoL中的拟 labels 学习通过预测不确定性适当地调节，从而有选择地抑制不可靠的监督并在含糊区域稳定学习。
### Conclusion
在多种2D和3D分割基准测试上，UnCoL持续优于最先进的半监督方法和基础模型基准，并且在明显减少标注需求的情况下接近全监督性能。
## 27. `cs.AI` - H-LDM：从临床元数据生成可控和可解释的心音图合成的层次潜在扩散模型 [PDF](https://arxiv.org/pdf/2511.14312), [HTML](https://arxiv.org/abs/2511.14312)
### Authors
Chenyang Xu,Siming Li,Hao Wang
### Background
心音图（PCG）分析对于心血管疾病的诊断至关重要，但由于缺乏标记的病理数据，阻碍了AI系统的性能。为了解决这个问题，我们提出了H-LDM，一种用于生成具备临床准确性和可控性的PCG信号的层次潜在扩散模型。
### Innovation
我们的方法包括：（1）一种多尺度VAE，学习生理分离的潜在空间，区分节律、心音和杂音；（2）一种层次化的文本至生物信号管道，利用丰富的临床元数据实现对17种不同条件的精细控制；（3）一种通过新颖的医学注意力模块引导的可解释扩散过程。在PhysioNet CirCor数据集上的实验表明，该模型具有最先进的性能，实现了9.7的Fréchet音频距离、92%的属性分离得分，并且有87.1%的临床有效验证，得到心脏病专家的确认。我们的合成数据增强了诊断模型对罕见疾病的分类准确性，提高了11.3%。
### Conclusion
H-LDM为心脏诊断中的数据增强提供了一个新的方向，通过可解释的临床洞察，弥补了数据稀缺性。
## 28. `cs.AI` - 在粤语、日语和土耳其语上的现代大型语言模型评估：跨语言基准 [PDF](https://arxiv.org/pdf/2511.10664), [HTML](https://arxiv.org/abs/2511.10664)
### Authors
Chengxuan Xia,Qianye Wu,Hongbin Guan,Sixuan Tian,Yilun Hao,Xiaoyu Wu
### Background
大型语言模型（LLMs）在资源丰富的语言（如英语）上已经取得了显著成果，但在低资源和形态丰富的语言上的效果尚未充分研究。
### Innovation
本文对七款前沿LLMs（包括GPT-4o、GPT-4、Claude 3.5 Sonnet、LLaMA 3.1、Mistral Large 2、LLaMA-2 Chat 13B、Mistral 7B Instruct）进行了全面评估。采用了涵盖粤语、日语和土耳其语的新跨语言基准，并结合人类评估和自动化指标评估模型性能。
### Conclusion
虽然最大的专有模型在语言和任务上往往表现出领先地位，但对文化细微差别的理解和形态通用性的差距仍然存在。GPT-4o展现了多语言性能，并且Claude 3.5 Sonnet在知识和推理基准上达到了有竞争力的准确度。然而，所有模型在每种语言的独特语言挑战上都表现不佳。开源小型模型在流畅性和准确性上差距明显，表明了资源差距。提供了详细的定量结果、定性错误分析，讨论了开发更加文化意识强和语言通用化的LLMs的含义。同时也发布了基准和评估数据以促进可重复性研究和进一步研究。
## 29. `cs.AI` - 自适应图混合模型 [PDF](https://arxiv.org/pdf/2511.13062), [HTML](https://arxiv.org/abs/2511.13062)
### Authors
Mohit Meena,Yash Punjabi,Abhishek A,Vishal Sharma,Mahesh Chandran
### Background
图形神经网络（GNNs）因其强大的图结构数据学习能力而受到关注，但最近的研究表明，其性能提升已经开始出现瓶颈。在许多情况下，如GCN和GAT这样的成熟模型，在恰当调整下可以匹配甚至超越更复杂、更先进的架构。这表明目前存在一个关键限制：选择最适合特定图任务或数据集的模型的难度。
### Innovation
提出了自适应图混合模型（SAGMM），这是一种模块化且实用的框架，能够自动选择和组合来自多种架构的最合适的GNN模型。与依赖单一基本模型变体的先验混合专家方法不同，SAGMM利用架构多样性及拓扑感知注意力门控机制，根据输入图结构适应性地分配专家到每个节点。为了提高效率，SAGMM包括一个剪枝机制，在训练和推理过程中减少活跃专家的数量而不影响性能。此外，还探索了一种训练高效的变体，在这种变体中，专家模型被预训练并冻结，仅训练门控和特定任务层。
### Conclusion
SAGMM在覆盖节点分类、图分类、回归和链接预测任务的16个基准数据集上进行评估，并表明它能在性能上持续超越或匹配当前最先进的GNN基准和先验混合方法，提供了一种稳健且适应性强的图学习解决方案。
## 30. `cs.AI` - EGG-SR: 将符号等价嵌入到基于等价图的符号回归中 [PDF](https://arxiv.org/pdf/2511.05849), [HTML](https://arxiv.org/abs/2511.05849)
### Authors
Nan Jiang,Ziyi Wang,Yexiang Xue
### Background
符号回归的目标是从实验数据中发现物理定律，通过寻找闭形式表达式。然而，表达式的搜索空间呈指数增长，使得计算上具有挑战性。一种具有潜力但尚未充分探索的方向是通过挖掘符号等价性来缩小搜索空间和加速训练。由于许多看似不同的表达式实际上定义了相同的函数，现有的算法处理这些变体时会将其视为不同的结果，因此导致了冗余探索和学习缓慢的问题。
### Innovation
我们提出了EGG-SR框架，它将符号等价性整合到一种现代符号回归方法中，包括蒙特卡洛树搜索（MCTS）、深度强化学习（DRL）和大型语言模型（LLMs）。通过提出的EGG模块（利用等价图），EGG-SR可以紧凑地表示等价的表达式。该框架通过以下方式加速学习：(1) 在EGG-MCTS中修剪冗余的子树探索；(2) 在EGG-DRL中汇聚等价生成序列的奖励；(3) 在EGG-LLM中丰富反馈提示。理论上，我们展示了嵌入EGG的学习利益：它收紧了MCTS的遗憾边界并降低了DRL梯度估计量的方差。实验上，EGG-SR在几个基准测试中增强了多种符号回归模型，能够在相同的时间限制内发现更准确的表达式。
### Conclusion
EGG-SR框架在多个符号回归基准测试中增强了多种模型，并且能够在相同时间内发现更准确的表达式。从理论和实验上都展示了将符号等价性嵌入符号回归中的重要性，这可以显著提高学习效率和解决问题的能力。
## 31. `cs.LG` - 世界模型规划中重要位的位置：高效空间推理的配对混合位研究 [PDF](https://arxiv.org/pdf/2602.11882), [HTML](https://arxiv.org/abs/2602.11882)
### Authors
Suraj Ranganath,Anish Patnaik,Vaishak Menon
### Background
高效的空间推理需要可靠的全域模型，尤其是在严格的精度预算下。本文研究了低比特规划行为主要由总位宽还是位分配到不同模块所决定的问题。通过在DINO-WM上对Wall规划任务进行实验，我们发现精确度在特定位宽区间内的分配有显著影响。
### Innovation
本文采用配对目标混位评估方法，在统一、混位、不对称及分层变体下，分别在两种规划预算下进行实验，观察到在8位和6位设置下结果接近FP16，3位设置崩溃，4位设置敏感于位分配。过渡区域中，保持编码器精度比均匀量化更好，而且接近大小的不对称变体有相同的趋势。最后，这些发现表明了高效空间推理中模块感知和预算感知的量化策略的重要性。
### Conclusion
本文的研究结果表明，在效率空间推理中，位分配在特定区间内对结果有显著影响，因此提出模块感知、预算感知的量化策略作为未来研究的方向。
## 32. `cs.LG` - 基于扩散的全局概率降尺度 [PDF](https://arxiv.org/pdf/2602.11893), [HTML](https://arxiv.org/abs/2602.11893)
### Authors
Roberto Molinaro,Niall Siegenheim,Henry Martin,Mark Frey,Niels Poulsen,Philipp Seitz,Marvin Vincent Gabler
### Background
该研究介绍了一种通用的基于扩散的降尺度框架，该框架可以将确定性的低分辨率天气预报转换为概率性的高分辨率预测，而无需对特定模型进行微调。这种方法利用一种条件扩散模型，在大规模输入（约25公里分辨率）和高分辨率区域再分析目标数据（约5公里分辨率）上进行训练，并以完全无监督的方式应用于来自不同上游天气模型的确定性预报。这种方法特别关注近地表变量，在长达90小时的预报期中，对比独立的现场观测数据，进行评估。
### Innovation
该方法通过单一的条件扩散模型，能够将低分辨率的天气预报转换为高分辨率的概率性预测，且无需对特定模型进行调整，展示了一种可扩展且模型无关的降尺度方法，增强了天气预报中的空间分辨率和不确定性表示。
### Conclusion
扩散基降尺度框架提供了一种可扩展、模型无关的概率性接口，用于提高运营天气预报管道中空间分辨率不确定性的表示，且通过集合平均，降尺度后的预报相较于各自模型本有的确定性预报有持续的提高，尤其是在CRPS概率技能方面表现出显著的提升。
## 33. `cs.LG` - 时间序列预测中的时间统一对抗扰动 [PDF](https://arxiv.org/pdf/2602.11940), [HTML](https://arxiv.org/abs/2602.11940)
### Authors
Ruixian Su,Yukun Bao,Xinze Zhang
### Background
尽管深度学习模型在时间序列预测方面取得了显著的成功，但它们对对抗样本的脆弱性仍然是一个关键的网络安全问题。然而，时间序列数据固有的时间一致性在现有预测领域的攻击方法中经常被忽视，导致同一个时间戳在重叠样本中的扰动值存在分歧和矛盾。这种时间不一致性使得对抗攻击在实际数据操纵中变得不切实际。
### Innovation
为了应对这个问题，我们提出了时间统一对抗扰动（TUAP）方法，通过引入时间统一约束确保每个时间戳在所有重叠样本中的扰动一致。此外，我们提出了一种新的时间戳梯度累加方法（TGAM），提供了一种模块化且高效的策略来生成TUAP，通过聚合重叠样本的局部梯度信息。通过将TGAM与基于动量的攻击算法结合，我们确保了严格的时间一致性，同时充分利用了序列级的梯度信息来探索对抗扰动空间。
### Conclusion
我们在三个基准数据集和四个代表性最先进的模型上进行了全面的实验，结果表明，在满足TUAP约束的情况下，我们的方法在白盒和黑盒转移攻击场景下均显著优于基线方法。此外，我们的方法在无需TUAP约束的情况下也表现出更优异的转移攻击性能，证明了其生成时间序列预测模型对抗扰动的有效性和优越性。
## 34. `cs.LG` - 在异构场景中利用历史信息增强性能的模型对比联邦学习 [PDF](https://arxiv.org/pdf/2602.11945), [HTML](https://arxiv.org/abs/2602.11945)
### Authors
Hongliang Zhang,Jiguo Yu,Guijuan Wang,Wenshuo Ma,Tianqing He,Baobao Chai,Chunqiang Hu
### Background
联邦学习（FL）允许多个节点不共享原始数据的情况下协作训练模型。然而，在异构场景中部署FL系统时，节点之间在数据分布和参与频率上存在差异，这会损害FL系统的性能。
### Innovation
本文提出了一种名为PMFL的增强性能的模型对比联邦学习框架，该框架利用历史训练信息。PMFL在节点侧通过引入基于历史局部模型的新型模型对比项，以及在服务器侧利用节点累积参与度进行自适应权值调整，并结合历史全局模型来减少性能波动。
### Conclusion
全面的实验表明，PMFL方法在异构场景中相比现有联邦学习方法具有优越的性能。
## 35. `cs.LG` - A²V-SLP: 对齐感知的变量化模型用于解纠缠的手语生成 [PDF](https://arxiv.org/pdf/2602.11861), [HTML](https://arxiv.org/abs/2602.11861)
### Authors
Sümeyye Meryem Taşyürek,Enis Mücahid İskender,Hacer Yalim Keles
### Background
基于近年来为了手语生成提出的结构解纠缠框架，本文提出了一种对齐感知变量化框架A²V-SLP，该框架学习解纠缠的音素级潜在分布，而不是确定性嵌入。
### Innovation
该框架采用一种解纠缠的变分自编码器（VAE）将真实手语姿态序列编码并提取音素特定的均值和方差向量，用于训练非自回归Transformer。通过这种方式，可以避免确定性潜在嵌入的坍塌，从而保持音素级别的表示。此外，还集成了一个词汇注意力机制，以加强语言输入与动作之间的对齐。
### Conclusion
实验结果显示，该方法在确定性潜在回归方面具有明显的改进，并且在无绘词设置下的性能达到了最先进的后翻译精度，在动作真实感方面也有所提升。
## 36. `cs.LG` - 在AI法案中利用预测多模态性衡量个体性能 [PDF](https://arxiv.org/pdf/2602.11944), [HTML](https://arxiv.org/abs/2602.11944)
### Authors
Karolin Frohnapfel,Mara Seyfert,Sebastian Bordt,Ulrike von Luxburg,Kristof Meding
### Background
在开发决策支持的人工智能系统时，经常遇到预测多模态性现象，即不存在单一最佳模型，多个具有相似准确性的模型在具体个案上可能给出不同的预测结果。特别是在决策直接影响人类的情况下，这可能会导致极大的不满意度。由于高风险AI系统需要报告特定个体级别的性能，预测多模态性与AI法案的准确性要求相冲突，因此需要找到方法来评估和报告这一现象。
### Innovation
本文通过提出利用预测多模态性信息来符合AI法案的准确性要求，建议使用个体冲突比和δ-模糊性作为工具来量化模型在个体案例上的分歧，并据此提出了具体评估和实施方法，最终建议在AI法案下提供关于预测多模态性的信息，使部署者能够判断系统特定个体输出的可靠性。
### Conclusion
在AI法案框架下，可以通过提供关于预测多模态性的信息，有助于判断特定个体输出的可靠性。这包括使用个体冲突比和δ-模糊性来衡量模型在个体案例上的分歧，以及为模型提供者提供易于实施的评估方法。
## 37. `cs.LG` - 学习条件平均值 [PDF](https://arxiv.org/pdf/2602.11920), [HTML](https://arxiv.org/abs/2602.11920)
### Authors
Marco Bressan,Nataly Brukhim,Nicolo Cesa-Bianchi,Emmanuel Esposito,Yishay Mansour,Shay Moran,Maximilian Thiessen
### Background
该论文在PAC框架下探讨了条件平均值的学习问题。例如，在标准PAC学习中，学习者会收到一个未知目标概念（假设）的标记样本，并在已知概念类中进行学习。但是，目标并不是去学习目标概念本身，而是为每一个实例预测其邻域内的平均标签（邻域是包含该实例的任意子集），邻域的定义比较灵活，可以是任意的集合。当所有邻域都退化成单点集时，该问题与经典的PAC学习完全一致。总之，该研究将PAC学习扩展到了可以涵盖解释性、公平性和推荐系统等领域的学习任务。
### Innovation
文章的主要贡献是完全描述了哪些条件下条件平均值是可学习的，以及样本复杂度的紧边界（考虑到对数因素）。该描述的关键在于两个新的组合参数的联合有限性，这两个参数分别依赖于概念类和邻域系统，且与关联邻域图的独立数密切相关。
### Conclusion
该研究为条件平均值的学习提供了一个完整的参数化描述，这对理解如何扩大PAC学习框架的应用范围具有重要意义，并对特定任务的设计和分析提供了指导。
## 38. `cs.LG` - 将Puzzle扩展应用于专家混合推理模型，并应用于GPT-OSS加速 [PDF](https://arxiv.org/pdf/2602.11937), [HTML](https://arxiv.org/abs/2602.11937)
### Authors
Akhiad Bercovich,Nir Ailon,Vladimir Anisimov,Tomer Asida,Nave Assaf,Mohammad Dabbah,Ido Galil,Amnon Geifman,Yonatan Geifman,Izhak Golan,Roi Koren,Itay Levy,Zach Moshe,Pavlo Molchanov,Najeeb Nabwani,Mostofa Patwari,Omri Puny,Tomer Ronen,Itamar Schen,Elad Segal,Ido Shahaf,Oren Tropp,Ran Zilberstein,Ran El-Yaniv
### Background
大语言模型（LLMs）通过生成更长的推理步骤来提高答案质量，但这样会显著增加服务成本。因此，研究者们开始寻找优化推理过程的方法。
### Innovation
研究者通过扩展和应用Puzzle框架（一种后训练神经架构搜索框架），在GPT-OSS-120B模型上创建了GPT-OSS-Puzzle-88B，该模型在保持低生成长度的同时，通过混合专家池化、部分上下文注意力替换为窗注意力、FP8 KV缓存量化以及后训练强化学习等方法，实现了部署优化。该方法在长上下文和短上下文设置中分别达到了1.63倍和1.22倍的吞吐量加速，在单个NVIDIA H100 GPU上实现了2.82倍的吞吐量加速。
### Conclusion
GPT-OSS-Puzzle-88B在整个推理努力过程中提高了请求级别的效率，最高可达1.29倍，并且在各种基准测试中表现与或略优于其父模型，在推理努力过程中保持了100.8%至108.2%的保留率。这表明后训练架构搜索可以在不牺牲质量的情况下显著降低推理成本。
## 39. `cs.LG` - 在大型语言模型中的上下文函数学习 [PDF](https://arxiv.org/pdf/2602.11863), [HTML](https://arxiv.org/abs/2602.11863)
### Authors
Elif Akata,Konstantinos Voudouris,Vincent Fortuin,Eric Schulz
### Background
大型语言模型（LLMs）可以在推理时从少量演示中学习。本文通过高斯过程（GPs）的视角研究这种基于上下文的学习现象。构建了对照实验，其中模型观察来自已知GP先验的多元标量函数样本序列，评估预测误差，并与两个原理上的参考进行比较：（i）一个经验性的GP回归学习者提供了可实现误差的下界，（ii）1-最近邻规则提供了一个数据驱动的上界。
### Innovation
研究发现LLM的预测误差与演示数量有关，并接近GP下界。使用基于似然性的分析研究了这些模型的归纳偏见，发现LLM预测在不太光滑的GP核下最有可能。进一步研究表明，强化学习和监督微调可以有效地调整归纳偏见，以更好地适应具有更光滑核的GP产生的函数。研究框架量化了LLM行为类似于GP学习者的程度，并提供了引导其归纳偏见的工具，以提高连续函数学习任务的样本效率。
### Conclusion
研究发现LLM的学习曲线受到函数生成核的影响，且随着演示次数的增加接近GP下界。通过实验表明，LLM的归纳偏见主要受核光滑性影响。通过强化学习和监督微调，可以有效纠正这些偏见，提高对更光滑GP函数的样本效率。总体而言，框架不仅量化了LLM的GP学习性质，还提供了调整归纳偏见的方法，以优化连续函数学习任务的表现。
## 40. `cs.LG` - 缓解基于参考的偏好优化中的偏差 [PDF](https://arxiv.org/pdf/2602.11902), [HTML](https://arxiv.org/abs/2602.11902)
### Authors
Suqin Yuan,Xingrui Yu,Jiyang Zheng,Lei Feng,Dadong Wang,Ivor Tsang,Tongliang Liu
### Background
直接偏好优化（DPO）已成为大型语言模型中离线偏好对齐的事实标准，但它依赖于参考策略引入了关键矛盾。DPO 通过参考来衡量每次更新，这在可信区域内稳定了训练，但当参考策略偏好被拒绝的响应时，这种依赖性会变成问题。对于这类悲观对，DPO 会在策略边际（$triangle_theta$）仅仅击败参考边际（$triangle_{text{ref}}$）时过早减弱梯度，即使策略仍然错误（$triangle_theta < 0$）。这被称为提前满足，是训练与推断不匹配的一种具体表现。
### Innovation
我们通过引入Hybrid-DPO（HyPO），提出了一种DPO的简便修改，可以使DPO条件性地依赖于参考。HyPO仅在参考策略乐观或中立时才像DPO一样行为，当参考策略悲观时，通过将$triangle_theta - triangle_{text{ref}}$替换为$triangle_theta - text{max}times(0, triangle_{text{ref}})$来将参考视为中立。这一小改动在悲观对上加强了每个实例的学习信号，同时保持了DPO的目标形式和计算成本。通过条件消除悲观参考信号的偏差，HyPO缓解了提前满足的情况；实验结果显示，HyPO提高了推理一致性的度量标准，并实现了更高的两两胜率。
### Conclusion
我们的结果表明，直接偏好对齐可以通过条件性地消除了参考信号的偏差来得以增强，而不是完全丢弃这一信号。
