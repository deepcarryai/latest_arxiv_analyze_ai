# 20260315
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 通过语义路由基于LoRA的可逆终身模型编辑 [PDF](https://arxiv.org/pdf/2603.11239), [HTML](https://arxiv.org/abs/2603.11239)
### Authors
Haihua Luo,Xuming Ran,Tommi Kärkkäinen,Zhonghua Chen,Jiangrong Shen,Qi Xu,Fengyu Cong
### Background
现实世界的动态演化要求在大型语言模型中进行模型编辑。现有方法探索模块化隔离或参数高效策略，但这些方法仍会因为连续更新而遭受语义漂移或知识遗忘的问题。
### Innovation
本文提出了一种基于语义路由的LoRA框架——SoLA，以解决上述挑战。SoLA中的每个编辑被封装为独立的LoRA模块，该模块在训练后被冻结并通过语义路由映射到输入，允许通过语义匹配动态激活LoRA模块。此外，SoLA可以通过从语义路由中移除关键信息来撤销特定编辑，恢复模型的原始行为。这是现有文献中首次实现的可逆回滚编辑能力。SoLA将决策过程集成到编辑层中，消除了辅助路由网络的需求，实现了端到端的决策过程。
### Conclusion
广泛的实验表明，SoLA能够有效地学习和保留编辑知识，实现准确、高效和可逆的终身模型编辑。
## 2. `cs.AI` - 制衡与互补：人工智能与区块链的融合推动去中心化未来 [PDF](https://arxiv.org/pdf/2603.11299), [HTML](https://arxiv.org/abs/2603.11299)
### Authors
Yibai Li,Zhiye Jin,Xiaobing(Emily)Li,K. D. Joshi,Xuefei(Nancy)Deng
### Background
这篇编辑文章探讨了人工智能（AI）和区块链技术之间的关键交叉点，指出了它们在集中化和分散化方面不同的趋势。AI，特别是在大规模语言模型（LLMs）兴起的情况下，由于大型企业对数据和资源的垄断表现出强烈的集中化趋势。相比之下，区块链通过其固有的分散性、透明性和安全性提供了相反的力量。文章指出这些技术并非彼此排斥，而是具有互补的优势。
### Innovation
文章提出了一种新的理念——去中心化智能（DI），这是一个跨学科的研究领域，旨在开发无需中央控制即可运行的智能系统。区块链可以通过实现分散的数据管理、计算和治理来缓解AI的集中化风险，促进更大的包容性、透明性和用户隐私。同时，AI可以通过自动化智能合约管理、内容筛选和威胁检测来提升区块链的效率和安全性。
### Conclusion
核心观点是推动去中心化智能的发展，将其视为AI与区块链技术融合的结果，并共同为实现去中心化的未来做出贡献。这种融合有望促进社会的包容性、透明性和安全性。
## 3. `cs.AI` - 关注用户模拟中的Sim2Real差距 [PDF](https://arxiv.org/pdf/2603.11245), [HTML](https://arxiv.org/abs/2603.11245)
### Authors
Xuhui Zhou,Weiwei Sun,Qianou Ma,Yiqing Xie,Jiarui Liu,Weihua Du,Sean Welleck,Yiming Yang,Graham Neubig,Sherry Tongshuang Wu,Maarten Sap
### Background
随着自然语言处理（NLP）评估从静态基准转向多轮交互设置，基于大型语言模型（LLM）的模拟器已经成为用户代理的常用工具，承担生成用户回合和提供评价信号的双重角色。然而，这些模拟通常假定与实际人类行为一致，很少进行严格的验证。
### Innovation
该研究首次采用完整的$tau$-bench协议进行了大规模的人类实验（451名参与者，165项任务），对比了31种LLM模拟器的表现，并引入了一个新指标——用户-模拟器指数（USI）——来量化LLM模拟器与真实用户互动行为和反馈的相似程度。
### Conclusion
LLM模拟器的行为过于合作，风格一致性高，并且缺乏真实性和模糊性，导致“容易模式”，这会夸大代理的成功率，超出人类基线。真实的人类用户在多个质量维度上提供了细致的评估，而模拟用户提供了更多正面反馈。基于规则的奖励无法捕捉到由真实用户生成的丰富反馈信号。总体而言，更高的通用模型能力并不总是产生更忠实的用户模拟。因此，使用基于LLM的用户模拟器进行代理开发周期中应进行人类验证，以改进用户模拟模型。
## 4. `cs.AI` - PACED: 在学生技能前沿进行蒸馏 [PDF](https://arxiv.org/pdf/2603.11178), [HTML](https://arxiv.org/abs/2603.11178)
### Authors
Yuanda Xu,Hejian Sang,Zhengze Zhou,Ran He,Zhipeng Wang
### Background
标准大型语言模型（LLM）蒸馏浪费了大量计算资源：一方面，模型已经掌握的问题导致梯度几乎为零；另一方面，超出了模型能力范围的问题导致梯度混乱并侵蚀现有能力。这些浪费并非仅仅是直观上的，而是结构上不可避免的，因为蒸馏梯度的信号噪音比在两个通过率极端情况下会消失。
### Innovation
提出了Paced框架，通过一个基于结构梯度的得分函数 $w(p) = p^beta(1 - p)^beta$，将蒸馏的重点放在学生模型的近似学习区，即其能力的前沿。该框架证明了贝塔核 $w(p) = p^beta(1 - p)^beta$ 是由于蒸馏梯度信号噪音结构产生的主导重的权重家族，并且具有最小最大化鲁棒性。在蒸馏、自蒸馏和两阶段蒸馏实验中，此方法都能显示出显著的性能提升。
### Conclusion
所有配置仅需要学生模型的展开来估计通过率，无需进行架构更改，并且兼容任何KL（柯列利散度）方向，实验结果显示在标准的推理基准测试中取得了显著的改进，支持了一种覆盖模式然后巩固的蒸馏过程解释。
## 5. `cs.AI` - 评价大模型去学习的幻象：一种动态框架 [PDF](https://arxiv.org/pdf/2603.11266), [HTML](https://arxiv.org/abs/2603.11266)
### Authors
Raj Sanjay Shah,Jing Huang,Keerthiram Murugesan,Nathalie Baracaldo,Diyi Yang
### Background
大语言模型（LLMs）的目标是提高安全性、减轻偏见并遵守法律要求，例如被遗忘权利。然而，现有的去学习方法存在脆弱性：轻微的查询修改，如多跳推理和实体别名，可以恢复被认为被遗忘的信息。因此，现有的评估标准往往造成有效性假象，无法检测这些漏洞。该研究提出了一种动态框架，用于使用复杂结构化查询测试去学习的鲁棒性。研究表明，框架可以自动生成语义等价的查询，并与之前的评估结果相匹配，同时揭示了其他标准未能发现的新去学习失败，尤其在多跳设置中。
### Innovation
该研究提出了一种动态框架来挑战大语言模型的去学习效果。通过自动生成语义等价的查询和构建从简单查询到多跳链的探针，框架能够精确控制查询难度。此外，研究还通过激活分析表明单跳查询通常遵循主要计算路径，而多跳查询则使用其他路径，这解释了在多跳设置中去学习技术的脆弱性。该框架无需手动构建遗忘测试集，使得在实际应用中更容易应用。
### Conclusion
该框架提供了一种实用和可扩展的方法来评估去学习方法，能够揭示先前标准未能发现的新去学习失败。相比之前的静态非结构化基准，该框架能够更好地识别去学习的漏洞。研究结果还表明，单纯依靠被遗忘权利或其他静态基准可能不足以全面评估去学习方法的有效性。
## 6. `cs.AI` - DIVE: 在通用工具使用中扩展代理任务合成中的多样性 [PDF](https://arxiv.org/pdf/2603.11076), [HTML](https://arxiv.org/abs/2603.11076)
### Authors
Aili Chen,Chi Zhang,Junteng Liu,Jiangjie Chen,Chengyu Du,Yunji Li,Ming Zhong,Qin Wang,Zhengmao Zhu,Jiayuan Song,Ke Ji,Junxian He,Pengyu Zhao,Yanghua Xiao
### Background
最近的工作通过为训练后工具使用的大型语言模型（LLM）合成代理任务做了许多努力，但任务和工具集发生变化时任务的稳健泛化仍然是一个开放性挑战。这种脆弱性主要归因于合成任务中的多样性不足。由于训练需要任务保持可执行和可验证，而泛化则需要覆盖各种类型的工具、工具集组合以及异构的使用工具模式，因此增加多样性非常困难。
### Innovation
我们提出了DIVE，一种基于证据的配方，通过首先执行多元化的现实工具并反推出严格由这些轨迹推导出的任务，从而为注册提供构造方式。DIVE在两个可控轴上扩展了结构多样性，即工具池覆盖范围和每任务工具集多样性，并通过证据收集-任务推导循环进一步促进了在五个领域共计373种工具上跨步骤工具使用模式的丰富化。使用DIVE数据对抗复核三次、强化学习三次的数据训练Qwen3-8B，使其在9个OOD基准测试中的平均得分提高了22分，并超越了最强的8B基线8分。
### Conclusion
研究发现，即使数据少四倍，多样性的可控扩展在OOD泛化方面也始终优于数量的扩展。DIVE方法通过提供基于证据的任务推导和结构多样性扩展，有效改善了代理任务合成的工具使用模式，显著提升了模型的泛化能力。
## 7. `cs.AI` - 自动驾驶系统中推理的综述：开放挑战与新兴范式 [PDF](https://arxiv.org/pdf/2603.11093), [HTML](https://arxiv.org/abs/2603.11093)
### Authors
Kejin Yu,Yuhan Sun,Taiqiang Wu,Ruixu Zhang,Zhiqiang Lin,Yuxin Meng,Junjie Wang,Yujiu Yang
### Background
高级自动驾驶（AD）的发展正从感知为中心的限制转向更基本的瓶颈，即在鲁棒性和通用性推理方面的不足。当前的AD系统在管理结构化环境方面表现出色，但在长尾场景和复杂的社交互动中依然表现不佳，这些场景需要人类级的判断力。同时，大型语言模型和多模态模型（LLMs和MLLMs）的出现为将强大认知引擎集成到AD系统中提供了机遇，使其超越简单的模式匹配，转向真正的理解。然而，缺乏系统化的框架来指导这种集成。因此，本文提供了一篇关于这一新兴领域的全面综述，并认为推理应被提升为系统的认知核心。
### Innovation
本文提出了一个认知层级体系来分解monolithic驱乘任务，并归纳和系统化了七大核心推理挑战，如响应-推理权衡和社交博弈推理。本文还从两个不同的视角审查了最先进的方法，一个是从系统中心的角度构建智能代理，另一个是从评估中心的角度进行验证。研究结果显示，正在出现一个全面且可解释的“玻璃盒”代理的趋势。并且指出了LLMs推理的高延迟、反思性特征与车辆控制的毫秒尺度、安全性关键需求之间的根本性紧张。未来工作的主要目标是通过开发可验证的神经符号架构、鲁棒的不确定推理以及包容的社会协商的可扩展模型来弥合符号到物理的差距。
### Conclusion
在未来的工作中，主要目标是弥合符号到物理的差距，包括开发可验证的神经符号架构，鲁棒的不确定推理能力，以及包容的社会协商的可扩展模型。
## 8. `cs.AI` - COMPASS: 可解释的主权、可持续性、合规性及伦理自主框架 [PDF](https://arxiv.org/pdf/2603.11277), [HTML](https://arxiv.org/abs/2603.11277)
### Authors
Jean-Sébastien,Dessureault,Alain-Thierry,Iliho Manzi,Soukaina,Alaoui Ismaili,Khadim, Lo,Mireille,Lalancette,Éric,Bélanger
### Background
大规模语言模型（LLM）驱动的自主系统快速增长，引发关于数字主权、环境可持续性、监管合规性和伦理对齐的关键担忧。现有框架尽管可以在某个方面提供解决，但尚未有一个统一的架构能够系统地将这些需求整合进自主系统的决策过程中。论文提出了一种名为COMPASS（Compliance and Orchestration for Multi-dimensional Principles in Autonomous Systems with Sovereignty）的框架，旨在通过模块化和扩展的治理机制强制执行价值对齐的人工智能。
### Innovation
引入了名为COMPASS的新颖多智能体编排系统，该系统通过检索增强生成（RAG）增强的专门子智能体来实现多维度原则的自主系统，在监管、碳意识计算、合规性和伦理等四个领域进行治理。通过运用LLM作为法官的方法，该系统能够实时对冲突目标进行仲裁，并提供可解释的理由。研究成果表明，RAG集成显著提高了语义一致性，并减轻了幻觉风险。
### Conclusion
通过自动评估验证了COMPASS架构的有效性，研究成果显示框架的设计使得其能够无缝集成到多种应用领域，同时保持可解释性和追踪性。
## 9. `cs.AI` - AI Psychometrics: 使用心理量表评估大型语言模型的心理推理 [PDF](https://arxiv.org/pdf/2603.11279), [HTML](https://arxiv.org/abs/2603.11279)
### Authors
Yibai Li,Xiaolin Lin,Zhenghui Sha,Zhiye Jin,Xiaobing Li
### Background
大量的参数和深度神经网络使得大语言模型（LLMs）的复杂性与人类大脑相当，但这也使得它们成为难以评估和解释的“黑箱”系统。AI心理学量表是通过应用心理学量表方法学来评估和解释人工智能（AI）系统的精神特质和过程的一门新兴领域。本研究采用技术接受模型（TAM），对四个主流的LLMs：GPT-3.5、GPT-4、LLaMA-2和LLaMA-3的心理推理和总体心理量表有效性进行了评估。
### Innovation
本研究首次使用AI心理学量表评估了大型语言模型的心理推理能力及有效性，通过技术接受模型（TAM）考察了这些模型的容合效度、区分效度、预测效度和外部效度。研究发现，所有模型的心理学测量结果都满足了所有的有效性标准，并且性能更优的GPT-4和LLaMA-3模型在心理量表有效性方面优于其前身GPT-3.5和LLaMA-2。
### Conclusion
本研究结果表明，可以使用AI心理学量表对大型语言模型进行有效性评估和解释。
## 10. `cs.AI` - 评估AI代理在多步骤网络攻击场景中的进展 [PDF](https://arxiv.org/pdf/2603.11214), [HTML](https://arxiv.org/abs/2603.11214)
### Authors
Linus Folkerts,Will Payne,Simon Inman,Philippos Giavridis,Joe Skinner,Sam Deverett,James Aung,Ekin Zorer,Michael Schmatz,Mahmoud Ghanem,John Wilkinson,Alan Steer,Vy Hong,Jessica Wang
### Background
该研究评估了前沿AI模型在两个定制的网络攻击范围（一个包含32步的公司网络攻击和一个包含7步的工业控制系统攻击）中的自主攻击能力。这些攻击需要跨多个行动序列串联使用异构能力。
### Innovation
研究通过比较不同时间点（2024年8月至2026年2月）发布的七种模型，在固定计算预算下的表现趋势。发现：1.推理计算时间与模型性能呈对数线性关系，增加从1000万到1亿个标记至多可提升59%的表现；2.每一代模型在固定标记预算下的表现优于前一代，特别是在公司网络攻击范围中，使用1000万个标记时，完成的平均步骤从最初版本的1.7上升至最后版本的9.8。
### Conclusion
在公司网络攻击场景中，最佳单次运行完成了32步中的22步，相当于一个资深专家花费约14小时完成任务所需的时间的60%。而在工业控制系统攻击场景中，虽然最近的模型首次可靠地完成了攻击步骤，但平均仅完成了每个7步中的1.2到1.4步。
## 11. `cs.CV` - UCAN: 统一卷积注意力网络在轻量级超分辨率中的扩展感受野 [PDF](https://arxiv.org/pdf/2603.11680), [HTML](https://arxiv.org/abs/2603.11680)
### Authors
Cao Thien Tan,Phan Thi Thu Trang,Do Nghiem Duc,Ho Ngoc Anh,Hanyang Zhuang,Nguyen Duc Dung
### Background
混合CNN-变压器架构在图像超分辨率中取得了良好的效果，但扩大注意窗口或卷积核的大小会显著增加计算成本，限制了在资源受限设备上的应用。
### Innovation
UCAN 提出了一个轻量级网络，通过联合卷积和注意力机制扩展有效感受野。引入了基于 Hedgehog 机制的空间注意以及一个基于知识蒸馏的大尺度内核模块来保留高频结构而不增加大量计算。此外，通过跨层参数共享进一步减少复杂度。
### Conclusion
UCAN 在 Manga109($4times$) 上达到了 31.63 dB PSNR，仅需 48.4G MACs，优于最近的轻量级模型；在 BSDS100 上也超越了更大型模型的方法，广泛实验表明 UCAN 在准确性、效率和可扩展性之间取得了优异的平衡，适用于实际的高分辨率图像恢复任务。
## 12. `cs.CV` - PolyCrysDiff: 控制生成三维可计算多晶体材料结构 [PDF](https://arxiv.org/pdf/2603.11695), [HTML](https://arxiv.org/abs/2603.11695)
### Authors
Chi Chen,Tianle Jiang,Xiaodong Wei,Yanming Wang
### Background
多晶材料的三维微观结构对其力学和物理性能具有关键影响。虽然阐明结构-性能关系的关键步骤是构建真实的可控制微观结构，但这一挑战仍然艰巨。现有方法如马尔可夫随机场（MRF）和卷积神经网络（CNN）等，难以全面满足这一需求。
### Innovation
本文提出了一种名为PolyCrysDiff的框架，基于条件潜变量扩散，实现了从头生成可计算的三维多晶微观结构。多项质性和量化评估表明，PolyCrysDiff能够忠实再现目标晶粒形态、取向分布和三维空间关联，且在晶粒属性（如尺寸和球形度）控制方面达到$R^2$高于0.972，优于主流方法，如基于MRF和CNN的方法。生成的微观结构通过晶体塑性有限元方法（CPFEM）模拟验证了其计算能力和物理学有效性。利用PolyCrysDiff的可控生成能力，系统地探讨了晶粒级微观结构特征如何影响多晶材料的力学性能。
### Conclusion
此项发展有望为加速多晶材料的数据驱动优化和设计铺平关键路径。
## 13. `cs.CV` - PROMO: 可提示的高效高保真虚拟试穿 [PDF](https://arxiv.org/pdf/2603.11675), [HTML](https://arxiv.org/abs/2603.11675)
### Authors
Haohua Chen,Tianze Zhou,Wei Zhu,Runqi Wang,Yandong Guan,Dejia Song,Yibo Chen,Xu Tang,Yao Hu,Lu Sheng,Zhiyong Wu
### Background
虚拟试穿（VTON）已成为在线零售的核心能力，提供高度准确的试穿效果指导，有助于减少退货，惠及消费者和商家。尽管基于扩散的方法能够实现逼真的合成效果，但这些方法往往依赖于复杂的辅助参考网络，导致采样速度慢，从而在高保真度和效率之间存在持久的权衡。
### Innovation
我们从结构化图像编辑的角度来解决VTON问题，提出了一个包含主体保持、忠实纹理转移和无缝和谐化的强条件生成需求。为此，我们构建了一个通用的训练框架，该框架能够应用于更广泛的图像编辑任务。此外，由VTON生成的配对数据为训练通用编辑器提供了丰富的监督资源。提出了一种名为PROMO的可提示虚拟试穿框架，基于Flow Matching DiT骨干网络，结合潜在的多模态条件连接。通过利用条件效率和自我参考机制，该方法大幅减少了推理开销。PROMO在视觉保真度上超越了现有的VTON方法和一般图像编辑模型，同时提供了质量与速度的可竞争平衡。
### Conclusion
PROMO通过流匹配变换器、潜在多模态条件连接和自我参考加速技术，提供了一种高效且易于训练的高质量虚拟试穿解决方案。该框架展示了流匹配变换器在虚拟试穿领域的潜力，未来可拓展应用于更广泛的图像编辑任务。
## 14. `cs.CV` - OSCBench: 评估文本到视频生成中对象状态变化基准 [PDF](https://arxiv.org/pdf/2603.11698), [HTML](https://arxiv.org/abs/2603.11698)
### Authors
Xianjing Han,Bin Zhu,Shiqi Hu,Franklin Mingzhe Li,Patrick Carrington,Roger Zimmermann,Jingjing Chen
### Background
文本到视频（T2V）生成模型在生成视觉高质量和时间连贯的视频方面取得了快速进展。然而，现有的基准主要关注感知质量、文本与视频的对齐或物理合理性，忽略了动作理解的一个关键方面——文本提示中明确指定的对象状态变化（OSC）。OSC是由于某种行为引起一个对象状态的转变，如剥土豆或切柠檬。本文介绍了一个名为OSCBench的新基准，旨在评估T2V模型在OS方面的性能。OSCBench基于指令性烹饪数据构建，并系统性地组织了常规、新颖和组合场景的动作-对象交互，以测试分布内性能和泛化能力。
### Innovation
提出了OSCBench作为评估T2V模型在对象状态变化方面性能的新基准。该基准创新地从指令性烹饪数据中构建，并详细划分了动作-对象交互场景，以便测试模型在分布内性能和泛化的表现。采用人类用户研究和基于多模态大规模语言模型的自动评估方法来测评六种代表性的开源和专有T2V模型。
### Conclusion
尽管当前的T2V模型在语义和场景对齐方面表现出强大的性能，但在对象状态变化的准确性和时间一致性方面仍然存在挑战，尤其是对于新颖和组合设置。这些发现将对象状态变化定位为文本到视频生成的关键瓶颈，并确定了OSCBench作为一个诊断基准来促进面向状态的视频生成模型的发展。
## 15. `cs.CV` - 跨尺度注意网络用于高分辨率PM2.5预测 [PDF](https://arxiv.org/pdf/2603.11725), [HTML](https://arxiv.org/abs/2603.11725)
### Authors
Ammar Kheder,Helmi Toropainen,Wenqing Peng,Samuel Antão,Zhi-Song Liu,Michael Boy
### Background
Vision Transformers在时空预测方面取得了显著成功，但其对于现实环境监测中所需的超高清、大陆规模领域的可扩展性仍然有限。例如，1公里分辨率的整个欧洲空气质量图包含2900万个像素，远远超过了基本自我注意力的界限。
### Innovation
提出了CRAN-PM，这是一种双分支Vision Transformer，利用跨分辨率注意力来高效地融合25公里的全球气象数据与当前时间的1公里高分辨率PM2.5局部数据。引入了与海拔相关的自我注意力和受风向指导的交叉注意力，迫使网络学习物理一致的特征表示。该模型完全可训练且内存效率高，在单个GPU上生成完整的2900万个像素的欧洲地图仅需1.8秒。
### Conclusion
在2022年整个欧洲每天的PM2.5预测中（362天，2971个欧洲环境机构（EEA）站），与最佳单尺度基线相比，CRAN-PM将T+1和T+3的RMSE分别降低了4.7%和10.7%，并且复杂地形中的偏差降低了36%。
## 16. `cs.CV` - COTONET：基于YOLO11的一种棉花检测算法，用于发育阶段棉铃检测 [PDF](https://arxiv.org/pdf/2603.11717), [HTML](https://arxiv.org/abs/2603.11717)
### Authors
Guillem González,Guillem Alenyà,Sergi Foix
### Background
棉花采摘是一个关键阶段，棉铃被物理操作可能会导致纤维退化。为了维持最高的质量，采摘方法必须模拟细致的手动抓握，以保留棉花的内在特性。自动化这一过程需要能识别不同发育阶段棉铃的系统。然而，识别不同成熟度的棉铃是一项具有挑战性的任务。
### Innovation
本文提出了COTONET，一种增强的定制YOLO11模型，结合了注意力机制以提高难以检测实例的识别率。该模型对不可学习操作引入梯度，增强了形状和特征提取。主要的架构修改包括使用Squeeze-and-Exitation区块代替卷积块，重新设计结合注意力机制的骨干网络，以及用Content Aware Reassembly of Features (CARAFE)替换标准上采样操作。除此之外，模型还集成了Simple Attention Modules (SimAM)进行主特征聚合，以及Parallel Hybrid Attention Mechanisms (PHAM)进行通道、空间和坐标注意力。这种配置提供了更高的灵活性和鲁棒性，能够解释棉花作物生长的复杂性。
### Conclusion
COTONET在参数量较小的情况下（7.6M参数，27.8 GFLOPS），展示出为低资源边缘计算和移动机器人技术适用性。该模型优于标准YOLO基线，mAP50达到了81.1%，mAP50-95达到了60.6%。
## 17. `cs.CV` - SoulX-LiveAct: 通过邻接强制和ConvKV记忆实现小时尺度实时人类动画 [PDF](https://arxiv.org/pdf/2603.11746), [HTML](https://arxiv.org/abs/2603.11746)
### Authors
Dingcheng Zhen,Xu Zheng,Ruixin Zhang,Zhiqi Jiang,Yichao Yan,Ming Tao,Shunshun Yin
### Background
自回归（AR）扩散模型为视频合成等序列生成任务提供了一个有前景的框架，结合了扩散建模和因果推理。尽管它们支持流式生成，但现有的AR扩散方法在效率上存在挑战。论文主要讨论了长时间人类动画中的两个关键挑战：首先，大多数强迫策略会导致不匹配的扩散状态传播采样级表示，进而产生不一致的学习信号和不稳定的收敛；其次，历史表示可能会无限制增长且缺乏结构，这使得状态的有效重用受到限制，从而严重影响推理效率。
### Innovation
为解决上述挑战，作者提出了邻接强制（Neighbor Forcing），这是一种满足扩散步骤一致性的时间相邻帧传播方法，它在相同的噪声条件下传播时序相邻帧。这种方法提供了分布对齐的学习信号并保持整个AR链中的漂移。在此基础上，作者引入了一个结构化的ConvKV记忆机制，该机制将因果注意力中的键和值压缩为固定长度的表示，从而实现常量记忆推理并真正支持无限视频生成，而不依赖于短期运动帧记忆。实验表明，该方法在训练收敛、小时尺度生成质量和推理效率方面明显优于现有AR扩散方法。
### Conclusion
通过SoulX-LiveAct方法，实现了小时尺度的实时人类动画，并支持在最少两块NVIDIA H100或H200 GPU上达到每秒20帧的实时视频推理。数值结果显示，该方法在唇部同步准确性、人类动画质量和情感表达方面达到了最先进的性能，同时具有最低的推理成本。
## 18. `cs.CV` - 基于隐匿感知稀疏3D手关节的可控第一人称视频生成 [PDF](https://arxiv.org/pdf/2603.11755), [HTML](https://arxiv.org/abs/2603.11755)
### Authors
Chenyangguang Zhang,Botao Ye,Boqi Chen,Alexandros Delitzas,Fangjinhua Wang,Marc Pollefeys,Xi Wang
### Background
在虚拟现实和嵌入式AI的自视角应用中，可控的运动可自视频生成至关重要。然而，现有的方法在实现3D一致的手部详细动作时经常遇到困难。它们通过2D轨迹或隐式姿态来处理，这使得3D几何结构被转换为空间上模糊的信号，或过度依赖于人类先验知识。在严重的自视角遮挡下，这会导致运动不一致性和虚构的图像，同时也阻碍了不同身体替代品间的手部通用性。
### Innovation
本文提出了一种新颖的框架，通过单个参考帧生成自视角视频，利用稀疏3D手关节作为无特定载体控制信号，具有清晰的语义和几何结构。该框架包含一个高效的控制模块，可以解决遮挡问题，同时完全保留3D信息。具体而言，该模块从源参考帧中提取感知遮挡的特征，并采用基于3D的加权机制来动态处理目标关节的遮挡现象，同时直接将3D几何嵌入引入潜空间，以强制结构一致性。此外，开发了一种自动标注流水线，产生了超过一百万个高质量的自视角视频片段，并配以精确的手部轨迹。在此基础上构建了一个跨载体基准。大量实验表明，该方法在性能上显著优于现有基线，生成了高质量的自视角视频，并且在不同身体替代品间的手部通用性方面表现出色。
### Conclusion
本文提出的方法在可控第一人称视频生成领域表现优异，不仅提高了生成视频的质量和现实感，还克服了不同身体之间的通用性问题。通过稀疏3D手关节和自动标注流程的创新应用，证明了其在实际应用中的潜力。
## 19. `cs.CV` - BackdoorIDS: 预训练视觉编码器的零样本后门检测 [PDF](https://arxiv.org/pdf/2603.11664), [HTML](https://arxiv.org/abs/2603.11664)
### Authors
Siquan Huang,Yijiang Li,Ningzhi Gao,Xingfu Yan,Leyu Shi
### Background
现有的自我监督和多模态视觉编码器能够学习强大的视觉表示，广泛应用于下游视觉任务和大型视觉-语言模型（LVLMs）。然而，下游用户经常依赖于来历不明的第三方预训练编码器，这使得他们暴露在后门攻击中。本文主要研究了预训练视觉编码器的后门攻击检测方法。
### Innovation
本文提出了一种名为BackdoorIDS的简单而有效的后门检测方法，可以在推理时间进行零样本检测。该方法基于注意力劫持和修复两个观察成果，通过在输入遮盖过程中提取嵌入序列并使用基于密度的聚类方法（如DBSCAN）进行检测。BackdoorIDS可以在不重新训练和完全零样本的方式下工作，使其与各种编码器架构兼容，包括CNN、ViTs、CLIP和LLaVA-1。
### Conclusion
全面的实验表明，BackdoorIDS在各种攻击类型、数据集和模型家族下都能持续优于现有防御方法。此外，BackdoorIDS是一种即插即用的解决方案，适用于多种编码器架构，确保了其广泛的应用适用性。
## 20. `cs.CV` - VTEdit-Bench: 虚拟试穿中的多参考图像编辑模型综合基准 [PDF](https://arxiv.org/pdf/2603.11734), [HTML](https://arxiv.org/abs/2603.11734)
### Authors
Xiaoye Liang,Zhiyuan Qu,Mingye Zou,Jiaxin Liu,Lai Jiang,Mai Xu,Yiheng Zhu
### Background
随着虚拟试穿（VTON）技术的不断发展，其应用场景日益多样化，现有的专门化VTON模型难以满足这些新需求。与此同时，通用多参考图像编辑模型取得了显著进展，展示了强大的视觉编辑能力，为更加灵活的VTON系统提供了新的方向。然而，由于缺乏系统性评估基准，对通用编辑器在VTON中的能力和局限性研究仍显不足。
### Innovation
本文介绍了VTEdit-Bench，一种全面的评估基准，用来评估跨多种真实VTON场景的通用多参考图像编辑模型，包含24,220对测试图像，涵盖五种代表性的VTON任务，有助于系统地分析鲁棒性和泛化能力。还提出了VTEdit-QA，一种参考感知的VLM为基础的评估者，从模型一致性、布料一致性及整体图像质量三个方面评估VTON的性能。
### Conclusion
通过框架评估了八种通用编辑模型和七种专门化VTON模型，结果表明顶级的通用编辑模型在常规任务上具有竞争力，并且在更困难的场景下泛化更稳定，但在复杂的参考配置尤其是多布料条件下发面仍有挑战。
