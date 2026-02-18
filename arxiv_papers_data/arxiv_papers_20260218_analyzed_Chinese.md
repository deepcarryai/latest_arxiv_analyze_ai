# 20260218
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 作为轨迹主导帕累托优化的智能 [PDF](https://arxiv.org/pdf/2602.13230), [HTML](https://arxiv.org/abs/2602.13230)
### Authors
Truong Xuan Khanh,Truong Quynh Hoa
### Background
尽管人工智能取得了近期的进步，许多系统在长时适应性方面仍表现出停滞不前的状态，尽管一直有持续的性能优化。这些限制并非主要来自于不足的学习、数据或模型容量，而是源于智能随时间优化的更深层次结构属性。研究表明智能应被视为一个多目标权衡的轨迹水平现象。
### Innovation
我们提出了轨迹主导的帕累托优化，这是一种面向路径的帕累托最优性推广，其中主导性是在完整轨迹上定义的。我们定义了陷阱逃逸难度指数（TEDI），这是一种综合几何度量，捕捉逃逸距离、结构约束和行为惯性。方法显示了动态智能天花板作为轨迹层面上主导性的不可避免的几何结果，独立于学习进度或架构规模。此外，我们引入了帕累托陷阱的形式分类，并使用最小的代理环境模型来说明由此产生的轨迹层面上的分歧。
### Conclusion
这些结果将智能的焦点从终端性能转移到优化几何，提供了一种原则性的框架来诊断和克服适应系统中的长时发展限制。
## 2. `cs.AI` - VeRA: 大规模验证推理数据增强 [PDF](https://arxiv.org/pdf/2602.13217), [HTML](https://arxiv.org/abs/2602.13217)
### Authors
Zerui Cheng,Jiashuo Liu,Chunjie Wu,Jianzhu Yao,Pramod Viswanath,Ge Zhang,Wenhao Huang
### Background
当前大多数评估方案的“静态”性质是一个主要问题：重复使用相同的问题，这使得模型可以记住问题，利用格式，最终导致能力饱和。为了准确衡量AI进步，我们需要一种自开始就具备鲁棒性、而非经过检测的方法。为解决这一挑战，本文提出了VeRA（验证推理数据增强）框架，该框架能够将基准问题转换为可执行规范，生成无限数量的可靠标签，无需人力成本，并能够自行验证并计算答案。
### Innovation
VeRA通过自然语言模板、有效配置生成器和确定性验证器组成，实现基于单一种子问题生成无限数量的验证变体。VeRA有两种操作模式：VeRA-E专注于逻辑保持不变的问题重写，用于识别记忆形式和真正的推理能力；VeRA-H系统地增加复杂性以维持可验证性，用于生成新的、难以处理的任务。该框架不仅提高了评估质量，还能够实现无监管的困难任务生成，并确立了验证基准的通用范式。
### Conclusion
VeRA改变了基准的静态对象性质，使其转化为可即时生成新鲜、验证过的实例的执行规范，增强了评估的稳健性和成本效益。通过VeRA，评估在任何可验证领域都可以无限扩展且不影响标签完整性。本文还启用了所有代码和数据集的开源，以刺激未来研究。
## 3. `cs.AI` - 何时快思与慢思？AMOR：基于熵的元认知门控以动态切换SSM-注意力 [PDF](https://arxiv.org/pdf/2602.13215), [HTML](https://arxiv.org/abs/2602.13215)
### Authors
Haoran Zheng
### Background
传统的Transformer模型在处理每一个位置时分配相同的计算量，而不管该位置的难度如何。状态空间模型（SSMs）虽然提供了高效的选择，但在长时间序列信息检索时却难以精确获取所需信息。鉴于认知二元理论（卡尼曼，2011），本文提出了一种混合架构AMOR，该架构仅在SSM骨干网络“不确定”（通过预测熵衡量）时动态地参与稀疏注意力。
### Innovation
AMOR通过将SSM隐藏状态投影为稀疏注意力的键值（Ghost KV），在每层避免了O(n^2)的繁复注意力计算，而是重用了O(n)的SSM计算。与标准Transformer相比，AMOR在小型合成检索任务上获得了更高的效率和性能，仅在22%的位置上启用注意力机制，即在这一点上实现了完美的检索精度。此外，预测熵可靠地指示了检索需求，两者之间的差距为1.09 nats（几乎占熵范围的一半），并且提出的这种方法提供了可解释的自适应计算，注意力路由决策可以从信息理论的角度理解。
### Conclusion
AMOR能够动态且高效地切换SSM-注意力，通过预测熵来决定何时以及在何处进行稀疏的注意力计算，实现了高效的计算和准确的信息检索。
## 4. `cs.AI` - BotzoneBench: 通过分级AI锚点实现可扩展的LLM评估 [PDF](https://arxiv.org/pdf/2602.13214), [HTML](https://arxiv.org/abs/2602.13214)
### Authors
Lingfeng Li,Yunlong Lu,Yuefei Zhang,Jingyu Yao,Yixin Zhu,KeYuan Cheng,Yongyi Wang,Qirui Zheng,Xionghui Yang,Wenxin Li
### Background
大型语言模型（LLMs）在需要战略决策的交互环境中越来越普及，但对其能力的系统评估仍然具有挑战性。现有评估基准主要通过孤立任务评估静态推理能力，难以捕捉动态的战略能力。最近的游戏评估使用了LLM对战LLM的竞赛来产生相对排名，但这种方法依赖于瞬时模型池，计算成本高昂，并无法提供稳定的长期性能基准。核心挑战在于建立一个可扩展的评估框架，该框架能够以一致且可解释的标准来衡量LLMs的战略推理能力，而非与波动的同侪模型进行比较。
### Innovation
通过将LLM评估基准与固定技能分级的游戏AI（GAI）链路联系起来，实现线性时间的绝对技能测量，并具有跨时间的稳定可解释性。基于Botzone平台的成熟竞争基础设施，博兹区基准（BotzoneBench）评估了八个不同游戏中的五个旗舰模型的177,047个状态-动作对，揭示了显著的性能差异，并识别出不同的战略行为。表现最佳的模型在多个领域达到与中到高专业化游戏AI相当的熟练度。
### Conclusion
这种锚定评估范例不仅适用于游戏领域，还可以推广到任何拥有明确技能等级结构的领域，建立了评估交互式AI能力的可扩展且可复用框架。
## 5. `cs.AI` - PlotChain: 确定性检查点评估多模态大语言模型在工程图读取中的表现 [PDF](https://arxiv.org/pdf/2602.13232), [HTML](https://arxiv.org/abs/2602.13232)
### Authors
Mayank Ravishankara
### Background
随着多模态大语言模型（MLLMs）的发展，评估这些模型在特定领域的复杂任务上表现的需求日益增加。工程图读取是一项重要任务，它需要从经典图表（如Bode/FFT图、阶跃响应图、应力-应变图、泵曲线等）中提取定量值。传统的评估方法主要依赖于光学字符识别（OCR）或自由形式的标题生成。PlotChain为这类任务提供了一个新的基准，使得可以进行定量的、可复现的评估。
### Innovation
PlotChain通过确定性的、基于检查点的评估方法革新了对MLLMs在工程图读取上的测试。其创新点包括：1）包含15个图族，共450个渲染图，每个项目都从已知参数生成，并配对有精确的地面真相值；2）每项包含中间’cp_‘字段，可以隔离子技能并定位失败；3）使用标准化的评估协议和评分标准，包括使用 per-field 损差阈值，以反映人类读图的细致程度；4）公开生成器、数据集、模型输出、评分代码和校验和，以支持完全可复现的运行和重新评分。
### Conclusion
经过标准化协议评估，顶级模型在不同参数下的通过率分别为80.42%（Gemini 2.5 Pro）、79.84%（GPT-4.1）和78.21%（Claude Sonnet 4.5）。尽管许多领域表现出强大性能，但频域任务仍然脆弱，带通响应和FFT谱图等特定技能仍有困难。研究者提供了开源资源，以便其他研究者进行回顾性重新评分，采用不同的评估标准。
## 6. `cs.AI` - 基于变异的关键：一种基于变异的大型语言模型生成文本检测框架 [PDF](https://arxiv.org/pdf/2602.13226), [HTML](https://arxiv.org/abs/2602.13226)
### Authors
Xuecong Li,Xiaohong Li,Qiang Hu,Yao Zhang,Junjie Wang
### Background
检测由大型语言模型（LLMs）生成的文本至关重要但具有挑战性。现有的检测器依赖于不切实际的假设，如白盒设置，或仅依赖于文本级别的特征，这导致了检测精度不高。
### Innovation
本文提出了一种简单但有效且实用的LLM生成文本检测方法，即VaryBalance。VaryBalance的核心思想在于，与LLM生成的文本相比，人类文本及其通过LLM重写的版本之间存在更大的差异，该方法利用这一观察通过均值标准差进行了量化，并成功区分开人类文本和LLM生成的文本。
### Conclusion
全面的实验表明，VaryBalance在AUROC指标上优于当前最先进的检测器Binoculars，高达34.3%，并且在多种生成模型和语言方面保持了鲁棒性。
## 7. `cs.AI` - LLMs中幻觉的几何分类 [PDF](https://arxiv.org/pdf/2602.13224), [HTML](https://arxiv.org/abs/2602.13224)
### Authors
Javier Marín
### Background
在大型语言模型（LLMs）中，术语'幻觉'混杂了具有不同几何特征的多种现象。研究者们提出了一个分类系统，将幻觉分为三类：不忠实（未能与提供的上下文互动）、虚构（创造与语义无关的内容）和事实错误（在正确概念框架内的错误断言）。他们发现标准基准下的幻觉检测具有领域局部性，但在不同领域间却呈现零水平的检测表现。
### Innovation
研究引入了一种几何分类法来区分和描述大型语言模型中的幻觉现象。通过对比模型自动生成的幻觉与人类创作的虚构内容，研究揭示了不同类型的幻觉在几何空间中的不同特征。研究还指出了基准测试和人类创作内容之间的差异，并提出了一个几何分类系统来界定基于嵌入式检测的范围。
### Conclusion
嵌入式检测只适用于前两类幻觉，即不忠实和虚构，而事实错误需要外部验证机制进行检测。几何结构的不同源于底层现象的不同。这项贡献提供了一种清晰的几何分类法，明确了嵌入式检测的应用范围。
## 8. `cs.AI` - 商业保险承保中的代理式AI及其对抗性自我批判机制 [PDF](https://arxiv.org/pdf/2602.13213), [HTML](https://arxiv.org/abs/2602.13213)
### Authors
Joyjit Roy,Samaresh Kumar Singh
### Background
商业保险承保是一个劳动密集型过程，需要手动审查大量文档以评估风险和确定保费。尽管人工智能能显著提高效率，但现有解决方案在确保可靠性方面缺乏全面的推理能力和内部机制，尤其是在受监管的高风险环境中。完全自动化在这种情况下是不可行和不建议的，因为人类判断和问责制至关重要。
### Innovation
本研究提出了一种代理式的决策负反馈、人工在环系统，结合了对抗性自我批判机制作为受监管承保工作流的边界安全架构。该系统中的批评者代理会在向人类审阅者提交建议前挑战主要代理的结论。这种内部系统提供了平衡机制，填补了受监管工作流中AI安全的关键空白。此外，研究还开发了一种形式化的失败模式分类体系，以定义决策负反馈代理的潜在错误类型，该分类体系为高风险应用中的风险识别和风险管理提供了结构化的框架。实验使用500个专家验证的承保案例评估表明，对抗性批判机制将AI幻觉率从11.3%降低到3.8%，并且决策准确性提高到96%。
### Conclusion
这些发现表明，对抗性自我批判支持在监管领域更安全的AI部署，并为人类监督不可或缺的责任性整合提供了模型。
## 9. `cs.AI` - 扩展逻辑Scaling the Scaling Logic: 基于代理的逻辑推理的元合成 [PDF](https://arxiv.org/pdf/2602.13218), [HTML](https://arxiv.org/abs/2602.13218)
### Authors
Bowen Liu,Zhi Wu,Runquan Xie,Zhanhui Kang,Jia Li
### Background
强化学习（RL）从验证奖励进行训练时，验证训练信号的可扩展性仍然是一个关键瓶颈。逻辑推理是一个自然的载体：约束是形式化的，答案是通过程序验证的。然而，早期生成管道要么依赖专家编写的代码，要么在固定的模板或骨架中操作，这限制了增长主要局限于实例级别的扰动。
### Innovation
提出了一种名为SSLogic的基于代理的元合成框架，通过迭代生成和修复生成器-验证器程序对，实现任务家族级别的可扩展性，形成封闭的“生成-验证-修复”循环，允许连续的家庭进化，同时控制难度。为了确保可靠性，引入了一种多门验证协议，结合了多种策略的一致性检查和对抗性盲审，独立的代理通过编写和执行代码解决实例来筛选含糊或不良定义的任务。
### Conclusion
从400个种子家族开始，经过两轮进化，扩展到953个家族和21,389个可验证实例（从5,718个）。以SSLogic生成的训练数据为准，则在匹配的训练步骤中，SSLogic总体上提高了SynLogic 5.2%，BBEH 1.4%，AIME25 3.0%和Brumo25 3.7%的表现。
## 10. `cs.AI` - Stay in Character, Stay Safe: 双环对抗自进化以确保安全的角色扮演代理 [PDF](https://arxiv.org/pdf/2602.13234), [HTML](https://arxiv.org/abs/2602.13234)
### Authors
Mingyang Liao,Yichen Wan,shuchen wu,Chenxi Miao,Xin Shen,Weikang Li,Yang Li,Deguo Xia,Jizhou Huang
### Background
基于LLM的角色扮演技术正迅速提高真实性，然而，基于角色的限制更强会增加脱缰攻击（jailbreak attacks）的风险，尤其是针对高风险或消极角色。当前大多数研究主要采用训练时的解决方案（如数据清理或目标导向的正则化）来解决这一问题。然而，这些方法在不同条件下需要高昂的维护成本，可能会影响角色的在场表现行为，并且对于闭包权重的大规模LLM来说通常不可行。
### Innovation
本文提出了一种无需训练的双环对抗自进化框架（Dual-Cycle Adversarial Self-Evolution framework）。该框架包含两个耦合循环：一个针对角色的攻击循环不断生成更强的脱缰提示，另一个角色扮演防御循环从观察到的失败中提取分层知识库，涵盖全局安全规则、以角色为基础的限制以及安全的任务示例。在推理时，防御循环检索并组合层级化的知识来指导生成，确保生成的响应忠实于目标角色同时满足安全约束。广泛的实验结果显示，该框架在多个私有LLM上表现出对角色保真度和脱缰攻击抵抗性的持续改进，并且能够在未见过的角色和攻击提示上展现出稳健的一般性。
### Conclusion
本文提出的双环对抗自进化框架显著提升了角色扮演代理的角色忠诚度和安全防范能力，并且在面对未见过的角色和攻击提示时仍然保持了稳健的一般性。该框架能够明显优于现有的强大基准，展示了在未见过的角色或挑战性的攻击提示下的潜力。
## 11. `cs.AI` - 负责任的人工智能在商业中的应用 [PDF](https://arxiv.org/pdf/2602.13244), [HTML](https://arxiv.org/abs/2602.13244)
### Authors
Stephan Sandfuchs,Diako Farooghi,Janis Mohr,Sarah Grewe,Markus Lemmen,Jörg Frochte
### Background
人工智能（AI）和机器学习（ML）已经从研究和试点项目扩展到日常商业运营，并且生成式AI正在加速这些技术在流程、产品和服务中的应用。
### Innovation
本文提出了面向组织实践的责任人工智能的概念，特别是针对中小企业。文章将责任人工智能分为四个关键区域，分别是合法合规、易于理解、可持续和数据主权，从而确保AI系统的引入和运行。
### Conclusion
文章总结了建立治理、文档记录、安全运营以及可持续性考虑，并提出了一套实施路线图的具体步骤。
## 12. `cs.AI` - 评判者评判：多LLM评估高质量K--12科学教学材料的人类验证 [PDF](https://arxiv.org/pdf/2602.13243), [HTML](https://arxiv.org/abs/2602.13243)
### Authors
Peng He,Zhaohui Li,Zeyuan Wang,Jinjun Xiong,Tingting Li
### Background
为K--12科学设计高质量且符合标准的教材需要大量时间和专业知识，本研究通过分析人类专家对AI生成评估的审查，探索将这些见解转化为未来基于通用人工智能（GenAI）的教学材料设计代理的设计原则。
### Innovation
研究选择12个高质量的跨生命科学、物理科学和地球科学的课程单元，使用EQuIP评估标准，促使GPT-4、Claude和Gemini生成评分和理由，经过两个科学教育专家的独立评审，揭示LLM评估与专家观点的契合点和差异点，以优化未来基于GenAI的教学材料设计。
### Conclusion
本研究通过人类专家验证多LLM评估，揭示了其在评估高质量K--12科学教学材料方面的优势与不足，为开发专门领域内的GenAI代理提供了直接的设计依据和原则，以支持K--12科学教育中的教材设计。
## 13. `cs.AI` - Boltz 是亚原子级别表示学习的强基线 [PDF](https://arxiv.org/pdf/2602.13249), [HTML](https://arxiv.org/abs/2602.13249)
### Authors
Hyosoon Jang,Hyunjin Seo,Yunhui Jang,Seonghyun Park,Sungsoo Ahn
### Background
分子学习中的基础模型发展了两条并行的路径：蛋白质模型，通常利用进化信息学习氨基酸级别的折叠表示；以及小分子模型，专注于学习原子级别的表示以进行如ADMET性质预测等任务。最新的蛋白质中心模型如Boltz已经在蛋白质-配体共折叠中达到了原子级别的精细度，但它们在小分子任务上的原子级别表现尚未被探索。开放的问题在于这些共折叠模型是捕捉到了可转移的化学物理学知识，还是依赖于蛋白质进化信号，这将限制它们在小分子任务上的实用性。
### Innovation
本文研究了Boltz在多种小分子基准上的原子级别表示质量。结果显示，Boltz在ADMET性质预测任务中与专门的基线竞争，并有效用于分子生成和优化。这些发现表明，最先进的蛋白质中心模型的表示能力尚未完全开发，并将Boltz定位为小分子亚原子级别表示学习的有效基线。
### Conclusion
Boltz在小分子亚原子级别表示学习中展示了强大的基准性能，这表明这些最新的蛋白质模型在亚原子级别的表示能力上有着未被充分开发的潜力。
## 14. `cs.AI` - Directional Concentration Uncertainty: A representational approach to uncertainty quantification for generative models [PDF](https://arxiv.org/pdf/2602.13264), [HTML](https://arxiv.org/abs/2602.13264)
### Authors
Souradeep Chattopadhyay,Brendan Kennedy,Sai Munikoti,Soumik Sarkar,Karl Pazdernik
### Background
在使生成模型可靠和可信的关键任务中，不确定性量化（UQ）的方法已经开始显示出令人鼓舞的潜力。然而，这些方法往往依赖于固定的启发式方法，无法跨任务和模态进行有效的泛化。先前的UQ方法通常需要针对特定任务的启发式策略，这限制了它们的广泛应用。
### Innovation
本文提出了一种新颖的UQ框架，这种方法高度灵活，并且在性能上接近或超过了先前启发式方法的水平。本文引入了基于vMF分布的“方向集中不确定性”（DCU）这一新的统计程序，通过测量连续生成输出嵌入的空间分散来捕捉不确定性，而不依赖任何特定任务的启发式策略。此外，研究还展示了DCU在多模态复杂任务中的良好泛化能力。
### Conclusion
本文提出了DCU这一代表性的不确定性量化方法，并为该方法在多模态和自主系统框架中的更广泛应用建立了框架。DCU测量生成输出嵌入的空间分散，从而量化不确定性，这些输出来自语言模型且无需任何任务特定的启发式策略。实验结果表明，DCU的校准水平与先前作品如语义熵相当，并且在多模态任务中的泛化性能优良。
## 15. `cs.AI` - 通过虚拟节点诱导边预测增强的LLM谣言检测 [PDF](https://arxiv.org/pdf/2602.13279), [HTML](https://arxiv.org/abs/2602.13279)
### Authors
Jiran Tao,Cheng Wang,Binyan Jiang
### Background
社交媒体网络上谣言的传播削弱了信息的可信度，当前的检测方法难以捕捉谣言传播的复杂模式。传统的基于文本嵌入的方法忽略了谣言传播路径上的文本一致性，影响了谣言识别的准确性。
### Innovation
本文提出了一种新的框架，利用大型语言模型（LLMs）来解决问题。该方法通过分析信息子链、分配谣言概率和智能构建虚拟节点连接来捕捉微妙的谣言信号，修改原始的图结构，这在捕捉细微谣言信号上是一个重要突破。同时，为了应对LLMs在谣言识别中的固有限制，开发了一个结构化的提示框架，以减轻模型偏差并确保稳健的图学习性能。
### Conclusion
提出的方法是模型无关的，可以方便地与未来进一步微调的LLMs和图技术集成，这可能在不修改原始算法的情况下提高预测性能。
## 16. `cs.AI` - 全球技术治理中的AI偏见审计 [PDF](https://arxiv.org/pdf/2602.13246), [HTML](https://arxiv.org/abs/2602.13246)
### Authors
Jason Hung
### Background
本文展示了一个针对大型语言模型（LLMs）全球审核项目探索性阶段的结果。该探索性阶段使用了全球AI数据集（GAID）项目作为框架，对Llama-3 8B模型进行了压力测试，评估了技术AI治理意识中的地理和经济差异偏见。通过在213个国家和8项技术指标下使用1,704次查询进行压力测试，作者发现了一条显著的数字障碍和南北差异，即只有11.4%的查询回答能提供数字/事实响应，且这些响应的实证有效性仍有待验证。全球北方地区和技术知识高度集中在高收入区域，而全球南方的低收入国家则在系统性信息方面处于不利地位。
### Innovation
该研究通过使用全球AI数据集项目框架，对Llama-3 8B模型进行了压力测试，评估了技术AI治理意识中的地理和经济差异偏见，并提出了当前AI对齐和训练过程强化了地缘经济和地缘政治不对称的现象，并呼吁需要更多的包容性数据表示以确保AI成为真正的全球资源。
### Conclusion
当前的AI对齐和训练过程强化了地缘经济和地缘政治不对称，研究指出需要更多包容性数据表示来确保AI作为全球资源的有效性。
## 17. `cs.AI` - 一种用于Spiking神经网络在线和硬件感知训练的反馈控制优化器 [PDF](https://arxiv.org/pdf/2602.13261), [HTML](https://arxiv.org/abs/2602.13261)
### Authors
Matteo Saponati,Chiara De Luca,Giacomo Indiveri,Benjamin Grewe
### Background
与传统的类神经网络不同，生物神经网络可以通过稀疏的神经元活动、循环连接和局部学习规则来解决复杂的认知任务。这些机制为神经形态计算提供了设计原则，解决了现代计算中的能耗问题。然而，大多数混合信号神经形态设备依赖半监督或无监督的学习规则，这在监督学习任务中不太有效。这种缺乏针对芯片内学习的可扩展解决方案限制了混合信号设备在实现可持续智能边缘系统方面的潜力。因此，本研究旨在解决这一挑战。
### Innovation
本文提出了一种全新的学习算法，该算法将基于突触的权重更新与反馈控制信号集成到Spiking神经网络(SNN)中，并通过一个突发控制生成器来引导SNN活动和驱动权重更新，实现可扩展和本地的芯片内学习。该研究首先在各种分类任务中评估了该算法，表明带有反馈控制训练的一层SNN可以达到与人工神经网络相当的性能。然后在混合信号神经形态设备上测试其实现，并评估了其对于超参数不匹配的鲁棒性。结果显示，反馈控制优化器适用于神经形态应用，促进了边缘应用中可扩展和芯片内学习解决方案的发展。
### Conclusion
研究结果表明，该反馈控制优化器是兼容神经形态应用的，为进一步在边缘应用中实现可扩展和芯片内学习解决方案打开了新的可能性。
## 18. `cs.AI` - LLMs中 transgender人群的隐性偏见 [PDF](https://arxiv.org/pdf/2602.13253), [HTML](https://arxiv.org/abs/2602.13253)
### Authors
Micaela Hirsch,Marina Elichiry,Blas Radi,Tamara Quiroga,David Restrepo,Luciana Benotti,Veronica Xhardez,Jocelyn Dunstan,Enzo Ferrante
### Background
大型语言模型（LLMs）对LGBTQ+人群表现出偏见。尽管安全培训可以减少明确的偏见表达，但过去的研究表明，隐含的刻板印象驱动的关联仍然存在。这项研究中，研究者调查了LLMs在两种主要场景下对跨性别者的隐性偏见：首先，通过词义关联测试评估LLMs对“跨性别”和“cisgender”负面与正面概念的不对等关联；其次，在医疗决策应用的背景下评估LLMs的隐性偏见。逻辑建模了医疗预约分配任务，以考察LLMs在医疗决策中的表现。
### Innovation
研究设计了一个医疗预约分配任务，让模型在医学领域中更容易产生刻板印象的领域进行选择，其中模型扮演调度代理选择跨性别和cisgender候选人。这项研究是第一个使用医疗情境评估LLMs关于跨性别群体的隐性偏见的研究。研究揭示了LLMs在外观、风险和真实性等类别中对跨性别群体的负面影响。
### Conclusion
研究结果显示，跨性别群体在外观、风险和真实性等类别中表现出更强的负面关联。在分配任务中，跨性别候选人更多地被分配到性传播感染和心理健康服务中，而cisgender候选人则更倾向于被分配到妇科和乳腺护理领域。研究表明，需要进一步研究来解决LLMs中的隐性刻板印象驱动的偏见，以确保在医疗应用中对跨性别群体的公平对待。
## 19. `cs.AI` - MergePipe：一种预算感知的参数管理系统以实现可扩展的大语言模型合并 [PDF](https://arxiv.org/pdf/2602.13273), [HTML](https://arxiv.org/abs/2602.13273)
### Authors
Yuanyi Wang,Yanggan Gu,Zihao Wang,Kunxi Li,Yifan Yang,Zhaoyi Yan,Congkai Xie,Jianmin Wu,Hongxia Yang
### Background
现有的大语言模型（LLM）合并实现将模型参数视为无结构的文件，并以无状态的一次性方式执行合并，这导致了过多的磁盘I/O、冗余的参数扫描以及较差的可扩展性。随着专家模型数量的增加，这一问题变得尤为严重。
### Innovation
MergePipe是一个参数管理系统，用于实现可扩展的大语言模型合并。它首次将LLM合并视为数据管理和执行问题，并引入了参数注册表驱动的抽象。MergePipe的核心是一个成本感知的规划器，它显式地建模专家参数的I/O，并遵守用户指定的I/O预算。随后是一个流式执行引擎，它在事务保证下实现合并模型。研究的关键见解是，尽管基模型读取和输出写入是不可避免的，但专家参数读取主导了合并成本并成为主要优化目标。通过在整个规划和执行过程中使专家访问预算化，MergePipe减轻了天真管道的O(K) I/O增长，并实现了可预测的可扩展行为。实验结果表明，MergePipe将总I/O降低了最多一个数量级，并相比最先进的大语言模型合并管道实现了最多11倍的端到端加速（高达90%的墙时间减少）。
### Conclusion
MergePipe通过预算感知的方式管理和执行参数，显著降低了大语言模型合并中的I/O开销，并实现了可预测的可扩展性。该系统展示了在合并多个专家模型时的高效率和强大的扩展性。
## 20. `cs.AI` - 学习有生理依据的语音频谱表示以进行语音情绪识别 [PDF](https://arxiv.org/pdf/2602.13259), [HTML](https://arxiv.org/abs/2602.13259)
### Authors
Xu Zhang,Longbing Cao,Runze Yang,Zhangkai Wu
### Background
言语情绪识别（SER）对于人造机器人任务至关重要，包括社交机器人互动和机器人心理诊断，其中可解释和高效的模型对于安全性和性能至关重要。现有的深度模型尽管在大型数据集上训练，但仍难以解释，往往无法充分建模情绪的声学信号，无法捕捉和分析情绪语音行为的核心生理特征。现有的生物学研究揭示，声音幅度和相位的变化与通过声门源和声道滤波器传递的情绪有关联。然而，大多数现有深度模型只涉及到幅度，而未能将幅度和相位的生理特征结合起来。
### Innovation
本文提出了PhysioSER，一种基于生理学的语音频谱时空表征学习方法。PhysioSER通过结合幅度和相位视图来补充SSL模型，这些幅度和相位视图受到声学生理学（VAP）的启发，从而构造了一个紧凑且插拔即用的设计。该VAP启发式框架包含两条并行工作流：一个语音特征表示分支，基于VAP分解语音信号，嵌入到四元数领域，并使用Hamilton结构的四元数卷积来建模动态交互；一个基于固定SSL主干的潜在表示分支。然后，通过对比投影对齐框架对两个工作流的语音片段级特征进行对齐，并通过浅层注意力融合头部进行SER分类。
### Conclusion
通过在14个数据集、10种语言和6个骨干网络上的广泛评估，PhysioSER被证明在SER中是可解释且高效的，其实际有效性也在一个人造机器人平台上进行了实时部署验证。
## 21. `cs.CL` - GRRM: 组相对奖励建模在机器翻译中的应用 [PDF](https://arxiv.org/pdf/2602.14028), [HTML](https://arxiv.org/abs/2602.14028)
### Authors
Sen Yang,Shanbo Cheng,Lu Xu,Jianbing Zhang,Shujian Huang
### Background
虽然组相对政策优化（GRPO）为LLM训练后的模型提供了一个强大的框架，但在开放性领域如机器翻译中，其效果依赖于准确的组内排名。标准的标量质量度量（SQM）在此情境下表现不佳，因为它们孤立地评估候选者，缺乏必要的上下文比较，无法区分细微的语义差异。
### Innovation
本文提出了组质量度量（GQM）范式和基于此的组相对奖励模型（GRRM）。GRRM不同于传统的独立评分者，它能联合处理整个候选组，利用比较分析来严格解决相对质量的评估并实现自适应粒度差异。实验证明，GRRM在所有基线中的排名准确度具有竞争力。在此基础上，GRRM被集成到GRPO训练循环中，以优化翻译策略，并验证其能够不仅提升翻译质量，还解锁了类似于最新推理模型的推理能力。
### Conclusion
实验结果表明，我们的框架不仅提高了翻译的一般质量，还解锁了与最先进的推理模型相当的推理能力。我们提供了代码、数据集和模型检查点以供参考。
## 22. `cs.CL` - 上下文塑造大语言模型增强检索的事实核查有效性 [PDF](https://arxiv.org/pdf/2602.14044), [HTML](https://arxiv.org/abs/2602.14044)
### Authors
Pietro Bernardelle,Stefano Civelli,Kevin Roitero,Gianluca Demartini
### Background
大语言模型（LLMs）在多种任务中显示出强大的推理能力，但在长上下文条件下的表现仍不够一致。先前的研究主要集中在问答中的中段上下文降级问题上，而本研究则探讨了大语言模型在基于事实验证中的上下文影响。
### Innovation
研究使用了HOVER、FEVEROUS和ClimateFEVER三个数据集，以及不同参数量和模型家族的五个开源模型，评估了参数化事实知识以及证据在不同上下文长度中的影响。研究发现，LLMs不仅具备非平凡的参数化事实知识，而且验证准确性随上下文长度增加而下降。研究表明，上下文信息的置入位置对验证准确性有重要影响。
### Conclusion
研究结果强调了检索增强事实核查系统中提示结构的重要性。相关证据放置在提示的开始或结束处更有利于提高验证准确性，而放置在中段则效果较差。
## 23. `cs.CL` - 使用大规模语言模型进行临床阿尔茨海默病评估和诊断的链式推理 [PDF](https://arxiv.org/pdf/2602.13979), [HTML](https://arxiv.org/abs/2602.13979)
### Authors
Tongze Zhang,Jun-En Ding,Melik Ozolcer,Fang-Ming Hung,Albert Chih-Chieh Yang,Feng Liu,Yi-Rou Ji,Sang Won Bae
### Background
阿尔茨海默病(AZD)已成为全球范围内普遍存在的神经退行性疾病。传统的诊断方法仍然主要依赖医学影像和医生的临床评估，这在人手资源和医疗资源方面往往是耗时且耗费资源的。近年来，尽管大规模语言模型(LLMs)已在使用电子健康记录(EHRs)的医学领域得到广泛应用，但其在AZD评估中的应用仍受到限制，尤其是在AZD涉及复杂多因的病理情况下，影像学检查难以直接观察到这些因素。
### Innovation
本文提出了一种利用LLMs进行临床AZD评估并辅助诊断的新方法。不同于直接对LLMs进行EHR数据微调以进行AD分类，本方法利用LLMs生成的链式推理路径，提供模型明确的AD诊断推理解释，并结合结构化的链式推理进行预测。这种方法不仅增强了模型在诊断复杂因素方面的能力，还能提高诊断过程在不同AZD进展阶段的可解释性。实验结果表明，所提出的链式推理诊断框架能够显著提高稳定性与诊断性能，在多个CDR等级任务上实现了高达15%的F1分数提升。
### Conclusion
所提出的基于链式推理的大规模语言模型临床AZD评估框架不仅增强了模型的复杂因素诊断能力，还提高了不同AZD进展阶段预测过程的可解释性，相较于零样本基线方法，F1分数提高了15%。
## 24. `cs.CL` - LogitsCoder:通过Logits偏好解码实现高效链式思考路径搜索的代码生成 [PDF](https://arxiv.org/pdf/2602.14054), [HTML](https://arxiv.org/abs/2602.14054)
### Authors
Jizheng Chen,Weiming Zhang,Xinyi Dai,Weiwen Liu,Kounianhua Du,Yasheng Wang,Ruiming Tang,Yong Yu,Weinan Zhang
### Background
代码生成是一个具有挑战性的任务，需要准确且结构化的推理。现有的测试时间缩放（TTS）方法，包括结构化树搜索，已经在探索推理路径方面取得了进展，但仍面临两个主要挑战：(1) 深度不足的推理链，倾向于未能捕捉问题的全部复杂性；(2) 过度详细的推理导致效率降低和计算成本增加。为解决这些问题，本文提出了一种名为LogitsCoder的新型框架，通过轻量级、logit级别控制机制增强链式推理，以改善代码生成。
### Innovation
LogitsCoder通过Logits偏好解码引导token选择并将统计上更偏好的方式引导至推理路径，然后使用Logits排名基于路径选择和思想聚合选择和聚合多样化推理路径。这导致了更加连贯和有效的推理链，能够平衡深度和效率。实验表明，LogitsCoder生成的推理链更高效且质量更高，从而在代码生成性能上优于基线方法。
### Conclusion
 extensive experiments demonstrate that LogitsCoder produces more efficient and higher-quality reasoning chains, leading to superior code generation performance compared to baseline methods.
## 25. `cs.CL` - LM-Lexicon: 通过协调语义专家提高定义建模 [PDF](https://arxiv.org/pdf/2602.14060), [HTML](https://arxiv.org/abs/2602.14060)
### Authors
Yang Liu,Jiaye Yang,Weikang Li,Jiahui Liang,Yang Li,Lingyong Yan
### Background
该研究提出了一个创新的定义建模方法——LM-Lexicon，该方法结合了数据聚类、语义专家学习和模型合并。通过将定义建模任务分解为专门的语言子领域，在这些子领域中训练小型语言模型作为领域专家，相比之前最先进的方法，LM-Lexicon在五个广泛使用的基准上取得了显著改进（比之前最先进的模型提高了7%的BLEU分数）。
### Innovation
该研究创新地采用了稀疏混合专家架构，并进行数据聚类、语义专家学习和模型合并。将定义建模任务分解为专门的语言子领域，在这些子领域中训练小型语言模型作为领域专家。这种方法在基准测试中取得了显著的性能提升，具体表现为细粒度专家专业化提高了约10%的定义质量，语义感知领域定向机制相较于常规标记级别路由方式，提高了专家效率1%，以及通过测试时计算和语义专家扩展，还可以进一步获得性能提升。
### Conclusion
LM-Lexicon推动了定义建模的发展，为语义密集型应用的高效语言模型开发提供了宝贵的见解。
## 26. `cs.CL` - 几何保持聚合用于混合专家嵌入模型 [PDF](https://arxiv.org/pdf/2602.14039), [HTML](https://arxiv.org/abs/2602.14039)
### Authors
Sajjad Kachuee,Mohammad Sharifkhani
### Background
混合专家（MoE）嵌入模型通过加权线性求和结合专家输出，假设嵌入空间存在线性子空间结构。然而，这种假设与专家表示的几何结构不一致。通过对现代MoE嵌入模型的几何分析发现，专家输出位于由紧密集中范数和显著角度间隔定义的共享超球面流形上。在这种几何结构下，线性聚合会导致向流形内部的收敛，这会扭曲向量的大小和方向，降低嵌入的可比性。
### Innovation
引入了一种几何保持聚合运算（SBA，Spherical Barycentric Aggregation），该运算分离径向和角度成分，以保持超球面结构同时与现有路由机制完全兼容。实验结果表明，SBA 在相同的训练成本下实现了一致性能提升，并且具有全稳定性。
### Conclusion
对选定任务，包括语义相似性、聚类和重复问题检测，在Massive Text Embedding Benchmark（MTEB）上进行的额外几何分析确认了SBA防止聚合引起的退化并保持超球面的一致性，强调了MoE嵌入架构中几何感知聚合的重要性。
## 27. `cs.CL` - 从信息瓶颈视角探讨大语言模型自我解释的充分性与简洁性权衡 [PDF](https://arxiv.org/pdf/2602.14002), [HTML](https://arxiv.org/abs/2602.14002)
### Authors
Ali Zahedzadeh,Behnam Bahrak
### Background
大语言模型依赖于自我解释（如链式思考推理）来提高多步骤问题回答的性能。虽然这些解释提高了准确性，但也变得冗长且成本高昂，引发了必要解释量的多少值得探讨的问题。
### Innovation
本文基于信息瓶颈原理，提出了一种解释为压缩表示的观点，其只保留对于生成正确答案至关重要的信息。研究引入了一种评估管道，通过限制解释长度并使用多个语言模型在ARC挑战数据集中评估充分性。实验覆盖了英语和波斯语，结果显示更简洁的解释往往仍然具有充分性，能保持准确性的同时大幅减少解释长度，过度压缩则会导致性能退化。
### Conclusion
实验表明，更简洁的解释通常也是足够的，这在保持准确性的前提下大大减少了解释长度；然而，过度压缩会导致性能下降。
## 28. `cs.CL` - HLE-Verified：Humanity's Last Exam的系统验证与结构化修订 [PDF](https://arxiv.org/pdf/2602.13964), [HTML](https://arxiv.org/abs/2602.13964)
### Authors
Weiqi Zhai,Zhihai Wang,Jinghang Wang,Boyu Yang,Xiaogang Li,Xiang Xu,Bohan Wang,Peng Wang,Xingzhe Wu,Anfeng Li,Qiyuan Feng,Yuhao Zhou,Shoulin Han,Wenjie Luo,Yiyuan Li,Yaxuan Wang,Ruixian Luo,Guojie Lin,Peiyao Xiao,Chengliang Xu,Ben Wang,Zeyu Wang,Zichao Chen,Jianan Ye,Yijie Hu,Jialong Chen,Zongwen Shen,Yuliang Xu,An Yang,Bowen Yu,Dayiheng Liu,Junyang Lin,Hu Wei,Que Shen,Bing Zhao
### Background
Humanity's Last Exam (HLE) 作为评估前沿大语言模型在多领域挑战性问题上表现的基准测试工具已被广泛使用。然而，社区驱动的分析揭示了HLE中存在非平凡数量的噪声项目，这些噪声项目可能偏斜评估结果并扭曲跨模型比较。
### Innovation
该研究提出了HLE-Verified，这是一个经过验证和修订的HLE版本，具有透明的验证协议和细粒度的错误分类法。通过两阶段验证和修复工作流，第一阶段对每个项目进行二元验证，第二阶段进行严格的修订以保留原始评估意图，最终生成一个认证基准。HLE-Verified 在评估标准大语言模型时表现出了平均绝对准确性提高7-10个百分点，特别是在原始问题描述或参考答案错误的项目中，提高了30-40个百分点。
### Conclusion
HLE-Verified 改进了HLE风格的评估，通过减少标注噪声并使模型能力的测量更为忠实。
## 29. `cs.CL` - 使用NLP技术进行支付数据命名实体识别 [PDF](https://arxiv.org/pdf/2602.14009), [HTML](https://arxiv.org/abs/2602.14009)
### Authors
Srikumar Nayak
### Background
命名实体识别（NER）已经变成了自动化处理金融交易的关键组件，特别是在从不结构化的支付数据中抽取结构化信息时。本文对专门设计用于支付数据抽取的最新NER算法进行了全面分析，包括条件随机场（CRF）、双向长短时记忆网络与CRF相结合（BiLSTM-CRF）以及基于变换器的模型（如BERT和FinBERT）。研究人员在包含50,000条标注支付交易的跨多个支付格式（如SWIFT MT103、ISO 20022、本地支付系统）的数据集上进行了大量实验。
### Innovation
本文介绍了使用PaymentBERT，一种将特定领域金融嵌入与上下文表示结合的新型混合架构，其F1分数为95.7%，在保持实时处理能力的同时，超越了传统CRF基础方法12.8个百分点。此外，通过交叉格式泛化分析、消融研究和部署考虑，提供了详细的分析结果。
### Conclusion
本研究为金融机构实施自动化制裁筛查、反洗钱（AML）合规及支付处理系统提供了实用的见解。
## 30. `cs.CL` - 中世纪法语和拉丁文手稿自动转录的预编辑规范化 [PDF](https://arxiv.org/pdf/2602.13905), [HTML](https://arxiv.org/abs/2602.13905)
### Authors
Thibault Clérice,Rachel Bawden,Anthony Glaise,Ariane Pinche,David Smith
### Background
近年来，自动文本识别(ATR)技术的进步提高了对历史档案的访问，但古文字转录和标准化数字版本之间仍存在方法学上的差距。虽然针对古文字更优化的数据集训练的ATR模型显示出了更好的泛化能力，但它们的原始输出仍难以与大多数读者和下游NLP工具兼容，从而造成了使用上的差距。另一方面，生产标准化输出的ATR模型难以适应新领域，且倾向于过度规范化和幻想。PEN任务即根据编辑规范对图示化的ATR输出进行规范化，该方法在保持古文字忠实度的同时为实用性提供了标准化版本。
### Innovation
引入了预编辑规范化(PEN)任务，即根据编辑规范对图示化ATR输出进行规范化。作者创建了一个从CoMMA语料库派生的新数据集，该数据集与用passim对齐的数字化古法语和拉丁文版对齐。作者还生成了一个手动校正的黄金标准评估集。作者使用基于ByT5的序列到序列模型对该资源进行了基准测试，其中包括规范化的任务和预注释任务。贡献包括PEN的正式定义、包含4.66M样本的银训练语料库、1800样本的黄金评估集以及一个CER为6.7%的规范化模型，该模型显著优于先前的任务模型。
### Conclusion
作者基于ByT5的序列到序列模型在规范化任务和预注释任务上进行了基准测试，引入了预编辑规范化任务，开发了一个包含4.66M样本的银训练语料库和1800样本的黄金评估集，并提出了一个CER为6.7%的规范化模型，这个模型大大优于先前的任务模型。
## 31. `cs.LG` - 基于条件齐贯式检验马丁格尔的分布偏移检测 [PDF](https://arxiv.org/pdf/2602.13848), [HTML](https://arxiv.org/abs/2602.13848)
### Authors
Shalev Shaer,Yarin Bar,Drew Prinster,Yaniv Romano
### Background
现有CTM检测器通过不断增长参考集来构建检验马丁格尔，以评估新样本相对于过去观察的异常情况，这种方法能提供即时有效的I型错误控制，但会导致检测延迟增加和检测力降低。
### Innovation
提出了通过将每个新样本与固定参考数据集进行对比的顺序测试方法，避免检测时的污染。主要技术贡献在于构建鲁棒马丁格尔，通过明确考虑由有限参考集引起的参考分布估计误差，保持在零假设参考数据条件下的一次有效I型错误控制，并提供了渐近检测力为1和检测延迟有界保证。
### Conclusion
与标准CTMs相比，该方法能更快地检测分布偏移，提供强大的且可靠的分布偏移检测器。
## 32. `cs.LG` - 具有瞬时速度约束的均值流策略在一步动作生成中的应用 [PDF](https://arxiv.org/pdf/2602.13810), [HTML](https://arxiv.org/abs/2602.13810)
### Authors
Guojian Zhan,Letian Tao,Pengcheng Wang,Yixiao Wang,Yiheng Li,Yuxin Chen,Masayoshi Tomizuka,Shengbo Eben Li
### Background
在强化学习（RL）中，学习表达性强且高效的策略函数是一个有希望的方向。尽管基于流的策略在建模复杂动作分布方面表现良好，且具有快速确定性采样的过程，但它们仍然面临表达性和计算负担之间的权衡问题，通常通过流步骤的数量来控制。
### Innovation
本文提出了一种新的生成性策略函数——均值速度策略（MVP），用于实现最快的一步动作生成。通过在训练过程中引入瞬时速度约束（IVC），确保其高表达性。理论证明表明这种设计明确地起到了关键边界条件的作用，从而提高学习精度并增强策略的表达性。
### Conclusion
实验结果显示，MVP 在 Robomimic 和 OGBench 等多个具有挑战性的机器人操作任务中达到了最先进的成功率。同时，MVP 在训练和推理速度上也显著优于现有的基于流的策略基线。
## 33. `cs.LG` - MEMTS：通过参数化记忆内部化领域知识的时间序列基础模型无检索域适应 [PDF](https://arxiv.org/pdf/2602.13783), [HTML](https://arxiv.org/abs/2602.13783)
### Authors
Xiaoyun Yu,Li fan,Xiangfei Qiu,Nanqing Dong,Yonggui Huang,Honggang Qi,Geguang Pu,Wanli Ouyang,Xi Chen,Jilin Hu
### Background
虽然时间序列基础模型（TSFMs）在通用预测中表现出色，但在具有时间分布变化和领域特定周期结构的现实垂直领域中，其性能显著下降。现有的解决方案主要受到两种范式的限制：领域适应预训练（DAPT），可以改进短期领域适应，但会因灾难性遗忘频繁地破坏先前学习的整体时间模式；检索增强生成（RAG），引入外部知识但增加了大量的检索开销。这导致了严重的可扩展性瓶颈，无法满足实时流处理的高效率要求。
### Innovation
本文提出了Memory for Time Series (MEMTS)，一种轻量级的即插即用方法，用于无检索的时间序列领域适应。MEMTS的关键组件是知识保持模块（KPM），该模块将领域特定的时间动态（如周期性季节模式和趋势）内化为一组可学习的潜在原型，将片段的历史观测转化为连续的、可参数化的知识表示。这使MEMTS能够实现准确的领域适应，实时推理时间常数，接近零的延迟，同时有效缓解了对整体时间模式的灾难性遗忘，无需对冻结的TSFM主干进行任何架构修改。
### Conclusion
在多个数据集上的广泛实验表明，MEMTS的性能达到了最先进的水平（SOTA性能）。
## 34. `cs.LG` - AnomaMind：使用工具辅助推理的时间序列异常检测 [PDF](https://arxiv.org/pdf/2602.13807), [HTML](https://arxiv.org/abs/2602.13807)
### Authors
Xiaoyu Tao,Yuchong Wu,Mingyue Cheng,Ze Guo,Tian Gao
### Background
时间序列异常检测在许多实际应用中至关重要，要求有效方法能够定位异常区域并支持在复杂环境中进行可靠决策。然而，大多数现有方法将异常检测简化为固定特征输入的纯粹判别预测任务，而非证据驱动的诊断过程。这导致它们在处理具强上下文依赖性和多样模式的异常时表现出色度不足。
### Innovation
本文提出了一种名为AnomaMind的时间序列异常检测框架，它重新定义了异常检测为连续决策过程。AnomaMind通过逐步细化的方式定位异常区间，通过多轮工具交互实现自适应特征准备，通过自我反思改善异常决策。此外，AnomaMind设计了一种特定于任务的混合推理机制，用以增强工具辅助的异常检测。在通用模型负责自主工具交互和自我反思改进的同时，核心异常检测决策通过强化学习在可验证的工作流程级别反馈下学习，由此在灵活的推理框架中进行特定任务优化。
### Conclusion
在多种场景下进行的广泛实验表明，AnomaMind能够持续改进异常检测性能。相关代码托管于此 [链接]。
## 35. `cs.LG` - sleep2vec: 统一的跨模态对齐方法用于异质夜间生物信号 [PDF](https://arxiv.org/pdf/2602.13857), [HTML](https://arxiv.org/abs/2602.13857)
### Authors
Weixuan Yuan,Zengrui Jin,Yichen Wang,Donglin Xie,Ziyi Ye,Chao Zhang,Xuesong Chen
### Background
睡眠阶段划分到临床诊断等任务历来依赖于标准多导睡眠图（PSG）、床边监测和穿戴设备捕获的夜间多样生物信号（如EEG、EOG、ECG、SpO₂）。然而，不同设备的异质性以及传感器频繁掉线等问题使得这些多模态信号的统一建模成为一个显著挑战。
### Innovation
提出了sleep2vec，这一基础模型通过跨模态对齐学习共享表示。sleep2vec通过一种Demography、Age、Site & History意识的信息NCE目标进行对比预训练，该目标整合了生理和采集元数据（如年龄、性别、记录地点）以动态加权负样本并减轻特定人群的捷径。在睡眠阶段分类和临床结果评估等下游任务中，sleep2vec在任何可用模态的子集和传感器掉线情况下，均能稳定超越强大基线。此外，研究了夜间生物信号随模态多样性和模型容量变化的缩放定律。
### Conclusion
统一的跨模态对齐与合理的缩放策略相结合，使得在真实世界的夜间生物信号中实现高效、通用的标记模型成为可能。
## 36. `cs.LG` - Pawsterior: 变分流匹配方法的结构化仿真推断 [PDF](https://arxiv.org/pdf/2602.13813), [HTML](https://arxiv.org/abs/2602.13813)
### Authors
Jorge Carrasco-Pollo,Floor Eijkelboom,Jan-Willem van de Meent
### Background
许多仿真推断（SBI）问题涉及到受到结构化领域约束的后验，例如有界物理参数或混合离散-连续变量。标准的流匹配方法通常只在未加约束的空间中工作，这导致了学习效率低下，并且难以遵守物理约束。
### Innovation
在简化端点诱导的仿射几何限制的基础上，将结构几何直接融入推理过程，从而改善了采样期间的数值稳定性，提高了后验拟合准确性。更重要的是，通过其变分参数化方法，Pawsterior 能够处理传统流匹配方法无法处理的具有离散潜在结构的任务（如切换系统），从而扩展了流匹配方法的应用范围。
### Conclusion
Pawsterior 通过同时解决几何约束和离散潜在结构，将流匹配扩展到了先前无法解决的更广泛的结构化 SBI 问题类别中。
## 37. `cs.LG` - 基于物理驱动的快速未训练网络用于高度非线性逆散射问题 [PDF](https://arxiv.org/pdf/2602.13805), [HTML](https://arxiv.org/abs/2602.13805)
### Authors
Yutong Du,Zicheng Liu,Yi Huang,Bazargul Matkerim,Bo Qi,Yali Zong,Peixian Han
### Background
未训练神经网络（UNNs）在电磁逆散射重建中提供高精度，但由于高维空间域优化计算限制，速度相对较慢。
### Innovation
提出了一种基于物理驱动的快速傅里叶频谱（PDF）求解器，通过频域维度减少实现了亚秒级重构。通过使用截断的傅里叶基扩展感应电流，将优化限制在由散射测量支持的低频参数空间内。求解器整合了收缩积分方程（CIE）以减轻高对比度非线性问题，并加入了对比增强操作符（CCO）以校正频谱引起的衰减。此外，还提出了一个消除假界面损失以增强相邻散射器之间的边界锐度。
### Conclusion
实验和数值结果表明，与最先进的UNNs相比，该方法的速度提高了100倍，并且在噪声和天线不确定性下具有鲁棒性能，使实时微波成像应用成为可能。
## 38. `cs.LG` - Cast-R1：学习工具增强的序列决策策略以进行时间序列预测 [PDF](https://arxiv.org/pdf/2602.13802), [HTML](https://arxiv.org/abs/2602.13802)
### Authors
Xiaoyu Tao,Mingyue Cheng,Chuang Jiang,Tian Gao,Huanjian Zhang,Yaguo Liu
### Background
时间序列预测长期以来主要依赖于模型中心的方法，将预测视为从历史观察到未来值的一次映射过程。尽管最近取得了一些进展，但在复杂和不断变化的环境中，这些方法常常难以应对。主要原因在于大多数预测模型缺乏自主获取有用信息、推理潜在未来变化或通过迭代决策过程修订预测的能力。
### Innovation
本文提出了Cast-R1，这是一种学习时间序列预测框架，将预测重新定义为一种序列决策问题。Cast-R1引入了一种基于记忆的状态管理机制，可以在交互步骤中维持与决策相关的信息，从而支持长期推理中的上下文证据积累。在此基础上，预测是通过一个工具增强的代理工作流程实现的，在该工作流程中，代理自主与模块化的工具套件交互，提取统计特征，调用轻量级预测模型进行决策支持，进行基于推理的预测，并通过自我反思迭代改进预测。
### Conclusion
在多个真实世界时间序列数据集上的广泛实验表明，Cast-R1的有效性。希望本文的工作能为未来对代理范式进行时间序列建模的进一步探索提供实用的步骤。源代码可从提供。
## 39. `cs.LG` - MechPert: 功能机制共识作为预测未知扰动的诱导偏差 [PDF](https://arxiv.org/pdf/2602.13791), [HTML](https://arxiv.org/abs/2602.13791)
### Authors
Marc Boubnovski Martell,Josefa Lia Stoisser,Lawrence Phillips,Aditya Misra,Robert Kitchen,Jesper Ferkinghoff-Borg,Jialin Yu,Philip Torr,Kaspar Märten
### Background
对于理解基因调控和优先进行大规模扰动实验而言，预测未见遗传扰动的转录响应至关重要。现有方法要么依赖于静态的、可能不完整的知识图谱，要么直接利用语言模型获取功能相似基因之间的关联，这些关联主要基于科学文献中对称共现的逻辑而非定向调控逻辑。
### Innovation
我们提出了一个轻量级框架MechPert，鼓励L Lake码代理生成定向调控假设，而不是仅依赖功能相似性。通过多个代理独立提出带有置信评分的候选调控因子，并通过共识机制筛选虚假关联，生成用于后续预测的加权邻域。我们在四种人类细胞系的Perturb-seq基准测试上评估MechPert，在低数据量情况下（观测扰动数量为50），MechPert比基于相似性的基准提高了10.5%的皮尔逊相关系数；在实验设计中，MechPert选择的锚定点基因比标准网络中心性启发式算法在充分表征的细胞系中表现好46%。
### Conclusion
MechPert在预测高质量的低数据量下的转录响应和实验设计方面优于现有方法，特别是在基本表征的良好细胞系中更为出色。
## 40. `cs.LG` - 最小范数插值深度ReLU网络的稳定性充分条件 [PDF](https://arxiv.org/pdf/2602.13910), [HTML](https://arxiv.org/abs/2602.13910)
### Authors
Ouns El Harzli,Yoonsoo Nam,Ilja Kuzborskij,Bernardo Cuenca Grau,Ard A. Louis
### Background
算法稳定性是分析学习算法泛化误差的经典框架，它预测如果一个算法对训练集的小扰动（如删除或替换一个训练点）不敏感，则该算法具有小的泛化误差。尽管稳定性已经对许多知名算法进行了证明，但这种框架在深度神经网络的研究上应用有限。本文的研究对象是使用参数最小$ L_2 $范数实现训练误差为零的深度ReLU同构神经网络，即最小范数插值现象，在过参数化模型中通过梯度基础上的算法训练时可以看到这种现象。
### Innovation
本文研究了实现零训练误差的深度ReLU同构网络在最小范数插值条件下的稳定性充分条件。网络的稳定性取决于其子网络是否稳定以及后续层权重矩阵的秩属性。研究发现：1）当网络包含一个具有低秩权重矩阵的稳定子网络时，它可以保证稳定性；2）如果后续层不是低秩矩阵，即使存在稳定的子网络，也不能保证网络的稳定性。这种低秩假设源于近期的实验和理论结果，这些结果表明，深度神经网络在最小范数插值和权重衰减正则化条件下倾向于低秩权重矩阵。
### Conclusion
文章明确了实现零训练误差的深度ReLU同构网络在最小范数插值条件下的稳定性充分条件，指出其与子网络稳定性及后续层权重矩阵低秩性的关系。
