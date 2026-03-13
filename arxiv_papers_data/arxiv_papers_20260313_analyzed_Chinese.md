# 20260313
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 通过不精确概率口头化大语言模型的高阶不确定性 [PDF](https://arxiv.org/pdf/2603.10396), [HTML](https://arxiv.org/abs/2603.10396)
### Authors
Anita Yang,Krikamol Muandet,Michele Caprio,Siu Lun Chau,Masaki Adachi
### Background
尽管对大型语言模型（LLMs）中固有不确定性的需求日益增长，但实证证据表明，适用于古典概率不确定框架发展的提取技术往往无法充分捕获LLM的行为。这种不匹配导致了系统性的失败模式，特别是在涉及模糊问题回答、在线学习和自我反思的环境中。现有的技术无法有效处理这些复杂的情境。
### Innovation
该研究提出了基于不精确概率的新颖提示基础不确定提取技术，这是一种用于表示和提取更高阶不确定性的原则性框架。提出了一般提示和后处理程序来直接提取和量化两种类型的不确定性。该方法在不同场景中展现出了有效性和可靠性，使得大语言模型能够更真实地报告不确定性，从而提高决策的可信度。
### Conclusion
这一方法使大语言模型的不确定性报告更加忠实，提高了模型的可信度，支持下游决策过程。
## 2. `cs.AI` - 无奖励自我微调智能体实现自适应RAN切片控制 [PDF](https://arxiv.org/pdf/2603.10564), [HTML](https://arxiv.org/abs/2603.10564)
### Authors
Yuanhao Li,Haozhe Wang,Geyong Min,Nektarios Georgalas,Wang Miao
### Background
将生成式AI模型整合到AI原生网络系统中，为实现自主和适应性控制提供了变革性的途径。然而，将此类模型应用于连续控制任务时，由于固有的架构限制，如有限的上下文窗口、缺乏显式的奖励信号以及长时间上下文的退化，这一应用受到阻碍。
### Innovation
该论文提出了一种新颖的自我微调框架，使智能体能够通过直接与环境的交互来学习参数，并通过自动语言反馈生成的偏好数据集来构造偏好数据集，从而将长时经验提炼为模型参数。这一框架克服了传统强化学习和现有大型语言模型代理的缺陷，在样本效率、稳定性和多指标优化方面表现更优。
### Conclusion
实验结果表明，该框架在动态无线接入网络切片任务中优于标准的强化学习基线和现有的大型语言模型代理，展示了自改善生成智能体在连续控制任务中的潜力，为其未来AI原生网络基础设施铺平了道路。
## 3. `cs.AI` - LLM对齐真的需要多样性吗？RLVR方法适应道德推理的实证研究 [PDF](https://arxiv.org/pdf/2603.10588), [HTML](https://arxiv.org/abs/2603.10588)
### Authors
Zhaowei Zhang,Xiaohan Liu,Xuekai Zhu,Junchao Huang,Ceyao Zhang,Zhiyuan Feng,Yaodong Yang,Xiaoyuan Yi,Xing Xie
### Background
强化学习与验证奖励（RLVR）在逻辑推理任务上取得了显著成功，但大型语言模型（LLM）对齐是否需要根本不同的方法仍然不清楚。鉴于在道德推理中对多个有效响应的宽容性，自然假设对齐任务本质上需要寻求多样性的分布匹配算法，而不是奖励最大化的行为策略方法。研究团队首次在MoReBench上对这两种方法进行了全面比较，通过构建基于评判标准的奖励管道来训练Qwen3-1.7B评判模型，以稳定训练RLVR。
### Innovation
研究团队首次全面比较了RLVR方法和分布匹配方法在MoReBench上的表现。通过构建依据评判标准的奖励管道，开发了Qwen3-1.7B裁判模型，以确保RLVR训练的稳定性。研究证明，分布匹配方法在对齐任务上的表现并未如预期般优于奖励最大化方法。通过语义可视化，研究进一步揭示道德推理任务的高奖励分布比数学推理任务更为集中，多样化的解决方案策略也能获得相似的高奖励。
### Conclusion
研究结果表明，对齐任务并不需要保护多样性的算法，标准的奖励最大化RLVR方法可以在不需要明确的多样性机制的情况下有效转移到道德推理中。
## 4. `cs.AI` - HEAL：基于反事实熵辅助学习的推理提炼 [PDF](https://arxiv.org/pdf/2603.10359), [HTML](https://arxiv.org/abs/2603.10359)
### Authors
Wenjing Zhang,Jiangze Yan,Jieyun Huang,Yi Shen,Shuming Shi,Ping Chen,Ning Wang,Zhaoxiang Liu,Kai Wang,Shiguo Lian
### Background
从大型推理模型(LRMs)中蒸馏出推理能力并将其转移到较小模型通常受限于拒绝采样的限制。标准方法将教师视为静态过滤器，丢弃教师独立探索不到有效解决方案的复杂“边缘情况”问题，这人为地限制了学生的推理能力。
### Innovation
本文提出了基于反事实熵辅助学习(HEAL)框架，这是一种无需强化学习的框架，旨在弥合推理能力差距。HEAL结合了三个核心模块：指导下的熵辅助修复(GEAR)、 perplexity-不确定性比率估计器(PURE)和渐进式的答案指导课程演化(PACE)。
### Conclusion
在多种基准测试上的实验结果表明，HEAL显著优于传统的SFT蒸馏和其他基线。
## 5. `cs.AI` - 数据产品优化的代理控制中心 [PDF](https://arxiv.org/pdf/2603.10133), [HTML](https://arxiv.org/abs/2603.10133)
### Authors
Priyadarshini Tamilselvan,Gregory Bramble,Sola Shirai,Ken C. L. Wong,Faisal Chowdhury,Horst Samulowitz
### Background
数据产品能够帮助用户更好地洞悉其数据，例如通过提供例题-SQL配对或数据库表的视图。然而，生成有用的数据产品存在挑战，通常需要领域专家手动创建支持资产。
### Innovation
本文提出了一种系统，通过专门的AI代理在连续优化循环中自动化改善数据产品。通过呈现问题、监控多维质量指标并支持人机协作控制，该系统将数据转化为可观测并可调整的资产，实现自动化与信任监督的平衡。
### Conclusion
该系统不但提升了数据产品的质量和自动化水平，还增强了人工干预和监督，从而更好地平衡了自动化与信任之间的关系，提高了数据产品的可操作性和可信度。
## 6. `cs.AI` - 超越标量：通过几何进展和稳定性评估与理解LLM推理 [PDF](https://arxiv.org/pdf/2603.10384), [HTML](https://arxiv.org/abs/2603.10384)
### Authors
Xinyan Jiang,Ninghao Liu,Di Wang,Lijie Hu
### Background
使用标量概率来评估大规模语言模型（LLM）的可靠性往往无法捕捉到推理过程中的结构性动态。现有的评估方法侧重于单一数值指标，未能全面揭示模型在推理过程中的详细表现。
### Innovation
提出了一种名为TRACED的框架，通过理论支持的几何动力学评估推理质量。TRACED将推理痕迹分解为进展（位移）和稳定性（曲率），揭示了推理正确性和幻觉之间的显著拓扑差异，从而提供了一种物理视角来解码机器思考的内部动态。
### Conclusion
基于这些特征，概率框架在多种基准测试中展现出了竞争力和优越的鲁棒性。TRACED通过将高曲率映射为“犹豫回路”和位移映射为“确定性累积”，建立了几何学与认知之间的桥梁，为理解LLM的推理过程提供了新的视角。
## 7. `cs.AI` - IH-Challenge：改进前沿大模型指令层次结构的训练数据集 [PDF](https://arxiv.org/pdf/2603.10521), [HTML](https://arxiv.org/abs/2603.10521)
### Authors
Chuan Guo,Juan Felipe Ceron Uribe,Sicheng Zhu,Christopher A. Choquette-Choo,Steph Lin,Nikhil Kandpal,Milad Nasr, Rai (Michael Pokorny),Sam Toyer,Miles Wang,Yaodong Yu,Alex Beutel,Kai Xiao
### Background
指令层次（IH）定义了LLMs在冲突时如何优先处理系统、开发者、用户和工具的指令，为了解决指令冲突提供了具体的、可信的政策。IH对于防止出舱、系统提示提取和代理提示注入至关重要。然而，稳健的IH行为难以训练：IH失败可能与指令遵守失败混淆，冲突可能很微妙，模型可能会学习过度拒绝等捷径。
### Innovation
作者提出了IH-Challenge，一个强化学习训练数据集，以解决这些问题。通过在线生成对抗性示例对GPT-5-Mini进行微调，IH-Challenge在16个分布内、分布外和人类红队测试基准上平均改进了10.0%的IH鲁棒性，减少了有害行为从6.6%到0.7%，同时在一般安全性评估中提高了帮助性，内部静态代理提示注入评估饱和，并且几乎没有能力回退。
### Conclusion
作者发布了IH-Challenge数据集，以便支持未来关于稳健指令层次的研究。
## 8. `cs.AI` - 在资源受限环境下的结合大规模语言模型和图注意力机制的西洋棋决策框架 [PDF](https://arxiv.org/pdf/2603.10512), [HTML](https://arxiv.org/abs/2603.10512)
### Authors
Tianhao Qian,Zhuoxuan Li,Jinde Cao,Xinli Shi,Hanjie Liu,Leszek Rutkowski
### Background
人工智能通过游戏系统的发展得到了显著进步，这些系统为决策、战略规划和适应性学习提供了严格的测试平台。然而，资源受限环境带来了关键挑战，因为传统的深度学习方法依赖于大量的数据集和计算资源。在资源受限的环境下，开发高效的游戏AI系统具有挑战性。
### Innovation
本文提出了一个针对西洋棋Amazons游戏的轻量级混合框架，该框架通过结合基于图的学习的结构推理和大规模语言模型生成能力，探索从弱到强的泛化模式。具体来说，利用Graph Attention Autoencoder指导多步骤的Monte Carlo Tree Search，采用Stochastic Graph Genetic Algorithm优化评估信号，并利用GPT-4o-mini生成合成训练数据。与依赖专家示范的传统方法不同，该框架从嘈杂和不完美的监督中学习。实验结果表明，Graph Attention机制有效地作为结构过滤器，净化LLM的输出。在10×10的Amazons棋盘上，与基线相比，该混合方法在决策准确性上提高了15%—56%，并显著优于其老师模型GPT-4o-mini，在节点数N=30时达到45.0%的竞争力得胜率，在N=50时达到66.5%的高胜率。
### Conclusion
这些结果验证了在严格的计算约束下，从通用基础模型进化出特定的高性能游戏AI的可行性。
## 9. `cs.AI` - CUAAudit: 自监督视觉语言模型作为自主计算机使用代理审计员的元评估 [PDF](https://arxiv.org/pdf/2603.10577), [HTML](https://arxiv.org/abs/2603.10577)
### Authors
Marta Sumyk,Oleksandr Kosovan
### Background
计算机使用代理（CUAs）正在成为人机交互的一个新范式，能够通过感知高级自然语言指令在桌面上执行任务。随着这些代理变得越来越有能力并在不同的桌面环境中部署，评估其行为的可扩展性和可靠性成为了一个关键挑战。现有的评估管道依赖于静态基准、基于规则的成功检查或人工检查，这些方法脆弱、成本高且与实际使用情况不匹配。
### Innovation
本文研究了视觉语言模型（VLMs）作为自主审计员，直接从可观察的交互中评估CUA的任务完成情况。对五个VLMs进行了大规模的元评估，这些模型在给定自然语言指令和最终环境状态的情况下判断任务成功。评估覆盖了macOS、Windows和Linux环境下的三种常用CUA基准，从准确度、置信度估计的校准以及模型间的共识三个互补维度分析审计员的行为。
### Conclusion
结果表明，尽管最先进的VLMs在准确度和校准方面表现出色，但所有审计员在更复杂或异构环境中表现出显著的性能下降，即使是高表现的模型之间也存在显著的判断分歧。这些结果揭示了当前基于模型的审计方法的基本局限性，并强调了在真实环境中部署自主CUAs时需要明确考虑评估员的可靠性和不确定性的重要性。
## 10. `cs.AI` - GUI代理中的混合自演化结构记忆 [PDF](https://arxiv.org/pdf/2603.10291), [HTML](https://arxiv.org/abs/2603.10291)
### Authors
Sibo Zhu,Wenyi Wu,Kun Zhou,Stephen Wang,Biwei Huang
### Background
视觉-语言模型(VLMs)的发展使GUI代理能够以接近人类的方式与计算机交互，但在实际的计算机使用任务中，由于存在长周期的工作流程、多样化的界面以及频繁的中间错误，任务依然充满挑战。之前的工作通过构建大规模轨迹集合的外部记忆来装备代理，但这些方法依赖于平面检索和离散总结或连续嵌入的连续检索，无法提供人类记忆的结构化组织和自我进化特点。
### Innovation
受大脑的启发，本研究提出了一种图基记忆——Hybrid Self-evolving Structured Memory (HyMEM)。HyMEM结合了离散的高层符号节点和连续的轨迹嵌入，维护一个图结构以支持多跳检索、通过节点更新操作实现的自我进化以及推理期间的工作记忆刷新。
### Conclusion
在大量实验中，HyMEM持续改善了开源的GUI代理，使得7B/8B模型能够达到或超越强大的封闭源模型；特别地，它提高了Qwen2.5-VL-7B的性能+22.5%，并且超过了Gemini2.5-Pro-Vision和GPT-4o的表现。
## 11. `cs.CL` - BiasCause: 评估大型语言模型的社会偏见因果推理 [PDF](https://arxiv.org/pdf/2504.07997), [HTML](https://arxiv.org/abs/2504.07997)
### Authors
Tian Xie,Tongxin Yin,Vaishakh Keshava,Xueru Zhang,Siddhartha Reddy Jonnalagadda
### Background
随着大语言模型（LLMs）在社会中的作用日益显著，研究显示它们仍会生成反映对敏感群体的社会偏见的内容。现有基准能够有效识别这些偏见，但仍然缺乏理解产生这些偏见的根本推理过程的知识。本研究通过评估LLMs在回答社会偏见问题时的因果推理，来填补这一空白。
### Innovation
本研究提出了一个正式的分类方案，将因果推理分为三种类型（错误的、偏向的和基于上下文的），并通过合成涵盖八个敏感属性的1788个问题来探测特定类型的因果推理。研究发现，所有LLMs在这些问题上都表现出偏见的因果推理，甚至会犯‘错误的偏向’推理错误，这种错误是从相关性推及因果性随后应用偏向性的因果推理。通过对LLMs生成无偏因果推理案例的分析，研究还发现三种避免偏见的策略，为未来的去偏见努力提供了思路。
### Conclusion
本研究通过引入新的评估方法，揭示了LLMs偏见因果推理的主要机制，并提出了三种避免偏见的策略，为未来的去偏见提供了宝贵的经验和理论支持。”
## 12. `cs.CL` - TOSSS: CVE基线软件安全性基准用于大型语言模型 [PDF](https://arxiv.org/pdf/2603.10969), [HTML](https://arxiv.org/abs/2603.10969)
### Authors
Marc Damie,Murat Bilgehan Ertan,Domenico Essoussi,Angela Makhanu,Gaëtan Peter,Roos Wensveen
### Background
随着大型语言模型（LLMs）能力的增强，它们在许多行业中得到广泛应用，成为软件工程师的重要工具，支持各类开发任务。然而，随着LLMs在软件开发流程中的应用增多，人们开始关注LLMs在软件安全方面的表现。由于全球组织都大幅投入于网络安全领域，以减少受到破坏性攻击的暴露程度，将LLMs集成到软件工程流程中可能会引入新的漏洞，削弱现有的安全措施。因此，需要评估LLMs在选取安全性代码方面的表现。
### Innovation
本文提出了TOSSS（Two-Option Secure Snippet Selection），这是一个基于CVE数据库的基准测试框架，旨在评估LLMs选择安全代码片段的能力。TOSSS不同于现有的其他安全基准测试，它依赖于CVE数据库，并提供了一个可扩展的框架，可以随着时间逐步整合新披露的漏洞。每个模型的得分范围为0到1，分数越高表示模型选择安全代码片段的能力越强。本文评估了14种广泛使用的开源和闭源的LLM模型在C/C++和Java代码上的表现，分数区间从0.48到0.89。此外，由于LLMs提供商已经发布了他们的模型基准测试得分，TOSSS有望成为这些报告中的一个补充的、聚焦于安全的因素。
### Conclusion
TOSSS是一个测评LLMs安全性能的基准，它提供了比现有基准更全面的疫苗氰评价，有助于提高软件开发中代码安全性的自动评估能力。通过使用TOSSS，可以更好地理解LLMs在软件工程与网络安全之间的平衡点，从而指导未来的研究和应用发展方向。
## 13. `cs.CL` - Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning [PDF](https://arxiv.org/pdf/2502.01968), [HTML](https://arxiv.org/abs/2502.01968)
### Authors
Jinlong Pang,Na Di,Zhaowei Zhu,Jiaheng Wei,Hao Cheng,Chen Qian,Yang Liu
### Background
研究表明，在大型语言模型（LLMs）的监督微调（SFT）过程中，数据质量比数量更重要。现有的大部分数据清洗方法主要集中在过滤整个样本，但样本内的单个词质量可能差异很大。即使在高质量的样本中，也不排除存在与任务无关、冗余、无信息或有害的模式或短语。继续对这些模式进行微调可能提供有限的收益，甚至可能损害下游任务性能。
### Innovation
本文从噪声标签的角度调查了token质量，并提出了一个适用于SFT任务的通用token清洗流程。该方法过滤掉了无信息的token并保留了包含关键任务相关信息的token。具体而言，首先通过检查模型更新对每个token的影响来评估token质量，然后使用基于阈值的分离方法。token影响可以用固定参考模型的一次通过测量，也可以用自进化参考模型逐步迭代测量。通过误差上界理论分析了两种方法的优点和局限性。广泛的实验表明，该框架能够一致地改善下游性能。代码可以在该网址获取。
### Conclusion
我们的框架一致地提高了下游性能。在单一通过和自进化参考模型的两种方法中，研究了它们优缺点的理论误差上界。实验结果证明，我们的方法能够有效提高SFT任务的性能。
## 14. `cs.CL` - EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation [PDF](https://arxiv.org/pdf/2410.21271), [HTML](https://arxiv.org/abs/2410.21271)
### Authors
Shih-Yang Liu,Maksim Khadkevich,Nai Chit Fung,Charbel Sakr,Chao-Han Huck Yang,Chien-Yi Wang,Saurav Muralidharan,Hongxu Yin,Kwang-Ting Cheng,Jan Kautz,Yu-Chiang Frank Wang,Pavlo Molchanov,Min-Hung Chen
### Background
后训练压缩技术虽然可以有效减少大型语言模型（LLMs）的内存占用、延迟和功耗，但仍会导致明显的准确率下降，并受限于硬件和内核限制，支持的压缩格式有限，这限制了模型在各种部署场景中的灵活性。
### Innovation
本文提出了EoRA（Eigenvalue Space Low-Rank Approximation），这是一种无需微调的新方法，通过与压缩LLMs结合低秩矩阵，允许用户迅速提高特定任务的性能，并自由平衡准确性和计算开销之间的权衡，超越压缩格式的限制。EoRA在恢复压缩LLMs的准确率方面始终优于之前的训练免费低秩方法，尤其是在LLaMA3-8B压缩至3比特的情况下取得了显著的准确率提升（例如：ARC-Challenge达到10.84%，MathQA达到6.74%，GSM8K达到11.45%）。此外，EoRA还引入了优化的CUDA内核，使推理加速多达1.4倍，并通过量化EoRA减少了内存开销。
### Conclusion
EoRA为不同用户需求下的压缩模型准确性提升提供了一种便捷解决方案，使LLMs的部署更加高效和灵活。
## 15. `cs.CL` - 使用大规模语言模型建模语言 [PDF](https://arxiv.org/pdf/2404.09579), [HTML](https://arxiv.org/abs/2404.09579)
### Authors
Jumbly Grindrod
### Background
本文指出，语言学研究不应仅仅关注语言认知过程背后的机制，还应将其视为一种外部的社会实体。文章认为大规模语言模型可以在这种背景下作为科学模型发挥作用。
### Innovation
文章借鉴Weisberg (2007) 的模型构念概念，提出可以利用最近在计算语言学方面的研究成果，进一步理解大规模语言模型的内部运作机制，并将其作为一种语言模型来使用。
### Conclusion
文章认为，即便有反对意见声称这些语言模型并不能提供有价值的语言学洞察，但通过上述创新方法，可以更好地利用这些模型来构建语言模型的科学模型价值。
## 16. `cs.CL` - COMIC：自动生成草稿喜剧 [PDF](https://arxiv.org/pdf/2603.11048), [HTML](https://arxiv.org/abs/2603.11048)
### Authors
Susung Hong,Brian Curless,Ira Kemelmacher-Shlizerman,Steve Seitz
### Background
该论文提出了一种全自动化的人工智能系统，能够生成类似于《周六夜现场》等喜剧秀的短视频。系统从角色参考开始，运用一系列基于真实制作工作室角色的代理，通过迭代的竞争、评估和改进来优化质量和多元性的想法和输出。
### Innovation
该系统的一个关键贡献是引入了与现实观众偏好相契合的语言模型批评者，通过分析YouTube上喜剧视频的语料库，实现自动评价幽默效果。实验结果显示，该框架生成的结果接近专业制作草稿的品质，同时展示了在视频生成方面的先进性能。
### Conclusion
该框架展示了生成高质量喜剧视频的能力，具有竞争优势，为自动化喜剧生成领域带来了创新突破。
## 17. `cs.CL` - 通过伪对话注入攻击大型语言模型以实施目标劫持 [PDF](https://arxiv.org/pdf/2410.23678), [HTML](https://arxiv.org/abs/2410.23678)
### Authors
Zheng Chen,Buhui Yao
### Background
背景：目标劫持是一种针对大型语言模型（LLMs）的对抗性攻击，攻击者的目的是操纵模型产生特定的预设输出，而不考虑用户原始的输入。在目标劫持中，攻击者通常会在用户的提示后面添加一个精心构造的恶意后缀，以迫使模型忽略用户的原始输入并生成目标响应。
### Innovation
创新：本文提出了一种新颖的目标劫持攻击方法——伪对话注入，这种方法利用了LLMs在对话情境下角色识别的弱点。具体来说，通过伪造对用户初始提示的响应，然后提出一个恶意的新任务提示，使模型将初始提示和伪造的响应视为完成的对话，从而执行新的虚假提示。提出的三种伪对话构建策略分别针对不同场景：目标化的伪对话、通用化的伪对话和稳健的伪对话。
### Conclusion
结论：本实验在两个主流的大规模语言模型平台ChatGPT和Qwen上进行，结果显示，本文提出的方法在攻击有效性方面显著优于现有方法。
## 18. `cs.CL` - ThinkPatterns-21k：大规模语言模型中思考模式影响的系统研究 [PDF](https://arxiv.org/pdf/2503.12918), [HTML](https://arxiv.org/abs/2503.12918)
### Authors
Pengcheng Wen,Jiaming Ji,Chi-Min Chan,Juntao Dai,Donghai Hong,Yaodong Yang,Sirui Han,Yike Guo
### Background
大型语言模型（LLMs）在采用‘思考然后回答’的范式下表现出增强性能，即模型在生成最终响应前先生成内部思考（类似人类的System 2思考模式）。然而，现有研究缺乏对不同思考方式如何影响不同模型规模性能的系统性理解。
### Innovation
本文首次引入ThinkPatterns-21k数据集，该数据集包含21,000个由五个不同结构化及非结构化思考模式增强的指令-回应对。详细评估了不同规模模型（从3B到32B参数）对这些思考模式的反应，并发现了不同规模模型对思考模式的不同响应。
### Conclusion
研究发现，小规模模型（<30B参数）大部分可以受益于结构化思考模式，而大规模模型（32B参数）对某些结构化思考模式（如分解）可能会导致性能下降。相比之下，非结构化思考模式（独白）在不同规模模型中展现出更广泛的有效性。此外，研究发布了数据集、训练结果以及其他多元思考模式以增强研究的可重复性，旨在推动该领域的进一步研究。
## 19. `cs.CL` - $V_{0.5}$：稀疏强化学习采样下的先验通用价值模型 [PDF](https://arxiv.org/pdf/2603.10848), [HTML](https://arxiv.org/abs/2603.10848)
### Authors
Yi-Kai Zhang,Yueqing Sun,Hongyan Hao,Qi Gu,Xunliang Cai,De-Chuan Zhan,Han-Jia Ye
### Background
在基于强化学习的奖励验证（RLVR）中，构建一个稳健的优势基线对于策略梯度方法至关重要，能够有效引导策略模型增强所需行为。最新的研究引入了通用价值模型（如$V_0$），通过在上下文中显式编码模型能力进行预训练，从而可以消除与策略模型同步更新价值模型的需要。
### Innovation
本文提出了$V_{0.5}$，它通过将这种价值模型预测的基线（作为先验）与稀疏采样中获得的经验平均值动态融合，构建了一个既考虑计算效率又具有极低方差的稳健基线。具体来说，引入了实时统计测试和动态预算分配机制，以平衡稀疏采样带来的高方差与价值模型先验固有的系统偏差（或幻觉）。通过实时构造假设检验来评估先验的可靠性，并根据需要动态分配额外的采样预算。
### Conclusion
在涵盖六个数学推理基准测试中的广泛评估表明，$V_{0.5}$ 显著优于 GRPO 和 DAPO，实现了更快的收敛速度并提高了超过 10% 的性能。
## 20. `cs.CL` - LLMs中伪相关反馈的系统性研究 [PDF](https://arxiv.org/pdf/2603.11008), [HTML](https://arxiv.org/abs/2603.11008)
### Authors
Nour Jedidi,Jimmy Lin
### Background
研究表明，基于大规模语言模型（LLMs）的伪相关反馈（PRF）方法沿两大关键设计维度构建：反馈来源和反馈模型。反馈来源指代反馈文本的来源，而反馈模型是指使用给定的反馈文本如何来优化查询表示。然而，这两个维度中的每一个在措施有效性方面的作用仍不清楚，而且通常在实际评估中被纠缠在一起。论文通过系统性研究不同反馈来源和反馈模型的选择如何影响PRF的有效性来解决这个问题。
### Innovation
论文通过控制实验对不同反馈来源和反馈模型的选择如何影响PRF的有效性进行了系统性研究。研究覆盖了13个低资源BEIR任务和五种LLM PRF方法，结果显示：1. 反馈模型的选择对PRF的有效性起着关键作用；2. 仅从LLM生成的文本中提取的反馈提供了最具成本效益的解决方案；3. 在使用强首要检索器提取候选文档时，从语料库中提取的反馈最具益处。这些发现为理解PRF设计空间中哪些元素最为重要提供了更好的理解。
### Conclusion
该研究得出了以下结论：反馈模型的选择在PRF的有效性方面起着重要作用，从LLM生成的文本中提取的反馈是最具成本效益的解决方案，从语料库中提取的反馈在使用强首要检索器时最为有益。这些发现提高了对PRF设计空间的重要因素的理解。
