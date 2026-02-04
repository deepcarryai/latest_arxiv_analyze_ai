# 20260204
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 基于本地化和纠正错误的LLM规划器 [PDF](https://arxiv.org/pdf/2602.00276), [HTML](https://arxiv.org/abs/2602.00276)
### Authors
Aditya Kumar,William W. Cohen
### Background
大型语言模型（LLMs）在数学和编码推理方面表现出色，但在符号经典规划任务上经常失败。研究表明，由LLM生成的计划经常违背指令中的领域约束（例如，穿过墙壁）。目前，LLM生成的计划处理结果中普遍存在违反领域约束的问题。
### Innovation
本文提出了一种迭代理论局部上下文学习（L-ICL）方法，通过特定示例纠正具体失败步骤的错误，以此局部化和纠正LLM规划器生成的错误。相比传统的完整问题解决轨迹注入或显式指令以及其它基线方法，L-ICL技术更有效。例如，在8x8网格世界中，使用L-ICL可以使生成的有效计划比例从59%提高到89%，仅需60个训练样本。这种方法在外其他领域（网格世界导航、迷宫、Sokoban、积木世界）也都显示了显著改善。
### Conclusion
L-ICL方法在改善LLM生成计划正确性方面表现优异，特别是在有限的训练样本范围内，在不同领域和不同的LLM架构下均显示出明显的性能提升，有效地解决了现有方法中LLM规划器频繁违反领域约束的问题。
## 2. `cs.AI` - 评估窄调优引发领域级新兴不对齐的脆弱性 [PDF](https://arxiv.org/pdf/2602.00298), [HTML](https://arxiv.org/abs/2602.00298)
### Authors
Abhishek Mishra,Mugilan Arulvanan,Reshma Ashok,Polina Petrova,Deepesh Suranjandass,Donnie Winkelmann
### Background
随着语言模型被越来越多地用于自主任务，新兴不对齐（Emergent Misalignment）带来的风险变得越来越明显。该研究通过分析在不同领域进行调优的语言模型，评估它们在未触发和触发后门触发器情况下的表现，探讨了窄调优与领域脆弱性之间的关系。
### Innovation
本研究首次提供了按领域对新兴不对齐的分类和排名，利用这一分类对AI安全性及后训练过程具有重要意义。同时，研究还标准化了一种构建不对齐数据集的方法，并揭示了成员推理指标在预测模型广泛不对齐程度方面的作用。
### Conclusion
窄调优会导致广泛不对齐的风险增加，特定领域的脆弱性也有所不同。通过对不同数据集调优后的模型进行研究，发现可以从一个新兴不对齐模型中提取的方向有时可以推广到指导其他模型的行为。
## 3. `cs.AI` - 学习定价：动态市场中的可解释属性级模型 [PDF](https://arxiv.org/pdf/2602.00188), [HTML](https://arxiv.org/abs/2602.00188)
### Authors
Srividhya Sethuraman,Chandrashekar Lakshminarayanan
### Background
在高维市场的动态定价中，面临的主要挑战包括可扩展性、不确定性以及解释性不足。现有的低秩结合算法可以高效学习，但依赖于潜在特征，无法清晰展示单个产品属性如何影响价格。因此需要一种能够更好地解释产品属性对价格影响的模型。
### Innovation
提出了一种可解释的添加特征分解基于低维需求（Additive Feature Decomposition-based Low-Dimensional Demand, AFDLD）模型，以及基于此结构的ADEPT（Additive DEcomposition for Pricing with cross-elasticity and Time-adaptive learning）算法。该算法能够在属性空间直接操作，并实现了近似于亚线性遗憾的$tilde{text{O}}(text{d}^{1/2} text{T}^{3/4})$，并且该模型能够更快速地适应市场冲击和漂移，同时提供了透明的属性级价格解释。
### Conclusion
通过控制合成研究和实际数据集，该研究展示了ADEPT模型在动态市场条件下能够学习接近最优的价格，快速适应市场冲击和漂移，并能够提供透明的、基于属性的价格解释。这项工作表明，通过结构化的、以属性驱动的表示，可以同时实现自主定价代理的效率和解释性。
## 4. `cs.AI` - SayNext-Bench: Why Do LLMs Struggle with Next-Utterance Prediction? [PDF](https://arxiv.org/pdf/2602.00327), [HTML](https://arxiv.org/abs/2602.00327)
### Authors
Yueyi Yang,Haotian Liu,Fang Kang,Mengqi Zhang,Zheng Lian,Hao Tang,Haoyu Chen
### Background
尽管近年来大规模语言模型（LLMs）在自然对话方面取得了显著进步，能够与用户进行交互，但研究显示，即使是领先模型也难以预测人类讲话者的下一句话。相比之下，人类可以通过从上下文中的多种模态线索（如手势、眼神接触和情感语气）中预测即将出现的对话内容。本文旨在系统性地探讨LLMs能否复制这种能力。
### Innovation
本文提出了SayNext-Bench，这是一种评估LLMs和多模态LLMs（MLLMs）在根据多种模态线索预测上下文条件下的回应能力的标准。为了支持此基准，还构建了SayNext-PC，一个包含丰富模态线索的大型对话数据集。在此基础上，进一步开发了一种以认知启发设计为基础的双路径预测MLLM，SayNext-Chat。实验结果表明，该模型在词汇重叠、语义相似性和情感一致性等方面的性能优于最先进的MLLMs。
### Conclusion
该研究结果显示，从多种模态线索预测下一个对话单元在LLMs上是可行的，并强调了模态线索和主动预测性处理在自然人类交互中的不可或缺作用，这是当前MLLMs所缺乏的。这些发现为构建更加拟人、上下文敏感的人工智能交互提供了新的研究途径，这对于以人为中心的人工智能具有重要意义。基准测试和模型可以在此处找到：this https URL。
## 5. `cs.AI` - 从游戏轨迹到游戏机制：大规模语言模型的因果归纳 [PDF](https://arxiv.org/pdf/2602.00190), [HTML](https://arxiv.org/abs/2602.00190)
### Authors
Mohit Jiwatode,Alexander Dockhorn,Bodo Rosenhahn
### Background
深度学习代理能够在复杂的游戏领域实现高性能，但通常并不理解游戏背后的因果机制。为了克服这一局限，本文研究了因果归纳的能力，即通过大型语言模型（LLMs）从游戏玩法记录中推断出视频游戏描述语言（VGDL）规则。
### Innovation
本文选择了GVGAI框架下的九个代表性游戏，利用语义嵌入和聚类技术来减少冗余。对比了直接从观察中生成VGDL代码和分两步生成：首先推断结构因果模型（SCM），然后将该模型翻译为VGDL的方法。通过多种提示策略和控制上下文环境进行评估，结果显示基于SCM的方法生成的VGDL更接近真实情况，盲评估的胜率最高达81%，且产生了更少的逻辑不一致规则。
### Conclusion
所学的SCM可用于因果强化学习、可解释代理以及生成新的但逻辑上一致的游戏等下游应用场合。
## 6. `cs.AI` - 通过多值逻辑完全识别深度ReLU神经网络 [PDF](https://arxiv.org/pdf/2602.00266), [HTML](https://arxiv.org/abs/2602.00266)
### Authors
Yani Zhang,Helmut Bölcskei
### Background
研究显示，深层ReLU神经网络允许非平凡的功能对称性，即实现相同功能可以使用差异极大的架构和参数（权重和偏置）。本文探讨了给定函数f，如何推导出能够产生f的所有全连接ReLU神经网络的架构和参数的问题。
### Innovation
将ReLU神经网络转化为Lukasiewicz逻辑公式，通过逻辑公理调控的代数重写实现功能等价的网络转换。提出了一种组合范式，以简化从Lukasiewicz逻辑公式到ReLU网络的映射过程。基于Chang的完备性定理，表明每一功能等价类中所有的ReLU网络都可以由有限组对应的Lukasiewicz逻辑公理所表征。
### Conclusion
所有功能等价类中的ReLU网络可以通过有限组等同对连接起来，这些等同对对应Lukasiewicz逻辑公理的有限集合。这一方法类比于Shannon的经典成果，他在开关电路设计中通过布尔公式表示和逻辑公理驱动的代数重写实现了综合设计。
## 7. `cs.AI` - 观点：自主演进是演进LLM的道路 [PDF](https://arxiv.org/pdf/2602.00359), [HTML](https://arxiv.org/abs/2602.00359)
### Authors
Minhua Lin,Hanqing Lu,Zhan Shi,Bing He,Rui Mao,Zhiwei Zhang,Zongyu Wu,Xianfeng Tang,Hui Liu,Zhenwei Dai,Xiang Zhang,Suhang Wang,Benoit Dumoulin,Jian Pei
### Background
随着大型语言模型（LLMs）从受控训练集过渡到开放的现实世界环境，一个根本性局限逐渐显现：静态训练无法跟上部署环境的持续变化。仅仅扩大训练时和推理时时的计算规模来提高静态能力并不能解决这一训练-部署差距。
### Innovation
本文提出了“自主演进”的概念，认为现有的部署时适应方法（如参数微调或启发式记忆积累）缺乏足够的策略自主性来诊断失败并产生持久改进。本文认为，自主演进代表了LLM适应的必然未来，将进化本身从固定流水线提升为自主进化代理。为此，本文提出了一种名为A-Evolve的通用框架，将部署时改进视为对持续系统状态的目标导向优化过程。此外，还提出了进化扩展假设，即分配给自主进化的计算能力越大，适应能力越强，确立了自主进化作为人类环境内长期、持续适应路径的可扩展途径。
### Conclusion
本文通过提出A-Evolve通用框架，将部署时改进视为目标导向的优化过程，并提出了进化扩展假设，认为自主进化是解决LLMs进化过程中训练-部署差距的一种有效途径，可以支持LLMs在实际环境中的长期持续适应。
## 8. `cs.AI` - 使用元代理的自主数据处理 [PDF](https://arxiv.org/pdf/2602.00307), [HTML](https://arxiv.org/abs/2602.00307)
### Authors
Udayan Khurana
### Background
传统的数据处理管道通常是静止和手工定制的，适用于特定任务，限制了它们对变化需求的适应性。尽管通用代理和编程助手可以生成理解良好的数据管道的代码，但它们无法自主监控、管理和优化在部署后的端到端管道。
### Innovation
提出了自主数据处理使用元代理（ADP-MA），这是一个框架，通过分层代理编排动态构建、执行和迭代优化数据处理管道。ADP-MA 强调上下文感知优化、自适应工作负载分割和渐进采样以实现可扩展性。框架利用多种外部工具并可以重复使用先前设计的代理，减少了冗余并加快了管道构建。
### Conclusion
通过交互式演示展示了 ADP-MA，演示包括管道构建、执行监控和适应性改进，贯穿代表性数据处理任务。
## 9. `cs.AI` - 在医疗保健中可扩展且安全的AI推断：FastAPI和NVIDIA Triton Inference Server在Kubernetes上的比较基准分析 [PDF](https://arxiv.org/pdf/2602.00053), [HTML](https://arxiv.org/abs/2602.00053)
### Authors
Ratul Ali
### Background
现代生产环境中高效且可扩展地部署机器学习（ML）模型是必不可少的，尤其是在如医疗保健和制药等受监管的领域。这些领域的系统必须在多个要求之间取得平衡，包括最小化用于实时临床决策支持的推理延迟、最大化批量处理医学记录的吞吐量，以及遵守严格的数据隐私标准如HIPAA。本文通过比较两种主流部署模式来对这些需求进行分析：一种是基于Python的轻量级REST服务FastAPI，另一种是高性能的推理服务器NVIDIA Triton Inference Server。
### Innovation
使用医疗保健AI的参考架构，在Kubernetes上部署DistilBERT情感分析模型，以便在受控实验条件下测量中位数（p50）和尾部（p95）延迟以及吞吐量。结果表明，它们存在权衡。虽然FastAPI针对单请求工作负载提供了较低的延迟（22毫秒），但Triton通过动态批量处理实现了更好的可扩展性，单个NVIDIA T4 GPU上的吞吐量达到了每秒780个请求，几乎是基准线的两倍。此外，研究还评估了一种混合架构的方法，利用FastAPI作为保护健康信息去识别的安全网关，而将Triton用于后端推理。
### Conclusion
研究表明，该混合模型是企业临床AI的最佳实践。它为安全的、高可用性的部署提供了蓝图。
## 10. `cs.AI` - MHDash: 一种基准测试心理健康意识人工智能助手的在线平台 [PDF](https://arxiv.org/pdf/2602.00353), [HTML](https://arxiv.org/abs/2602.00353)
### Authors
Yihe Zhang,Cheyenne N Mohawk,Kaiying Han,Vijay Srinivas Tida,Manyu Li,Xiali Hei
### Background
大型语言模型（LLMs）在心理健康支持系统中的应用日益广泛，准确识别高风险状态如自杀念头和自伤是至关重要的。然而，现有的评估主要依赖于聚合性能指标，这些指标往往掩盖了特定风险失败模式，对真实场景中多回合对话中的模型行为提供了有限的洞察。
### Innovation
MHDash 是一个开源平台，旨在支持心理健康应用中的人工智能系统的开发、评估和审计。它将数据收集、结构化标注、多轮对话生成和基准评估集成在一个统一的管道中。该平台支持从多个维度进行标注，包括关注类型、风险等级和对话意图，从而实现了细致和风险感知的分析。
### Conclusion
我们的结果揭示了几项关键发现：(i) 简单基准和先进的LLM API在总体准确性上相当，但在高风险案例上表现出显著差异；(ii) 一些LLM在保持一致的顺序严重性排名时在绝对风险分类上失败，而其他则在总体得分合理但严重类别上出现高假阴性率；(iii) 在多回合对话中，性能差距放大，因为风险信号逐渐浮现。这些观察表明，传统的基准测试对心理健康关键领域来说是不够的。通过发布MHDash作为开放平台，我们旨在促进可重复的研究、透明的评估和定向安全的人工智能系统开发。
## 11. `cs.AI` - 按需事实度：Text生成中的事实度-信息量权衡控制 [PDF](https://arxiv.org/pdf/2602.00848), [HTML](https://arxiv.org/abs/2602.00848)
### Authors
Ziwei Gong,Yanda Chen,Julia Hirschberg,Chen Zhao,He He,Zhou Yu,Kathleen Mckeown
### Background
大型语言模型（LLMs）在回答查询时存在一个固有的权衡：它们可以生成不太有信息量但高度准确的答案，或者生成更有信息量但可能不太准确的回答。不同的应用需要在这个信息量和准确性之间的不同平衡。现有的研究和应用大多没有提供一个有效的方式让用户能够灵活地指定这些平衡。
### Innovation
本文提出了一种名为Factuality-Controlled Generation (FCG)的框架，允许用户在查询时指定事实度约束。通过训练模型生成既符合事实度要求又保持有信息量的输出，论文提出使用合成数据训练模型，并展示了这种方法能够显著提高模型的性能。
### Conclusion
本文通过研究FCG框架，提出了一种新的方法来控制Text生成中的事实度信息量权衡。通过合成数据训练模型，FCG能够满足用户的特定事实度要求并保持生成的文本有较高的信息量。
## 12. `cs.AI` - 基于 unary 算术的矩阵乘法单元在低精度 DL 加速器中的探索 [PDF](https://arxiv.org/pdf/2602.00838), [HTML](https://arxiv.org/abs/2602.00838)
### Authors
Prabhu Vellaisamy,Harideep Nair,Di Wu,Shawn Blanton,John Paul Shen
### Background
通用矩阵乘法(GEMM)是深度学习(DL)中的一个基本操作。随着DL向低精度迁移，近期工作提出了作为传统二进制GEMM硬件替代的新型一元GEMM设计。为了评估一元硬件在未来DL计算中的潜力，需要对这些设计进行严格的评估。
### Innovation
本文聚焦于基于整数的DL推理中的一元GEMM设计，详细评估了三种最新的设计提案：uGEMM、tuGEMM和tubGEMM，并与传统的二元GEMM进行了比较。此外，通过多种比特宽度和矩阵尺寸进行了仔细的后综合评估，分析了这些设计的权衡并确定了最佳的性能区域。同时，进行了权重稀疏性分析，涉及了8个预先训练的卷积神经网络(CNNs)以及大型语言模型LLaMA2。
### Conclusion
本文展示了在一元GEMM如何在未来的边缘AI加速器中实现高效的能源计算。
## 13. `cs.AI` - Don't Forget Its Variance! The Minimum Path Variance Principle for Accurate and Stable Score-Based Density Ratio Estimation [PDF](https://arxiv.org/pdf/2602.00834), [HTML](https://arxiv.org/abs/2602.00834)
### Authors
Wei Chen,Jiacheng Li,Shigui Li,Zhiqi Lin,Junmei Yang,John Paisley,Delu Zeng
### Background
分位数方法已经成为了密度比率估计（DRE）的一种强大框架，但是在实际应用中，这些方法的性能与其理论上路径无关性之间存在矛盾。尽管理论上是路径无关的，但其实现效果却高度依赖于所选择的路径计划。
### Innovation
该研究提出了最小路径方差（MinPV）原则，通过引入一种原则化启发式方法来最小化被忽视的路径方差。研究人员推导出路径方差的闭式表达式，将其从非可行问题转变为可行的优化问题。该方法通过参数化路径为灵活的Kumarswamy混合模型，学习到自适应的数据路径，从而减少了路径方差。这种方法优化了完整目标函数，得到了更准确和稳定的估计器，建立了新的基准状态。
### Conclusion
该方法通过优化整体目标，提供了更准确和稳定的密度比率估计，解决了传统方法依赖于路径因素的瓶颈，并且在具有挑战性的基准测试中实现了新的最优结果。
## 14. `cs.AI` - 通过经济博弈获取大语言模型的可信度先验 [PDF](https://arxiv.org/pdf/2602.00769), [HTML](https://arxiv.org/abs/2602.00769)
### Authors
Siyu Yan,Lusha Zhu,Jian-Qiao Zhu
### Background
构建以人为本、可信赖的人工智能系统的关键方面之一是维持校准的信任：适当的依赖人工智能系统优于过度信任（如自动化偏差）或不足信任（如弃用）。然而，一个根本性的挑战是如何量化和描述人工智能系统自身所表现出的信任水平。
### Innovation
该研究提出了一种基于迭代上下文学习的新颖提取方法（Zhu 和 Griffiths, 2024a），并将其应用于通过行为博弈论中的信任博弈来提取可信度先验。信任博弈将信任定义为基于对另一代理信念的自愿风险暴露，而不是自我报告的态度。该方法用于从多个领先的大型语言模型（LLMs）中获取可信度先验，并发现 GPT-4.1 的可信度先验与人类观察到的相似。研究进一步探讨了 GPT-4.1 在信任博弈中对不同玩家角色的反应，这是对这些模型如何根据不同代理的特点区分信任进行初步刻画。此外，研究表明，通过感知温暖和能力来构建的刻板印象模型很好地预测了提取的可信度变异。
### Conclusion
研究结果表明，通过使用一种新的提取方法，成功地从大语言模型中获取了可信度先验，并通过行为博弈的使用更好地理解了这些模型的行为和反应模式，预测了可信度水平的变异。
## 15. `cs.AI` - 控制蛋白质语言模型中的重复 [PDF](https://arxiv.org/pdf/2602.00782), [HTML](https://arxiv.org/abs/2602.00782)
### Authors
Jiahao Zhang,Zeqing Zhang,Di Wang,Lijie Hu
### Background
蛋白质语言模型（PLMs）在结构预测和从头设计蛋白质方面取得了进展，但它们在生成过程中经常会陷入病理性的重复。这种重复在蛋白质中会破坏结构的稳定性，影响功能的可实现性。目前，尚无系统地研究PLMs中的这种复制问题。
### Innovation
本文首次对PLMs中的复制问题进行了系统研究，提出了量化重复度量来表征基序级和同聚体重复，并展示了它们对折叠可靠性的负面影响。提出了UCCS（Utility-Controlled Contrastive Steering）方法，通过受约束的数据集引导蛋白质生成。这种方法在不损害折叠性能的前提下降低重复，且在推断时注入效果显著。
### Conclusion
实验结果表明，与解码惩罚和其他基线相比，本文方法显著降低了重复，同时保持了AlphaFold的信心分数。这些结果确立了重复控制是PLMs的核心挑战，并强调了以数据集为指导的引导作为可靠蛋白质生成的合理方法。
## 16. `cs.AI` - 使用因果先验的多目标多精度贝叶斯优化 [PDF](https://arxiv.org/pdf/2602.00788), [HTML](https://arxiv.org/abs/2602.00788)
### Authors
Md Abir Hossen,Mohammad Ali Javidian,Vignesh Narayanan,Jason M. O'Kane,Pooyan Jamshidi
### Background
多精度贝叶斯优化（MFBO）通过结合低成本的低精度近似来加速搜索黑盒函数的全局最优解。现有的MFBO方法主要捕捉输入、精度和目标之间的关联性依赖，而未能充分捕捉因果机制。这在低精度近似与目标精度不一致时可能导致表现不佳。
### Innovation
文章提出了RESCUE（在因果理解和估计中降低采样成本）方法，这是一种多目标MFBO方法，结合了因果推理来系统地解决上述挑战。RESCUE建立了一个结构因果模型，捕捉输入、精度和目标之间的因果关系，并使用此模型构建编码干预效应的概率多精度（MF）近似。RESCUE还引入了一种因果三维体积的知识梯度获取策略，该策略利用因果结构来选择平衡多目标改进期望和成本的输入-精度对。
### Conclusion
研究结果表明，与最先进的MF优化方法相比，RESCUE在合成问题和机器人、机器学习（AutoML）及医疗等领域的真实世界问题上能够提高样本效率。
## 17. `cs.AI` - RMFlow: 噪声注入步骤改进的高效多模态生成方法 [PDF](https://arxiv.org/pdf/2602.00849), [HTML](https://arxiv.org/abs/2602.00849)
### Authors
Yuhao Huang,Shih-Hsin Wang,Andrea L. Bertozzi,Bao Wang
### Background
MeanFlow能够高效生成高保真图像，但由于其单功能评估（1-NFE）生成往往无法产生令人信服的结果。RMFlow在此基础上进行改进，通过将粗糙的1-NFE MeanFlow传输与后续定制的噪声注入细化步骤相结合，解决这个问题。
### Innovation
RMFlow引入了一种新的噪声注入步骤，结合了MeanFlow的高效传输和定制的噪声注入细化步骤，通过神经网络训练了一个新的损失函数，该函数平衡了最小化概率路径之间的Wasserstein距离和最大化样本似然性。RMFlow在文本到图像、上下文到分子和时间序列生成上都取得了接近最新技术水平的结果，仅需1-NFE，且计算成本与基本MeanFlow相当。
### Conclusion
RMFlow通过噪声注入步骤的加入，使模型能够实现高效的多模态生成，同时保持高保真度和计算成本的合理性。
## 18. `cs.AI` - Latent Shadows: 隐影—— masked 扩散中的高斯-离散对偶性 [PDF](https://arxiv.org/pdf/2602.00792), [HTML](https://arxiv.org/abs/2602.00792)
### Authors
Guinan Chen,Xunpeng Huang,Ying Sun,Shijin Wang,Yanyong Zhang,Chao Wang
### Background
掩码离散扩散已成为高质量语言建模的主要范式，其中令牌在迭代中被损坏为掩码状态。然而，其推理效率受到缺乏确定性采样工具的瓶颈。尽管扩散对偶性可以用于均匀模型的确定性蒸馏，但这些方法通常在性能上逊于掩码模型，并依赖于复杂的积分算子。相比之下，在掩码领域，先前的方法通常假设不存在确定性轨迹，因此不得不依赖随机蒸馏方法。
### Innovation
文章通过建立掩码扩散对偶的显式关系，证明了掩码过程可通过一种新颖的最大值索引保持机制从连续高斯过程中投影产生。此外，引入了掩码一致性蒸馏（MCD），这是一个先验框架，利用这种对偶性来构建所需的确定性耦合轨迹进行一致性蒸馏，从而避免使用数值ODE求解器。这种方法严格改进了前随机蒸馏方法，实现了16倍的推理速度提升，同时不牺牲生成质量。
### Conclusion
本文不仅为掩码和连续扩散之间的联系提供了坚实的理论基础，还释放了确认蒸馏在离散高精度生成中潜在的全部性能。
## 19. `cs.AI` - JTok: 通过联合Token自调制作为另一个缩放轴的Token嵌入 [PDF](https://arxiv.org/pdf/2602.00800), [HTML](https://arxiv.org/abs/2602.00800)
### Authors
Yebin Yang,Huaijin Wu,Fu Guo,Lin Yao,Xiaohan Qin,Jingzhi Wang,Debing Zhang,Junchi Yan
### Background
LLMs通常沿着密集维度扩展，性能与近线性的计算成本增加相关联。虽然MoE将容量与计算解耦合，但它引入了巨大的内存开销和硬件效率挑战。为了解决这些问题，本文提出了一种新的、正交的扩展轴——标记索引参数，以解耦模型容量和FLOPs。具体来说，本文引入了Joint-Token (JTok)和Mixture of Joint-Token (JTok-M)，它们通过从辅助嵌入表中检索调制向量来增强Transformer层。这些向量通过轻量级的元素级操作对主干进行调制，不会产生显著的FLOPs开销。广泛的实验表明，这种方法在不同体量的模型中（从650M（190M + 460M嵌入）到61B（17B + 44B嵌入）），能够持续减少验证损失并显著提高下游任务性能（例如：MMLU +4.1，ARC +8.3，CEval +8.9）。进一步的等价FLOPs分析证实，JTok-M从根本上改变了质量-计算帕累托前沿，相对原始MoE架构实现35%的计算节约的同时保持了同等模型质量，我们验证了标记索引参数具有可预测的幂律缩放行为。
### Innovation
提出了一种新的、正交的扩展轴——标记索引参数，通过从辅助嵌入表中检索调制向量来增强Transformer层，使模型容量与FLOPs解耦合。这种方法包括Joint-Token (JTok)和Mixture of Joint-Token (JTok-M)，能够通过轻量级的元素级操作对主干进行调制，不会产生显著的FLOPs开销。广泛的实验表明，这种方法能够持续减少验证损失，显著提高下游任务性能，并通过减少计算成本来实现同等模型质量。
### Conclusion
本文提出的方法在不同体量的模型中持续减少了验证损失并显著提高了下游任务性能，进一步的等价FLOPs分析证实，这种方法从根本上改变了质量-计算帕累托前沿，实现了显著的计算节约。此外，标记索引参数表现出可预测的幂律缩放行为，高效实现确保了引入的开销始终保持在边缘状态。
## 20. `cs.AI` - BLOCK-EM: 阻止因果特征以防止出现意外对齐 [PDF](https://arxiv.org/pdf/2602.00767), [HTML](https://arxiv.org/abs/2602.00767)
### Authors
Muhammed Ustaomeroglu,Guannan Qu
### Background
当语言模型因其定窄目标的监督目标而微调时，可能会出现意外对齐：模型不仅学会目标行为，也可能发展出一些不希望的特定域外行为。研究发现通过识别那些可靠地控制意外行为的内部特征，并阻止模型在微调期间强化这些特征，可以采取机制性方法来防止意外对齐。
### Innovation
研究提出了一种机制性方法，通过识别能够可靠控制不希望的特定域外行为的内部特征，然后在微调期间阻止模型加强这些特征，以此来预防意外对齐。该方法在六个微调领域中实现了高达95%的相对减少不希望的特定域外行为，且未对模型质量或目标任务性能造成任何负面影响。该方法还通过不同的分离选择/评估划分、多个独立的评估者、关键设置的多重随机种子、质量度量指标以及广泛的消融分析（ablation studies）来增强实证的严肃性，证明了意外对齐减少与所识别机制的相关性。
### Conclusion
我们的结果表明，对特定微调期间内部机制采取有针对性的训练约束能够缓解意外对齐，而不会损害目标任务性能。同时研究还揭示了微调时间过长可能导致意外对齐重新出现的现象，并通过改进建议部分恢复了阻断对齐的效果。这一研究展示了在保持目标任务性能的情况下，针对内部机制施加训练时间限制可以缓解意外对齐问题。
## 21. `cs.CL` - D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use [PDF](https://arxiv.org/pdf/2602.02160), [HTML](https://arxiv.org/abs/2602.02160)
### Authors
Bowen Xu,Shaoyu Wu,Hao Jiang,Kai Liu,Xin Chen,Lulu Hu,Bin Yang
### Background
大型推理模型（LRMs）在应对复杂现实世界问题时，有效的工具使用和推理能力是至关重要的。当前的研究表明，这些模型在复杂工具使用场景中的子任务分解能力不足，导致了‘懒惰推理’的现象。
### Innovation
本文提出了一种两阶段训练框架D-CORE，该框架首先通过自我精炼激励LRM的子任务分解推理能力，随后通过多样化意识强化学习恢复其反思性推理能力。D-CORE在多个基准测试和不同模型规模下实现了稳健的工具使用改进，并在BFCLv3实验中证明了方法的优越性：D-CORE-8B的准确率达到77.7%，超过最佳8B模型5.7%；D-CORE-14B在14B规模下达到了79.3%的准确率，优于70B模型，尽管D-CORE-14B的规模只有后者的一五分之一。
### Conclusion
本文提出的D-CORE框架通过激励LRM的子任务分解和推理能力，在复杂工具使用场景中实现了显著的性能提升，尤其在BFCLv3实验中表现突出，验证了D-CORE的有效性和优越性。
## 22. `cs.CL` - 针对特定领域RAG系统的人工智能评估：AgriHubi案例研究 [PDF](https://arxiv.org/pdf/2602.02208), [HTML](https://arxiv.org/abs/2602.02208)
### Authors
Md. Toufique Hasan,Ayman Asad Khan,Mika Saari,Vaishnavi Bankhele,Pekka Abrahamsson
### Background
大型语言模型在知识密集型领域有潜力，但它们在农业领域的应用受到依存关系薄弱、英语训练数据的中心性以及有限的实际评估的影响。这些问题在低资源语言中尤为突出，尽管这些语言拥有高质量的领域文档，但获取这些文档对于通用模型来说仍然困难。
### Innovation
本文介绍了AgriHubi，这是一个为芬兰语农业决策支持设计的领域适应检索增强生成(RAG)系统。AgriHubi结合了芬兰农业文档与开放式的PORO家族模型，并通过显式源 grounding 和用户反馈支持迭代改进。该系统在八次迭代后通过两次用户研究进行了评估，显示出在答案完整性、语言准确性以及感知可靠性方面的明显提升。
### Conclusion
该研究提供了一种在低资源语言环境下设计和评估特定领域RAG系统的经验指南，并揭示了部署较大模型时响应质量与延迟之间的实际权衡。
## 23. `cs.CL` - 大型语言模型的拒绝不仅仅是单一方向的问题 [PDF](https://arxiv.org/pdf/2602.02132), [HTML](https://arxiv.org/abs/2602.02132)
### Authors
Faaiz Joad,Majd Hawasly,Sabri Boughorbel,Nadir Durrani,Husrev Taha Sencar
### Background
先前的研究认为，大型语言模型的拒绝行为可以通过单一的激活空间方向来调节，从而实现有效的引导和去除效果。这项研究反驳了这一观点。
### Innovation
研究人员发现拒绝行为在激活空间中对应着几何上不同的方向，尽管有多样性，但对不同拒绝相关方向的线性调节会产生几乎相同的拒绝-过度拒绝 trade-off，表明这是一个共享的一维控制旋钮。不同方向的主要影响不在于模型是否拒绝，而在于如何拒绝。
### Conclusion
研究发现大型语言模型的拒绝行为不仅仅是单一方向的问题，而是存在多个几何上不同的方向，这些方向通过线性调节共同影响模型的拒绝行为。
## 24. `cs.CL` - 斯里兰卡语言（僧伽罗语）物理常识推理数据集 [PDF](https://arxiv.org/pdf/2602.02207), [HTML](https://arxiv.org/abs/2602.02207)
### Authors
Nisansa de Silva,Surangika Ranathunga
### Background
该论文介绍了作为全球PIQA项目一部分创建的第一个僧伽罗语物理常识推理数据集。背景信息提到，之前可能存在其他语言的类似数据集，但这是首个针对僧伽罗语开发的数据集，为该语言的机器理解提供了新的数据支持。
### Innovation
创新之处在于它创建了首个僧伽罗语的物理常识推理数据集，这些数据集包含110个人工创建和验证的数据样本，每个样本包括一个提示、正确的答案和一个错误的答案。大多数问题都与斯里兰卡的背景有关，突显该数据集在特定文化环境中的应用价值。
### Conclusion
结论显示，该数据集为僧伽罗语机器理解的研究提供了坚实的基础，特别是在物理常识推理领域。通过对这些数据的训练，可以提升机器理解和回答有关僧伽罗语背景下的物理常识问题的能力。
## 25. `cs.CL` - Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge [PDF](https://arxiv.org/pdf/2602.02219), [HTML](https://arxiv.org/abs/2602.02219)
### Authors
Yuzheng Xu,Tosho Hirasawa,Tadashi Kozuno,Yoshitaka Ushiku
### Background
大型语言模型（LLMs）现在广泛用于评估文本质量，这种方法常被称为LLM作为评判者。尽管先前的研究主要集中在点级别的和成对的评估方法上，基于评分准则的评估（评分准则中LLMs从多个评分准则中选择一个分数）却较少受到分析。本文揭示了基于评分准则的LLM作为评判者中存在隐含的多选项设置，这导致位置偏见：LLMs更倾向于选择评分准则列表中特定位置的分数。
### Innovation
本文指出，基于评分准则的评估隐含地像是一个选择题设置，因此存在位置偏见。通过跨多个模型和数据集的受控实验，作者展示了这种位置偏见的一致性。提出了一种平衡的排列策略，均匀地分布在评分选项的位置上，以减少这种偏差。通过聚合平衡排列下的得分，不仅揭示了隐藏的位置偏见，还提高了LLM作为评判者与人类的关联性。研究表明，基于评分准则的LLM作为评判者并非固有的点级别或成对的，简单的排列校准可以显著提高其可靠性。
### Conclusion
本文表明，基于评分准则的LLM作为评判者并非固有的点级别或成对的。简单的位置排列校准可以显著提高其可靠性。
## 26. `cs.CL` - 超越内存障碍：一种针对具百万令牌上下文的高效训练系统 [PDF](https://arxiv.org/pdf/2602.02108), [HTML](https://arxiv.org/abs/2602.02108)
### Authors
Wenhao Li,Daohai Yu,Gen Luo,Yuxin Zhang,Fei Chao,Rongrong Ji,Yifan Wu,Jiaxin Liu,Ziyang Gong,Zimu Liao
### Background
训练大规模语言模型（LLMs）在长上下文上的训练受到显卡内存开销的严重限制，而非训练时间。主要原因是激活量的内存足迹与序列长度成线性关系。现有方法对此无有效的改进措施。
### Innovation
提出了OOMB（超越内存障碍），一种极其内存高效的大模型训练系统。OMMB引入了一种分块递归的训练框架，并在计算时动态重计算激活量，从而保持激活量的内存足迹恒定（O(1)）并使主要瓶颈转移到不断增长的键值缓存。为了管理键值缓存，OMMB结合了多种协同优化技术：为键值缓存及其梯度设计分页内存管理器、异步CPU卸载以隐藏数据传输延迟以及分页稀疏注意机制以减少计算复杂性和通信开销。
### Conclusion
实验证据表明，每增加10K令牌上下文，Qwen2.5-7B的端到端训练内存开销仅增加10MB。这使得在单个H200 GPU上训练Qwen2.5-7B达到4M令牌上下文的目标成为可能，而没有使用上下文并行的大规模集群才能实现。这项工作在长上下文语言模型训练资源效率方面取得了显著进展。
## 27. `cs.CL` - 在世界各地的语言中评估大型语言模型的元语言知识 [PDF](https://arxiv.org/pdf/2602.02182), [HTML](https://arxiv.org/abs/2602.02182)
### Authors
Tjaša Arčon(1),Matej Klemen(1),Marko Robnik-Šikonja(1),Kaja Dobrovoljc(1, 2, 3) ((1) University of Ljubljana, Faculty of Computer and Information Science, Slovenia (2) University of Ljubljana, Faculty of Arts, Slovenia, (3) Jožef Stefan Institute, Ljubljana, Slovenia)
### Background
现有的大型语言模型（LLMs）通常在语言使用任务上进行评估，但对其对语言结构的认知能力了解有限。现有的语言学基准测试主要关注狭窄的语言现象，侧重于资源丰富的语言，并且很少评估元语言知识——显式地思考语言结构而非语言使用。这项研究表明，在当前的LLMs中，元语言知识是有限的。GPT-4o的准确率最高，但也只有中等水平（0.367），而开源模型表现较差。所有模型都能超越随机猜测，但未能超过主要类别基准，这表明他们捕获了跨语言模式，但在细微的语法区分方面缺乏能力。在语言领域，准确率与数字语言状态有强烈相关性：拥有更高数字存在和资源可用性的语言在评估中得分更高，而资源匮乏的语言则表现出明显较低的性能。数据分析表明，与地理、谱系或社会语言学因素相比，资源相关的指标（如维基百科规模、语料库可用性）是更有力的准确率预测因子。因此，LLMs的元语言知识是零碎的，由数据可用性而非跨世界语言的一般化语法能力所塑造。
### Innovation
本研究使用准确率和宏F1，以及大多数类别和随机基线分析整体性能，并探讨了不同语言领域的变异。研究发现，LLMs的元语言知识是有限的，GPT-4o的准确率最高，但仍只有中等水平（0.367），而开源模型表现较差。所有模型均能超越随机猜测，但未能超过大多数类别基准，表明它们捕获了跨语言模式，但在细微的语法规则区分上还不够精细。性能在不同语言领域中有所不同，词汇特征表现出最高的准确度，而音系特征则最低，部分反映了在线可见度的差异。在语言层面，准确率与数字语言状态有很强的关联：数字存在感更高、资源充足的语言被更准确地评估，而资源匮乏的语言则表现出明显较低的性能。
### Conclusion
这些结果表明，LLMs的元语言知识是碎片化的，受数据可用性的驱动，而不是世界上各种语言一般的语法能力。为了支持系统的评估并鼓励未来LLMs拥有更广泛的全球语言多样性，我们发布了基准测试作为开源数据集。
## 28. `cs.CL` - Focus-dLLM: 通过置信引导的上下文聚焦加速长上下文扩散大规模语言模型推理 [PDF](https://arxiv.org/pdf/2602.02159), [HTML](https://arxiv.org/abs/2602.02159)
### Authors
Lingkun Long,Yushi Huang,Shihao Bai,Ruihao Gong,Jun Zhang,Ao Zhou,Jianlei Yang
### Background
差分大规模语言模型(dLLMs)能够提供强大的长上下文处理能力，并采用非自回归解码范式。然而，双向全注意机制的计算成本限制了推理效率。虽然稀疏注意机制很有前景，但现有方法仍然效率低下。问题根源在于需要估计尚未解码的标记的重要性，而在扩散过程中未知的未遮掩标记位置使得这一估计变得困难。
### Innovation
本文提出了一种名为Focus-dLLM的新型无训练稀疏化框架，专门针对准确高效的长上下文dLLM推理。通过发现相邻步骤标记置信度高度相关这一发现，该框架利用过去置信度指导的指示器预测未遮掩区域。在此基础上，通过有导向的剪枝策略准确地估计和移除冗余的注意力计算，同时保留影响力较高的注意力接收器。为了进一步减少开销，该策略利用跨层的一致性，重复识别的接收器位置在各层中进行复用。
### Conclusion
实验结果表明，在32K上下文长度下，我们的方法在无损失的情况下提供了超过29倍的加速。代码已公开，可在指定网址查看。
## 29. `cs.CL` - 在统一多模态模型内部理解与生成之间的差距量化 [PDF](https://arxiv.org/pdf/2602.02140), [HTML](https://arxiv.org/abs/2602.02140)
### Authors
Chenlong Wang,Yuhang Chen,Zhihan Hu,Dongping Chen,Wenhu Chen,Sarah Wiegreffe,Tianyi Zhou
### Background
近期一体化多模态模型（UMM）在理解和生成任务上已经取得了显著的进步，但这两个能力是否真正集成于单一模型中尚未明确。为了探讨这个问题，该文提出了GapEval，这是一个双向基准，用于量化理解与生成之间的差距，并定量测量两个方向的认知一致性。通过不同架构的UMM实验，揭示了两个方向之间存在持续的差距，表明当前模型实现的只是浅层次的统一而非深层次的认知合并。
### Innovation
引入了GapEval基准，该基准不仅用于量化统一多模态模型理解与生成之间的差距，还用于测量两个方向的认知一致性。此外，通过知识操作的实证研究，探讨了背后的原因，并揭示了知识在多模态模块中的不一致性和不同模态间知识同步的问题。
### Conclusion
实验证明，当前的UMM在理解和生成能力上存在差距，且知识在不同模态间的同步性较差，这表明它们仍处于浅层次的统一。进一步的实验证实了知识在统一多模态模型中的不一致性和不同模态间知识同步的问题。因此，未来的研究需要探索深层次的认知整合机制。
## 30. `cs.CL` - AR-MAP: 自回归大型语言模型是扩散大型语言模型的隐式教师？ [PDF](https://arxiv.org/pdf/2602.02178), [HTML](https://arxiv.org/abs/2602.02178)
### Authors
Liang Lin,Feng Xiong,Zengbin Wang,Kun Wang,Junhao Dong,Xuecai Hu,Yong Wang,Xiangxiang Chu
### Background
扩散大型语言模型（DLLMs）已成为自回归模型的强大替代方案，能够实现多位置的并行标记生成。然而，DLLMs在偏好对齐方面仍然具有挑战，这主要是因为基于证据下界（ELBO）的概率估计引入了高方差。
### Innovation
提出了一种新的迁移学习框架AR-MAP，通过利用偏好对齐的自回归大型语言模型（AR-LLMs）作为隐式教师来对DLLMs进行偏好对齐。该框架利用了这两种不同生成范式的共享架构结构，通过简单的权重缩放使DLLMs吸收AR-LLMs的对齐知识。AR-MAP避开了直接对DLLMs进行对齐的高方差和计算复杂度，并且在多种偏好对齐任务中表现出了竞争力或优越性，跨所有任务和模型平均得分为69.08%。
### Conclusion
实验结果表明，AR-MAP方法在多种偏好对齐任务中达到了竞争或优越的性能，达到了所有任务和模型平均得69.08%的分数。该代码可以在提供的链接处获得。
## 31. `cs.CV` - Learnable Total Variation with Lambda Mapping for Low-Dose CT Denoising [PDF](https://arxiv.org/pdf/2511.10500), [HTML](https://arxiv.org/abs/2511.10500)
### Authors
Yusuf Talha Basak,Mehmet Ozan Unal,Metin Ertas,Isa Yildirim
### Background
尽管Total Variation (TV)在噪声抑制和边缘保持方面表现出色，但它对标量正则化参数的依赖限制了其适应性。
### Innovation
提出了一种Learnable Total Variation (LTV)框架，该框架将展开的TV解算器与LambdaNet相结合，后者可以预测每个像素的正则化图。该框架通过端到端训练来联合优化重建和正则化，从而实现空间自适应平滑。
### Conclusion
在DeepLesion数据集上使用现实生活中的LoDoPaB-CT模拟实验表明，LTV在经典TV和FBP+U-Net上表现出一致的改进，实现了多达3.7 dB PSNR和8%相对SSIM的改进。LTV提供了低剂量CT去噪的一种可解释的替代方案，取代了黑盒CNN。
## 32. `cs.CV` - 超越余弦相似度：感知质量感知的Box-Cox变换CLIP无参考图像质量评估 [PDF](https://arxiv.org/pdf/2511.09948), [HTML](https://arxiv.org/abs/2511.09948)
### Authors
Zhicheng Liao,Dongxu Wu,Zhenshan Shi,Sijie Mai,Hanwei Zhu,Lingyu Zhu,Yuncheng Jiang,Baoliang Chen
### Background
近年来，研究人员重新利用了Contrastive Language-Image Pre-training (CLIP) 模型来实现无参考图像质量评估（No-Reference Image Quality Assessment，NR-IQA）。主要通过计算图像嵌入和文本提示（比如“一张好的照片”或“一张糟糕的照片”）之间的余弦相似度来实现。尽管这种方法能够捕捉到一定的语义相似性，但未能充分利用CLIP图像特征的幅度，而该幅度与感知质量之间存在明显的相关性。
### Innovation
本文引入了一种新颖的自适应融合框架，该框架结合了余弦相似度和幅度感知的质量提示。具体而言，作者首先提取了绝对CLIP图像特征，并采用Box-Cox变换对特征分布进行统计归一化，从而减少语义敏感性。这种方法产生的标量总结为辅助质量提示，增强基于余弦的提示匹配。为有效结合这两种提示，作者设计了一个基于置信度的自适应融合方案，根据各自的相对强度对每个项进行加权。
### Conclusion
在多个基准IQA数据集上进行的大量实验表明，本文方法在无参考图像质量评估任务上显著优于标准的CLIP方法和最先进的基准，且无需任何特定任务的训练。
## 33. `cs.CV` - UrbanIng-V2X：跨多个交叉路口的多车辆和多基础设施大型合作感知数据集 [PDF](https://arxiv.org/pdf/2510.23478), [HTML](https://arxiv.org/abs/2510.23478)
### Authors
Karthikeyan Chandra Sekaran,Markus Geisler,Dominik Rößle,Adithya Mohan,Daniel Cremers,Wolfgang Utschick,Michael Botsch,Werner Huber,Torsten Schön
### Background
近年来，合作感知数据集在推进智能移动应用方面发挥了关键作用，通过智能代理之间的信息交换，克服遮挡等挑战并提高整体场景理解。虽然一些现有的真实世界数据集包含了车辆到车辆和车辆到基础设施的交互，但这些数据集通常仅限于一个十字路口或单个车辆。目前缺乏一个覆盖多个交叉路口且由多辆联网车辆和基础设施传感器支持的大型感知数据集，影响了在各种交通环境中的算法基准测试。结果导致过度拟合，模型可能会由于相似的交叉口布局和交通参与者行为而表现出虚假的高性能。
### Innovation
本文介绍了UrbanIng-V2X，这是第一个支持涉及德国Ingolstadt市三个交叉路口安装的车辆和基础设施传感器的大规模多模态合作感知数据集。UrbanIng-V2X 包含34个时序对齐且空间校准的传感器序列，每个序列持续20秒，涵盖了两个车辆和最多三个基础设施安装的传感器杆在协调场景中的记录。UrbanIng-V2X提供了来自12个车辆安装的RGB相机、2个车辆LiDAR、17个基础设施热成像相机和12个基础设施LiDAR的详尽数据。所有序列以每秒10赫兹的频率标注，覆盖13个类别，总共约有712k注释实例。研究使用最先进的合作感知方法进行了全面评估，并公开了代码库、数据集、高清地图和完整的数据采集环境的数字孪生。
### Conclusion
UrbanIng-V2X为研究提供了丰富的合作感知数据，有助于在复杂的交通环境中对算法进行更准确的评估，从而推动智能移动技术的发展。
## 34. `cs.CV` - DOS: 在文本嵌入中对多对象图像生成进行方向性物体分离 [PDF](https://arxiv.org/pdf/2510.14376), [HTML](https://arxiv.org/abs/2510.14376)
### Authors
Dongnam Byun,Jungwon Park,Jungmin Ko,Changin Choi,Wonjong Rhee
### Background
近年来，文本到图像（T2I）生成模型的进展显著提高了生成高质量、与文本提示一致的图像的能力。然而，这些模型在处理涉及多个物体的提示时仍然存在问题，常导致物体忽略或物体混合。
### Innovation
本文通过研究识别了四个问题场景：相似形状、相似纹理、背景偏见不一致以及多个物体。受CLIP嵌入的两个关键观察启发，提出了一种名为DOS（Directional Object Separation）的方法，它修改了三种类型的CLIP文本嵌入，然后将其传递给文本到图像模型。实验结果表明，DOS可以提高多物体图像生成的成功率并减少物体混合。在人类评估中，DOS在四项基准测试中显著优于四个竞争方法。
### Conclusion
DOS被证明是一种实用且有效的解决方案，能够提高多物体图像生成。
## 35. `cs.CV` - GenTrack2：改进的视觉多目标跟踪的混合方法 [PDF](https://arxiv.org/pdf/2510.24410), [HTML](https://arxiv.org/abs/2510.24410)
### Authors
Toan Van Nguyen,Rasmus G. K. Christiansen,Dirk Kraft,Leon Bodenhagen
### Background
本文提出了一个结合随机和确定性机制的视觉多对象跟踪方法，以在非线性动态下确保未知和时间变化的目标数量的一致性标识符。该方法旨在处理非线性动力学和非高斯噪声问题，并在未来帧不可用的情况下，灵活地应用于预录制视频和摄像机现场流。
### Innovation
1. 基于随机粒子滤波器，利用粒子群优化（PSO）进行引导，通过融合运动一致性、外观相似性和与邻近目标的社会互动线索来控制粒子，降低发散性。2. 提出一种确定性关联方案，采用综合空间一致性、检测置信度和轨迹惩罚的成本矩阵来进一步确保标识符一致性。3. 提出一种新的方案，以平滑地更新目标状态，同时保持它们的身份，特别是在与其他目标的交互期间以及长时间遮挡的情况下，特别强化了弱轨迹的管理。4. 使用过去的状态进行速度回归，提供趋势种子速度，增强粒子采样和状态更新。
### Conclusion
实验结果表明，提出的跟踪器在与其他最先进的跟踪器的比较中表现出色。重新实现了本文提出的方法和比较跟踪器的源代码的GitHub链接也已提供。
## 36. `cs.CV` - UniCalli：一种用于中文书法列级别生成和识别的统一扩散框架 [PDF](https://arxiv.org/pdf/2510.13745), [HTML](https://arxiv.org/abs/2510.13745)
### Authors
Tianshuo Xu,Kai Wang,Zhifei Chen,Leyi Wu,Tianshui Wen,Fei Chao,Ying-Cong Chen
### Background
现有的中文书法复现方法存在局限性。一方面，它们在生成高质量独立汉字时往往忽略了诸如连笔和行距等页面级美学，另一方面，它们又可能在尝试生成整个页面时牺牲了书法的正确性。
### Innovation
UniCalli 提出了一种统一的扩散框架，用于列级别的识别和生成。该框架通过联合训练这两个任务来提高综合表现：识别任务限制生成器保留字符结构，生成任务提供风格和布局先验。这种协同作用促进了概念级的抽象，尤其是在数据量有限的情况下。该方法使用一个包含数字化超过 8000 份书法作品的自建数据集进行训练，并利用非对称噪声和矢量化边界图作为空间先验，结合合成、标记和未标记数据进行训练。
### Conclusion
UniCalli 模型在生成质量方面达到了最先进的水平，同时在连笔连续性和布局忠实度方面表现出色，并且在识别方面同样强劲。此外，该框架成功扩展到了其他古代文字系统，如甲骨文和埃及象形文字。完整的代码和数据可以在提供的链接中查看。
## 37. `cs.CV` - NP-LoRA：通过零空间投影统一LoRA融合中的主体和风格 [PDF](https://arxiv.org/pdf/2511.11051), [HTML](https://arxiv.org/abs/2511.11051)
### Authors
Chuheng Chen,Xiaofei Zhou,Geyuan Zhang,Yong Huang
### Background
现有的方法依赖于在共享的适应空间中基于权重的合并，导致独立训练的LoRA相互干扰，降低生成的保真度。
### Innovation
本文发现生成行为主要由少数主导方向控制，提出将LoRA融合重新定义为零空间投影问题，并提出了Null Space Projection LoRA (NP-LoRA)，这是一种基于投影的框架，通过设计确保子空间分离。NP-LoRA 通过奇异值分解（SVD）提取主导风格方向，并将主体LoRA投影到风格子空间的正交补空间，从而避免干扰。同时引入了一个软投影机制，提供在主体保真度和风格保留间连续控制的手段。
### Conclusion
实验证明，NP-LoRA 在各种训练好的LoRA对上一致优于强基线，并且无需重新训练即可良好泛化。
## 38. `cs.CV` - 关于高效视觉-语言-行动模型的研究 [PDF](https://arxiv.org/pdf/2510.24795), [HTML](https://arxiv.org/abs/2510.24795)
### Authors
Zhaoshu Yu,Bo Wang,Pengpeng Zeng,Haonan Zhang,Ji Zhang,Zheng Wang,Lianli Gao,Jingkuan Song,Nicu Sebe,Heng Tao Shen
### Background
视觉-语言-行动模型（VLAs）是自主智能的前沿领域，旨在弥合数字知识与物理世界互动之间的鸿沟。尽管这些模型表现出色，但由于大型架构的计算和数据需求高昂，基础VLAs受到限制。近期的研究主要集中在提高VLA效率上，但缺乏一个统一的框架来整合这些进展。
### Innovation
本文首次全面回顾了涵盖模型训练数据管道的高效视觉-语言-行动模型（Efficient VLAs）。提出了一个统一的分类系统，将当前技术分为三大支柱：（1）高效模型设计，侧重于高效架构和模型压缩；（2）高效训练，减少模型学习过程中的计算负担；（3）高效数据采集，解决机器人数据获取和利用的瓶颈。
### Conclusion
通过对该框架下先进技术的批判性回顾，本文不仅为社区提供了基础参考，还总结了代表性应用、阐明了关键挑战，并为未来研究绘制作图。持续更新的项目页面跟踪最新进展：this https URL.
## 39. `cs.CV` - T-MLA: 针对神经图像压缩的一种目标化多尺度对数-指数攻击框架 [PDF](https://arxiv.org/pdf/2511.01079), [HTML](https://arxiv.org/abs/2511.01079)
### Authors
Nikolay I. Kalmykov,Razan Dibo,Kaiyu Shen,Xu Zhonghan,Anh-Huy Phan,Yipeng Liu,Ivan Oseledets
### Background
神经图像压缩（NIC）已经成为了最先进技术，在压缩率和失真之间表现出优越的性能。然而，这些技术的安全脆弱性却比分类器（classifier）的安全问题理解得少得多。现有对NIC的对抗性攻击往往是将像素空间的方法进行简单的适应，忽视了压缩流水线的独特性和结构特性。
### Innovation
本文提出了一个更高级的攻击框架，称之为T-MLA——第一个目标化多尺度对数-指数攻击框架。该框架在小波域内引入对抗性扰动，这种扰动集中于不太具感知性的系数上，从而提高了攻击的隐蔽性（stealth）。
### Conclusion
针对多个最先进的NIC架构的广泛评估显示，当扰动在视觉上不可感知时，T-MLA实现了对重构质量的针对性降级。与PGD风格的基准相比，当攻击成功率相似时，T-MLA在提高扰动的不可感知性（更高的PSNR/VIF）方面表现更佳。研究结果揭示了生成性和内容分发管道的核心安全缺陷。
## 40. `cs.CV` - SciTextures: 收集并跨学科连接视觉模式、模型和代码 [PDF](https://arxiv.org/pdf/2511.01817), [HTML](https://arxiv.org/abs/2511.01817)
### Authors
Sagi Eppel,Alona Strugatski
### Background
理解视觉模式与其形成过程之间的联系代表着最深层次的视觉理解之一。自然界和人造世界中的花纹和模式，如云朵和波浪的纹理、城市和森林的生长、材料和地貌的形成，都是由底层机制产生的。现有的研究缺乏一种能够系统地探索这些视觉模式及其生成机制之间关系的工具或数据集。SciTextures 数据集填补了这一空白，它提供了来自科学、技术和艺术领域的花纹和视觉模式的巨大集合，以及生成这些图像的模型和代码。
### Innovation
SciTextures 数据集是通过自主的人工智能管道收集、实现和标准化科学及生成模型建立的，该管道还用于自主发明和实现生成视觉模式和纹理的新方法。该数据集不仅包含超过 1,270 种不同的模型和 100,000 张图案和纹理图像，还支持使用视觉语言模型（VLM）评估其将视觉模式与其生成源关联的能力，识别相同底层过程产生的不同模式。此外，还可以利用自然图像测试 VLM 从视觉模式推测生成机制并重建过程模型的能力。
### Conclusion
SciTextures 数据集和代码提供了评估 VLM 是否能够跨多个抽象层次理解和模拟物理系统的能力。研究结果表明，VLM 可以根据视觉模式理解并模拟从微观到宏观的物理系统。数据集和相关代码可以从指定的链接获得。
## 41. `cs.LG` - 希尔伯特空间中的随机插值 [PDF](https://arxiv.org/pdf/2602.01988), [HTML](https://arxiv.org/abs/2602.01988)
### Authors
James Boran Yu,RuiKang OuYang,Julien Horwood,José Miguel Hernández-Lobato
### Background
尽管扩散模型可以成功地适用于函数值数据，但随机插值作为灵活连接任意分布的方法仍局限于有限维度设置。本文为解决这一问题，在无限维希尔伯特空间中建立了随机插值的严格框架。
### Innovation
本文提供了全面的理论基础，包括正则性和显式误差界证明。通过使生成器能够在一个任意函数分布之间建立桥梁，本文的方法取得了现有的最先进的成果，提供了一种强大的、通用的工具用于科学发现。
### Conclusion
通过在无限维希尔伯特空间中建立随机插值的严格框架，本文为任意函数分布之间的链接提供了强有力的方法，并通过复杂PDE基准测试证明了其有效性，达到了目前的最先进水平。
## 42. `cs.LG` - 通过缺失数据实现隐私增强 [PDF](https://arxiv.org/pdf/2602.01928), [HTML](https://arxiv.org/abs/2602.01928)
### Authors
Simon Roburin(LPSM),Rafaël Pinot(LPSM ),Erwan Scornet(LPSM )
### Background
在医学和金融等领域，保护个人隐私非常重要。这类领域中的敏感个人数据需要在不泄露个体隐私的情况下进行分析。然而，这些应用中的数据集经常因为响应不完整、数据损坏或故意脱敏等原因存在缺失值。传统上，缺失数据被视为一种限制因素，因为它减少了可供分析的信息量，还可能降低模型性能。
### Innovation
该研究从隐私保护的角度重新审视了缺失数据，提出了一种新的观点：当特征缺失时，披露给数据分析师的信息量减少，这暗示缺失数据具有天然的隐私增强特性。研究还首次证明了不完整数据能够对差异隐私算法实现隐私放大。
### Conclusion
通过分析缺失数据作为一种隐私增强机制，在差分隐私的框架内，研究表明不完整数据可以提升隐私保护的效果。
## 43. `cs.LG` - 多头注意力中 token 选择的几何分析 [PDF](https://arxiv.org/pdf/2602.01893), [HTML](https://arxiv.org/abs/2602.01893)
### Authors
Timur Mudarisov,Mikhal Burtsev,Tatiana Petrova,Radu State
### Background
本文提出了一个几何框架来分析大型语言模型（LLMs）中的多头注意力。研究通过将标准注意力视作 top-N 选择来直接在值-状态空间中研究其行为，定义了几何度量（精确度、召回率和 F 值）来量化选定和非选定 token 之间的可分辨性，并在经验动机假设下推导出非渐近界，这些假设包括稳定值范数、压缩终端标记以及指数相似度衰减和分段注意力权重配置文件。
### Innovation
本文提出了一个新的几何框架，通过 top-N 选择的视角来分析多头注意力，并定义了几何度量来量化 token 之间的可分辨性。此外，推导出非渐近界，并发现多头注意力呈现出三个不同的几何特征区域：检索器、混合器和重置器。
### Conclusion
研究表明，top-N 选择可以增强 token 的可分辨性，终端相似度与召回率相关。多头注意力表现出作为结构化几何分类器的行为，具有可量化的 token 选择标准，这为多头注意力的头部层面解释和设计提供了指导。
## 44. `cs.LG` - 从远偏移传播先验知识：拖缆数据逐级恢复近偏移的自监督扩散框架 [PDF](https://arxiv.org/pdf/2602.01909), [HTML](https://arxiv.org/abs/2602.01909)
### Authors
Shijun Cheng,Tariq Alkhalifah
### Background
在海洋拖曳式线阵地震采集中，最近的水听器通常距离声源约200米，导致丢失了近偏移道数据，影响了诸如地震表面相关的多重消除、速度分析和全波形反演等关键处理流程。现有的重建方法，如变换域插值，常会产生动力学不一致和幅度失真，而监督深度学习方法则需要完整的近偏移参考数据，但在实际采集场景中这些数据是不可用的。
### Innovation
本文提出了一种基于自监督扩散的框架，能够在无需近偏移参考数据的情况下重建缺失的近偏移道数据。该方法利用重叠块提取结合来自可用远偏移段的单道位移来训练条件扩散模型，学习偏移依赖的统计特性。推理时，从最近记录的偏移逐道递归推断到零偏移，逐步将学习到的先验信息从远到近传播。生成性框架还能通过聚合采样提供不确定性估计，量化预测置信度。
### Conclusion
控制实验表明，该方法在合成数据集和实地数据集上的性能显著优于传统的抛物线拉东变换基线。实际部署表明即使在无法进行地面真实验证的情况下，该方法具有实用价值。由于仅训练在远偏移观察，重建波形能够保持现实的幅度-偏移趋势，而不确定性图可以准确识别推断困难的区域。
## 45. `cs.LG` - 通过基于量ile回归森林和配准校准的实时可靠VaR估计 [PDF](https://arxiv.org/pdf/2602.01912), [HTML](https://arxiv.org/abs/2602.01912)
### Authors
Du-Yi Wang,Guo Liang,Kun Zhang,Qianwen Zhu
### Background
快速变化的市场条件需要实时的风险监控，但其在线估计仍然具有挑战性。本文研究了最常用的风险度量VaR的在线估计问题，准确可靠的VaR估计对于及时的风险控制和明智的决策至关重要。
### Innovation
提出了使用量ile回归森林在离线-仿真-在线-估计(OSOA)框架中进行VaR的在线估计。进一步提出了一种配准化的估计器来校准在线VaR估计，这是首次在OSOA框架下利用配准校准来可靠地估计实时VaR。理论分析证明了所提估计器的一致性和覆盖率有效性。
### Conclusion
数值实验验证了所提出的方法，证明其在实践中具有有效性。
## 46. `cs.LG` - 你的AI生成图像检测器经过校准后可以实现SOTA准确性 [PDF](https://arxiv.org/pdf/2602.01973), [HTML](https://arxiv.org/abs/2602.01973)
### Authors
Muli Yang,Gabriel James Goenawan,Henan Wang,Huaiyuan Qin,Chenghao Xu,Yanhua Yang,Fen Fang,Ying Sun,Joo-Hwee Lim,Hongyuan Zhu
### Background
尽管现有的AI生成图像检测器在平衡数据集上进行训练，但在测试时经常会系统性地误分类假图像为真图像。认为这种行为源于假样本分布的变化和训练期间学习的隐式先验。具体来说，模型倾向于过度拟合到不适用于不同生成方法的表面特征，导致在面对测试时分布变化时出现决策阈值不一致。
### Innovation
提出了一种基于贝叶斯决策理论的理论上可行的后处理校准框架。引入了一个可学习的标量修正到模型的logits上，在目标分布的一个小验证集上进行优化，同时冻结主干。这种方法的参数调整可以补偿模型输出的分布变化，即使不使用ground-truth标签也能重新校准决策边界。
### Conclusion
在具有挑战性的基准测试上进行的实验表明，我们的方法显著提高了检测的鲁棒性而无需重新训练，为在开放环境中提供了一个轻量级且原理性的可靠且自适应的AI生成图像检测解决方案。
## 47. `cs.LG` - 小的且具普适性的提示预测模型能够在大规模推理模型的后训练中引导高效的强化学习 [PDF](https://arxiv.org/pdf/2602.01970), [HTML](https://arxiv.org/abs/2602.01970)
### Authors
Yun Qu,Qi Wang,Yixiu Mao,Heming Zou,Yuhang Jiang,Weijie Liu,Clive Bai,Kai Yang,Yangkun Chen,Saiyong Yang,Xiangyang Ji
### Background
强化学习能够提升大语言模型的推理能力，但优化过程中常常产生高昂的计算成本。在线提示选择作为一种可能的解决方案，通过优先使用信息量大的提示以提高训练效率。然而，当前的方法要么依赖昂贵的精确评估，要么构建针对特定提示的预测模型却缺乏跨提示的一般化。
### Innovation
这项研究引入了通用预测提示选择（GPS），使用在共享优化历史中训练的轻量级生成模型进行贝叶斯推理预测提示难度。通过优先选择中等难度的提示和基于历史的多样性，这一模型在批处理获取原则中有效地选择了信息量大的提示批。此外，这个小型的预测模型在测试时也能泛化，从而实现高效的计算分配。
### Conclusion
实验结果表明，该方法在各种推理基准测试中显著提高了训练效率，最终性能和测试时的效率，优于现有的基线方法。
## 48. `cs.LG` - 通过空间语义因子化学习稀疏视觉表示 [PDF](https://arxiv.org/pdf/2602.01905), [HTML](https://arxiv.org/abs/2602.01905)
### Authors
Theodore Zhengde Zhao,Sid Kiblawi,Jianwei Yang,Naoto Usuyama,Reuben Tan,Noel C Codella,Tristan Naumann,Hoifung Poon,Mu Wei
### Background
现有的自我监督学习（SSL）面临着语义理解和图像重建之间的基本冲突。高阶语义SSL（例如，DINO）依赖于全局tokens，这些tokens被强迫成为位置不变的，以实现增强对齐，但这一过程会丢弃用于重建的所需的空间坐标。相反，生成型SSL（例如，MAE）保留了密集特征网格以进行重建，但无法产生高级抽象。
### Innovation
本文引入了STELLAR框架，该框架通过将视觉特征分解为语义概念和它们的空间分布的低秩乘积来解决这种矛盾。这种解耦允许我们在语义tokens上执行DINO风格的增强对齐，同时保留用于像素级重建所必需的空间映射在定位矩阵中。
### Conclusion
少量16个稀疏token在这种分解形式下足以同时支持高质量的重建（2.60 FID）并匹配密集骨干网的语义性能（79.10% ImageNet准确率）。实验结果突显了STELLAR作为通用稀疏表示的潜力，它通过策略性地分离语义身份和空间几何，填补了鉴别性和生成性视觉间的缺口。
## 49. `cs.LG` - FluxNet：学习容量约束的局部传输算子以实现保守且有界的PDE近似 [PDF](https://arxiv.org/pdf/2602.01941), [HTML](https://arxiv.org/abs/2602.01941)
### Authors
Zishuo Lan,Junjie Li,Lei Wang,Jincheng Wang
### Background
时间步长操作的自回归学习为网格上的数据驱动偏微分方程(PDE)模拟提供了一种有效的方法。然而，对于守恒定律，当通过学习更新违反了全局守恒定律时，长时多步模拟经常变得不稳定，尤其是在许多实际应用中，需要额外的状态限制，例如非负质量以及在[0,1]区间内的密度或浓度。通过直接回归预测下一状态来强制这些联合约束仍然具有挑战性。
### Innovation
本文提出了一种框架，用于在正则网格上学习保守的传输算子，该框架受到格子玻尔兹曼风格离散速度传输表示的启发。该模型输出局部传输算子，通过邻域交换更新单元，这种构造保证了离散守恒。对于有界量，该方法通过在容量受限可行集中参数化的传输来解决范围约束问题，从而在构造上强制这些限制，而不是事后剪裁。
### Conclusion
FluxNet在1D对流扩散、2D浅水方程、1D交通流和2D相场分解相分离等方面得到了验证。在浅水方程和交通流实验中，与强大的基线相比，FluxNet展示了改进的模拟稳定性和物理一致性。对于相场分解相分离，该方法允许使用长距离传输进行大时间步长模拟，从而加速模拟并同时保持微观结构的演化，无论是定点还是统计指标上都能够做到。
## 50. `cs.LG` - SpikingGamma：无需梯度代理且具有时序精确性的在线SNN训练方法 [PDF](https://arxiv.org/pdf/2602.01978), [HTML](https://arxiv.org/abs/2602.01978)
### Authors
Roel Koopman,Sebastian Otte,Sander Bohté
### Background
神经形态硬件实现的脉冲神经网络（SNNs）通过稀疏、事件驱动的计算提供了能耗低、延迟低的AI计算。然而，在微细时域离散化下训练SNN仍是一项重大挑战，影响了其快速响应能力以及软件训练的SNN映射到高效硬件的有效性。目前方法中，脉冲神经元被建模为自递归单元，嵌入递归网络以维持时间状态，通过基于替代梯度的BPTT或RTRL变体进行训练。这些方法在时域分辨率较大的情况下表现不佳，而在线近似往往在长序列中表现不稳定，很难精确捕捉时序模式。
### Innovation
开发了具有内部递归记忆结构的脉冲神经元，结合sigma-delta脉冲编码，提出SpikingGamma模型。该模型可以直接进行反向传播训练，无需替代梯度，能够在在线模式下以最小脉冲数学习精细的时间序列模式，并可将前向传播SNN扩展到复杂的任务和基准上，同时对模型的时间分辨率不敏感。该方法提供了一种与当前依赖替代梯度训练的递归SNNs的替代方案，并且提供了一条直接将SNN映射到神经形态硬件的路径。
### Conclusion
SpikingGamma模型支持直接误差反向传播，可以在线模式下以少量脉冲学习细微的时间序列模式，并能够扩展前馈SNNs到复杂任务和基准，同时对模型的时间分辨率不敏感。该方法不仅为SNN训练提供了一种替代方案，还提供了一种直接将SNN映射到神经形态硬件的途径。
