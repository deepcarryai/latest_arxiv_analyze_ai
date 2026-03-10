# 20260310
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 动态的世界：代理基准的可编程演化 [PDF](https://arxiv.org/pdf/2603.05910), [HTML](https://arxiv.org/abs/2603.05910)
### Authors
Guangrui Li,Yaochen Xie,Yi Liu,Ziwei Dong,Xingyuan Pan,Tianqi Zheng,Jason Choi,Michael J. Morais,Binit Jha,Shaunak Mishra,Bingrou Zhou,Chen Luo,Monica Xiao Cheng,Dawn Song
### Background
现有的代理基准大多假设静态环境，具有固定的架构和工具集，忽视了现实世界环境的演变性质以及代理在面对环境变化时的鲁棒性。现有的代理与环境之间的互动和数据查询过程是在多轮对话中完成的。
### Innovation
本文提出了一个基于图的框架——ProEvolve，使得环境演化可以编程实现。该框架通过类型化的关系图提供了一个统一的、明确的环境表示，涵盖了数据、工具和架构。在此基础上，通过图转换编程动态生成环境和任务沙盒。
### Conclusion
验证结果显示，通过ProEvolve框架，一个环境可以演化成200个环境和3000个任务沙盒，并据此对代表性代理进行了基准测试。
## 2. `cs.AI` - 基于CliqueFlowmer的离线材料优化 [PDF](https://arxiv.org/pdf/2603.06082), [HTML](https://arxiv.org/abs/2603.06082)
### Authors
Jakub Grudzien Kuba,Benjamin Kurt Miller,Sergey Levine,Pieter Abbeel
### Background
当前计算材料发现（CMD）领域中，深度学习驱动的神经网络方法取得显著进展。在CMD领域，一个常见的问题是如何找到优化目标性质的材料。然而，广受欢迎的生成模型方法由于最大似然训练方式，在大胆探索材料空间的吸引区域方面效果不佳。
### Innovation
本文提出了一种基于离线模型优化（MBO）的新CMD技术，通过将目标材料性质的直接优化融入生成过程来解决上述问题。具体来说，我们引入了一个领域特异性模型CliqueFlowmer，该模型将基于团索引的MBO方法与变压器和流生成技术结合起来，以提高材料的优化能力。实验表明，CliqueFlowmer产生的材料显著优于现有生成基准提供的材料。
### Conclusion
为了使CliqueFlowmer能够在特定材料优化问题中广泛使用并支持跨学科研究，我们已经将模型的源代码公开，可以通过如下链接访问：this https URL。
## 3. `cs.AI` - 一种新的产品概念评估的交互式多智能体系统 [PDF](https://arxiv.org/pdf/2603.05980), [HTML](https://arxiv.org/abs/2603.05980)
### Authors
Bin Xuan,Ruo Ai,Hakyeon Lee
### Background
产品概念评估是企业中决定战略资源分配和项目成功的关键阶段。然而，传统的专家驱动方法存在主观偏差和高时间和成本要求等局限性。
### Innovation
本研究提出了一个利用基于大型语言模型（LLM）的多智能体系统（MAS）的自动化评估方法。该系统通过系统的研发和团队合作研究，建立了两个主要评估维度，分别是技术可行性和市场可行性。该系统由八个虚拟代理组成，代表不同的专业领域如研发和市场推广。这些代理使用检索增强生成（RAG）和实时搜索工具来收集客观证据，并根据既定标准进行结构化推理以验证概念。此外，通过使用专业的产品评价数据进一步对代理进行微调，提高了其判断准确性。
### Conclusion
案例研究发现，系统评估排名与资深行业专家的排名一致。这些结果证实了基于多智能体的评估方法的实用性，适用于支持产品开发决策。
## 4. `cs.AI` - Quantitative Bipolar 量化极性论辩框架下的聚合语义 [PDF](https://arxiv.org/pdf/2603.06067), [HTML](https://arxiv.org/abs/2603.06067)
### Authors
Yann Munro,Isabelle Bloch,Marie-Jeanne Lesot
### Background
论辩论法在人工智能中的应用正变得越来越广泛，它提供了一种有效且易于理解的方式来建模可能会出现冲突的信息。在定量极性论辩框架（QBAF）的具体背景下，由于论点具有固有的权重并可以互相攻击或支持，因此需要处理这种不对称性。
### Innovation
论文提出了一种新的聚合语义家族，称为聚合语义。这种语义在计算过程中分为三个步骤：首先分别计算攻击者和支持者的权重，然后将这两个值与论点的固有权重综合。这种方法在计算过程中分解为三个可以解释的步骤，从而使计算更加可参数化，并且这一步进一步保持了极性，使得提出的语义更加易于理解。
### Conclusion
论文讨论了在不同的情境下，聚合函数应满足的属性以及它们与经典渐进义原则的关系。通过各种简单示例和一个包含五个百种聚合语义的最后示例，论文展示了聚合语义可能表现出的各种行为，表明这为理解和应用于QBAF提供了大幅度的灵活性和可调节性。
## 5. `cs.AI` - DeepFact：协同演进基准和代理以提高深度研究事实性 [PDF](https://arxiv.org/pdf/2603.05912), [HTML](https://arxiv.org/abs/2603.05912)
### Authors
Yukun Huang,Leonardo F. R. Ribeiro,Momchil Hardalov,Bhuwan Dhingra,Markus Dreyer,Venkatesh Saligrama
### Background
现有的LLM代理能够生成深度研究报告，但验证报告中的命题事实性仍然是一个挑战。现有的事实检查器主要针对一般领域的、事实性的原子命题进行设计，没有一个基准可以测试这类验证器在深度研究报告中的适用性。由于构建这样一个基准本身就具有挑战性，研究者首先发现静态专家标注的基准在这个环境中非常脆弱。通过一项博士水平专家参与的研究，研究者发现专家在无辅助情况下对验证命题的准确性仅为60.8%，这表明现有的专家标注基准无法提供足够的验证依据。
### Innovation
研究者提出了Evolve Benchmarking via Audit-then-Score (AtS)方法，该方法允许基准标签和解释明确地进行修订。当验证器与当前基准不一致时，它必须提交证据，审计员裁定争端，接受的修订更新基准以供模型评分。DeepFact-Bench和DeepFact-Eval作为实例，通过版本化有审计依据的DRR事实性基准，展示了更高的准确性。与现有验证器相比，DeepFact-Eval在DeepFact-Bench上表现出色，并且可以很好地迁移到外部真实性的数据集。
### Conclusion
经过四轮AtS过程，专家对隐藏验证命题的微金标准的准确性从60.8%提高到90.9%，表明专家作为审计员比作为一次性标签者更为可靠。DeepFact-Bench和DeepFact-Eval方法为开发可靠的深度研究事实性验证器奠定了基础，有助于实现对深度研究报告中事实性的有效验证。
## 6. `cs.AI` - 推理模型难以控制其思维链 [PDF](https://arxiv.org/pdf/2603.05706), [HTML](https://arxiv.org/abs/2603.05706)
### Authors
Chen Yueh-Han,Robert McCarthy,Bruce W. Lee,He He,Ian Kivlichan,Bowen Baker,Micah Carroll,Tomek Korbak
### Background
思维链（CoT）监测被认为是检测模型不良行为和理解其动机的有效工具。然而，如果模型可以控制自己在思维链中所表达的内容，那么这种能力可能会削弱CoT的可监测性。因此，有必要评估模型在思考过程中控制其CoT的能力。
### Innovation
作者提出了一个名为CoT-Control的评估套件，包括需要模型遵循CoT指令解决问题的任务。研究发现，与最终输出的控制相比，推理模型在控制其思维链方面的表现较差；并且发现大型模型在思维链控制方面表现较好，但随着强化学习训练次数、测试阶段计算资源的增加以及问题难度的提高，这种控制能力会下降。此外，即使在给模型提供激励而非直接请求时，模型也难以控制其思维链。
### Conclusion
研究结果表明，当前思维链控制可能会妨碍思维链监测功能失灵的可能性较小，但其背后的机制尚不清楚。由于思维链监测的重要性，应对未来模型的思维链控制能力进行跟踪。
## 7. `cs.AI` - RoboLayout: 可承载实体代理的差分3D场景生成 [PDF](https://arxiv.org/pdf/2603.05522), [HTML](https://arxiv.org/abs/2603.05522)
### Authors
Ali Shamsaddinlou
### Background
最近的视觉语言模型（VLMs）在从开放语言指令中推理空间布局方面展现了强大的潜力，尤其在生成3D场景布局方面。然而，生成既符合语义又在物理上可交互的场景布局仍然是一个挑战，尤其是在受到物理限制的室内环境中。目前的模型虽然能够生成合理的布局，但是对于特定实体代理（如服务机器人、仓库机器人、人类、不同年龄的人或动物）的可交互性方面的表现仍然不够理想。
### Innovation
本文引入了RoboLayout，作为LayoutVLM的扩展模型。RoboLayout通过嵌入到可微优化过程中的可达性约束等手段，生成了既可导航又可操作的场景布局。同时，RoboLayout提出了局部优化阶段，可以对特定布局问题进行局部优化，提高优化效率，同时保持整体优化迭代次数不变。RoboLayout保留了LayoutVLM强大的语义一致性和物理合理性，并且能够根据需要进行场景布局设计，以适应不同类型的实体代理。
### Conclusion
RoboLayout展示了在室内场景中的实体代理中心化生成方面的应用潜力，通过实验结果验证了模型的有效性，特别是在适应不同实体代理的物理限制方面，改善了实体代理在场景中的交互性。
## 8. `cs.AI` - 通过逐步PDDL模拟的代理LLM规划：实证特征描述 [PDF](https://arxiv.org/pdf/2603.06064), [HTML](https://arxiv.org/abs/2603.06064)
### Authors
Kai Göbel,Pierrick Lorang,Patrik Zips,Tobias Glück
### Background
任务规划，即从初始状态通过序列化动作到达目标的过程，是自主机器人系统的核心能力要求。目前，关于大型语言模型（LLMs）是否可以作为可行的规划工具与经典符号方法并行工作的问题仍处于开放状态。本文探讨了如何利用LLMs作为交互式搜索策略进行任务规划，以优化人类与机器人的协作。
### Innovation
作者提出了一种名为PyPDDLEngine的开源PDDL模拟引擎，该引擎通过模型上下文协议（MCP）接口将规划操作暴露为LLM工具调用。该引擎使LLMs能够逐步进行规划，而不是一次性承诺整个动作序列，具有基于每次操作结果进行调整和重试的能力。并通过四种方法对102个国际规划竞赛块世界实例进行了评价，结果表明代理LLM方法在某些复杂度块中的表现优于传统方法，同时也指出了代理策略的成功依赖于环境反馈的性质。
### Conclusion
实验结果表明，代理LLM方法在某些复杂度块中虽然成功几率较低，但生成的计划比seq-sat-lama-2011更短。这些发现表明，代理性能依赖于环境反馈的性质。编码代理能从外部信号获得益处，而基于PDDL的代理则需要自我评估，这表明代理规划的改进可能更依赖于特定的学习数据回忆机制而非广泛的规划能力。
## 9. `cs.AI` - 基于经验驱动自我技能发现的医学影像代理演化 [PDF](https://arxiv.org/pdf/2603.05860), [HTML](https://arxiv.org/abs/2603.05860)
### Authors
Lin Fan,Pengyu Dai,Zhipeng Deng,Haolin Wang,Xun Gong,Yefeng Zheng,Yafei Ou
### Background
医学图像解读是一个多步和工具为中心的过程：临床医生通过一系列专门的程序，迭代结合视觉证据与患者背景，量化发现并调整决策。虽然基于大规模语言模型的代理有潜力协调这些异构医疗工具，但现有系统在部署后将工具集和调用策略视为静态，这样的设计在真实世界的领域变化、任务和诊断需求变化的情况下表现脆弱，预定义的工具链经常失效，需要昂贵的手动重新设计。
### Innovation
提出了一种自我进化、经验增强的医学代理MACRO，从静态工具组成转向经验驱动的工具发现。通过验证执行轨迹，代理自主识别重复的有效多步工具序列，这些序列被合成并注册为可重用的复合工具，形成新的高级原语，以持续扩展其行为能力。轻量级的图像特征记忆将工具选择置于视觉和临床背景中，而类似于GRPO的训练循环则强化了已发现的复合工具的可靠调用，实现了在最少监督下的闭环自我改进。
### Conclusion
对多种医学影像数据集和任务进行了广泛实验，结果表明，自主复合工具发现能够持续提高多步协调准确性和跨域泛化能力，优于强基线和近期的先进代理方法，弥补了脆弱静态工具使用和适应性强的上下文感知临床AI辅助之间的差距。代码将在接受后提供。
## 10. `cs.AI` - 实时AI服务经济：跨连续体的自主计算框架 [PDF](https://arxiv.org/pdf/2603.05614), [HTML](https://arxiv.org/abs/2603.05614)
### Authors
Lauri Lovén,Alaa Saleh,Reza Farahani,Ilir Murturi,Miguel Bordallo López,Praveen Kumar Donta,Schahram Dustdar
### Background
实时AI服务在设备-边缘-云连续体中运行，自主AI代理生成敏感延迟的工作负载，协调多阶段处理管道，并在政策和管理约束下竞争共享资源。论文探讨了依赖关系图结构对分布式、基于价格的资源分配可靠性的影响。
### Innovation
论文提出了一种新的架构，可以在运营商接口和市场其他部分之间封装复杂的子图，形成资源切片，从而降低价格波动并提高管理效率。研究证实了依赖图拓扑对价格稳定性和可扩展性的决定性影响，并且该协作架构通过减少价格波动最大70-75%而不牺牲吞吐量，验证了分布式协调可以达到集中式的最优分配质量。
### Conclusion
波动的价格和复杂性之间的权衡可以通过采用跨域整合者来减少，这些整合者能够在不牺牲吞吐量的情况下，封装复杂子图以呈现一个更简单且结构化的接口。在真实投标中，去中心化的市场能够达到集中的价值最优基准。因此，去中心化协调可以复制集中式分配质量。
## 11. `cs.CV` - VS3R: 通过深度3D重建实现稳健的全幅视频稳定 [PDF](https://arxiv.org/pdf/2603.05851), [HTML](https://arxiv.org/abs/2603.05851)
### Authors
Muhua Zhu,Xinhao Jin,Yu Zhang,Yifei Xue,Tie Ji,Yizhen Lao
### Background
视频稳定旨在减轻相机晃动，但面临着几何稳健性和全幅连贯性之间的根本权衡。二维方法由于剧烈裁剪而受到影响，三维技术则往往因脆弱的优化管道在极端运动情况下失效。
### Innovation
本文提出了VS3R框架，结合了前馈3D重建和生成视频扩散。该框架联合估计了相机参数、深度和遮罩，以确保所有场景的可靠性，并引入了混合稳定渲染模块，融合了语义和几何线索以实现动态一致性。最后，引入了双流视频扩散模型，通过结构引导与语义锚点的协同作用来修复消失区域并纠正伪影。
### Conclusion
综上所述，VS3R实现了不同摄像机模型的高保真、全幅视频稳定，并在稳健性和视觉质量方面显著优于现有方法。
## 12. `cs.CV` - 无需训练的基态帧间剪枝与注意力恢复 [PDF](https://arxiv.org/pdf/2603.05811), [HTML](https://arxiv.org/abs/2603.05811)
### Authors
Dennis Menn,Yuedong Yang,Bokun Wang,Xiwen Wei,Mustafa Munir,Feng Liang,Radu Marculescu,Chenfeng Xu,Diana Marculescu
### Background
当前的视频生成模型存在较高的计算延迟问题，使得实时应用的成本变得非常高。这主要是由于视频帧间存在大量的冗余信息。现有方法无法有效利用这种冗余性，从而影响了实时性能。
### Innovation
本文提出了一种名为Latent Inter-frame Pruning with Attention Recovery (LIPAR)的新框架，通过检测并跳过重复的基态视频片段，减少不必要的计算量。同时引入了一种新颖的注意力恢复机制，可以近似恢复剪枝掉的部分，从而避免简单剪枝方法带来的视觉伪影问题。实验结果表明该方法可以提高视频编辑的吞吐量，平均实现12.2 FPS，相比基准方法8.4 FPS，提升了1.45倍。该方法在不牺牲生成质量和无需额外训练的情况下，能够无缝集成到模型中。
### Conclusion
本文有效地将传统压缩算法与现代生成管道相结合，通过LIPAR框架提高了视频处理的实时性能，同时保持了生成的质量，为视频生成模型在实时应用中提供了新的解决方案。
## 13. `cs.CV` - Calibrated and Robust Vision Models通过Margin和一致性监督 [PDF](https://arxiv.org/pdf/2603.05812), [HTML](https://arxiv.org/abs/2603.05812)
### Authors
Salim Khazem
### Background
深度视觉分类器在保持高准确率的同时，往往难以校准且在小分布偏移下显得脆弱。现有方法往往在分类准确度和模型鲁棒性、校准性之间难以平衡。
### Innovation
提出了一种简单且架构无关的正则化框架——Margin和一致性监督（MaCS），它同时在逻辑空间中强化了分类边界，并确保了本地预测的稳定性。MaCS通过hinge-squared的边界惩罚（目标分类得分与次强竞争对手得分之间的差距）和基于对清洁输入和轻微扰动视图之间预测KL散度的最小化一致性正则化来工作，从而提高了泛化保证和鲁棒性的理论证明。
### Conclusion
在多个图像分类基准和多种骨干网络（包括CNN和Vision Transformers）上，MaCS皆能提高校准效果（如降低ECE和NLL）和对常见篡改的鲁棒性，同时保持或提升顶级准确率。该方法无需额外数据或架构更改，即使在推理过程中也几乎无额外开销，是一种有效的替代标准训练目标的方法。
## 14. `cs.CV` - EventGeM：基于事件的全局到局部特征匹配视觉地点识别 [PDF](https://arxiv.org/pdf/2603.05807), [HTML](https://arxiv.org/abs/2603.05807)
### Authors
Adam D. Hines,Gokul B. Nair,Nicolás Marticorena,Michael Milford,Tobias Fischer
### Background
动态视觉传感器，也称为事件相机，因其稀疏激活和高时间分辨率而在机器人和计算机视觉任务中迅速流行。事件相机已被用于需要在小时间尺度内进行准确定位的机器人导航和定位任务，或者当能量需求至关重要时。
### Innovation
提出了一种名为EventGeM的最先进的全局到局部特征融合管道，用于基于事件的视觉地点识别。该方法采用了预训练的ViT-S/16视觉变压器作为骨干网络，用于从事件直方图图像中获取全局特征块，并进行初始匹配预测嵌入。利用预训练的MaxViT骨干网络检测局部特征关键点，并进行基于2D同源变换的再排序，使用RANSAC。随后，使用预训练的视觉基础模型进行深度估计，以比较参考和查询之间的结构相似性。此工作在多种基准数据集和光照条件下表现出最先进的定位性能，同时能够在各种计算架构上实时运行。
### Conclusion
EventGeM在多个基准数据集和光照条件下与目前最先进的基于事件的地点识别方法相比，实现了最先进的定位性能，在各种计算架构上部署时也完全具备实时运行能力。并且我们在一个基于事件相机的事件流上实现在机器人平台上的在线定位演示了EventGeM的能力。
## 15. `cs.CV` - Cog2Gen3D：为3D生成塑造语义几何认知 [PDF](https://arxiv.org/pdf/2603.05845), [HTML](https://arxiv.org/abs/2603.05845)
### Authors
Haonan Wang,Hanyu Zhou,Haoyue Liu,Tao Gu,Luxin Yan
### Background
生成模型已经在产生具有语义合理性的2D图像方面取得了成功，但在3D生成方面仍面临挑战，尤其是在缺乏空间几何约束的情况下。现有的方法通常通过利用几何特征来增强对空间关系的意识，但这些方法仅能建模相对关系，并且容易产生几何尺度不一致的问题。因此，该研究强调了语义信息和绝对几何在促进3D认知方面的作用，从而实现对物理世界可控的3D生成。背景指出，当前方法在处理3D几何一致性方面存在局限性。
### Innovation
提出了一种名为Cog2Gen3D的3D认知指导扩散框架，用于3D生成。Cog2Gen3D包含三个核心设计：1) 意识特征嵌入。将不同模态编码为语义和几何表示，进一步提取逻辑表示；2) 3D潜在认知图。将不同表示结构化成语义-几何双流图，并通过基于共同点的交叉注意机制融合，生成3D认知图；3) 意识指导的潜在扩散。利用融合的3D认知图作为条件，引导3D高斯生成的潜在扩散过程。统一框架下的3D认知图确保了3D生成的物理合理性和结构合理性。
### Conclusion
通过在Marble World Labs构建验证子集，实验证明Cog2Gen3D显著优于现有的方法，在语义保真度和几何合理性方面表现更优。
## 16. `cs.CV` - 使用深度集成学习的遥感图像分类 [PDF](https://arxiv.org/pdf/2603.05844), [HTML](https://arxiv.org/abs/2603.05844)
### Authors
Niful Islam,Md. Rayhan Ahmed,Nur Mohammad Fahad,Salekul Islam,A.K.M. Muzahidul Islam,Saddam Mukta,Swakkhar Shatabda
### Background
遥感图像在众多应用中扮演重要角色，需要精确的计算机分类技术。可靠的分类对于将原始图像转换为结构化和可用信息至关重要。尽管卷积神经网络（CNNs）主要用于图像分类，它们在局部特征提取方面表现出色，但在捕获全局上下文信息方面存在不足。视觉变换器（ViTs）通过自注意力机制解决了这一限制，可以建模长距离依赖关系。将CNNs和ViTs结合起来，可以比单独的架构获得更好的性能。然而，使用额外的CNN和ViT组件并未带来进一步的性能提升，反而引入了由于冗余特征表示导致的瓶颈。
### Innovation
该研究提出了一种结合CNNs和ViTs的融合模型，用于遥感图像分类。为了解决性能瓶颈，该方法训练了四个独立的融合模型，这些模型整合了CNN和ViT骨干，并在最终预测阶段通过集成方法组合其输出。该方法在UC Merced、RSSCN7和MSRSI数据集上的准确率分别达到了98.10%，94.46%和95.45%，超越了竞争对手的架构，并突显了所提出解决方案的有效性，尤其是在训练过程中高效使用计算资源方面。
### Conclusion
该研究提出了一种结合CNNs和ViTs的融合模型，通过适当的集成方法，在多个遥感图像数据集上取得了出色的分类性能，证实了这种方法的有效性和优越性。
## 17. `cs.CV` - 文本到图像扩散变换器中的逐层实例绑定用于区域和遮挡控制 [PDF](https://arxiv.org/pdf/2603.05769), [HTML](https://arxiv.org/abs/2603.05769)
### Authors
Ruidong Chen,Yancheng Bai,Xuanpu Zhang,Jianhao Zeng,Lanjun Wang,Dan Song,Lei Sun,Xiangxiang Chu,Anan Liu
### Background
文本到图像生成中的区域布局控制在实际应用中非常实用，但现有方法存在一些限制：（i）基于训练的方法会继承数据偏见，并且常常降低图像质量；（ii）目前的技术在处理遮挡顺序方面存在困难，这限制了其在现实世界中的使用效果。
### Innovation
本文提出了一种名为LayerBind的方法，通过将区域生成建模为不同的层次并在生成过程中结合的方式，使得区域和遮挡控制更为精准。LayerBind方法基于这样一种观察：空间布局和遮挡在早期内噪点阶段就已经确定，因此重新安排早期的潜在结构足以修改最终输出。该方法分为两个阶段：实例初始化和后续的语义护理。实例初始化利用多模态联合注意中的上下文共享机制，创建根据自身区域进行关注的分支，而这些分支在早期步骤根据层次顺序融合以形成具有预设布局的统一潜在空间；语义护理阶段通过逐层的注意力增强提高区域细节，保持遮挡顺序。LayerBind方法无需训练且即插即用，可作为扩散变换器中的区域和遮挡控制器。此外，它支持可编辑的工作流程，允许对实例进行更改或重新安排可见顺序。
### Conclusion
实验证明，LayerBind 方法在质量上非常有效，同时展示了其在创意应用方面的强大潜力。
## 18. `cs.CV` - 视觉单词遇见BM25：基于稀疏自编码器的视觉单词评分在图像检索中的应用 [PDF](https://arxiv.org/pdf/2603.05781), [HTML](https://arxiv.org/abs/2603.05781)
### Authors
Donghoon Han,Eunhwan Park,Seunghyeon Seo
### Background
密集图像检索尽管准确但缺乏可解释性和可归因性，并且在大规模环境下计算密集。对于大型图像库，视觉单词文档频率分布极不均匀，遵循Zipfian型分布。
### Innovation
介绍了BM25-V方法，该方法将Okapi BM25评分应用于稀疏自编码器（SAE）在Vision Transformer补丁特征上的稀疏视觉单词激活。BM25-V作为一种有效的第一阶段检索器用于密集排序，能够在不进行微调的情况下将预训练的SAE在ImageNet-1K上零样本迁移到七个细粒度基准上，且检索决策可具体归因于特定视觉单词。
### Conclusion
BM25-V在七个基准上实现了@200召回率≥0.993，能够在一个查询中仅重新排序200个候选者，平均恢复近密集准确率在0.2%以内。
## 19. `cs.CV` - 2D-to-3D场景重建中特征上采样的频谱探针 [PDF](https://arxiv.org/pdf/2603.05787), [HTML](https://arxiv.org/abs/2603.05787)
### Authors
Ling Xiao,Yuliang Xiu,Yue Chen,Guoming Wang,Toshihiko Yamasaki
### Background
传统的2D到3D管道采用多视角图像作为输入，通过Vision Foundation Model（VFM）提取特征并上采样为密集表示以实现3D重建。如果不同视角的密集特征保持几何一致性，通过对这些特征进行差分渲染可以恢复准确的3D表示，因此特征上采样器成为关键组件。最近的可学习上采样方法主要集中在增强空间细节，例如更锐利的几何形状或更丰富的纹理，但它们对3D感知的影响尚未被充分探索。
### Innovation
本文引入了一种包含六个互补度量的频谱诊断框架，这些度量用于表征幅值重分布、结构频谱对齐以及方向稳定性。在使用CLIP和DINO骨干网络的经典插值方法和可学习的上采样方法中，观察到三个关键发现：1. 结构频谱一致性（SSC/CSC）是神经视觉评分（NVS）质量的最强预测指标，而高频频谱坡度漂移（HFSS）通常与重建性能呈负相关，这意味着仅强调高频细节并不一定能够改善3D重建；2. 几何形状和纹理响应不同的频谱特性：角度能量一致性（ADC）与几何相关度量的关联更为强烈，而SSC/CSC在纹理保真度方面略胜一筹；3. 网络学习的上采样器通常能够产生更锐利的空间特性，但它们在重建质量上往往不及经典插值方法，其有效性取决于重建模型。
### Conclusion
本文的结果表明，重建质量与保持频谱结构密切相关，而不仅仅是增强空间细节。这强调了频谱一致性的原则在设计2D到3D管道中的上采样策略时的重要性。
## 20. `cs.CV` - TumorChain: 交错的多模态链式思考推理以实现可追踪的临床肿瘤分析 [PDF](https://arxiv.org/pdf/2603.05867), [HTML](https://arxiv.org/abs/2603.05867)
### Authors
Sijing Li,Zhongwei Qiu,Jiang Liu,Wenqiao Zhang,Tianwei Lin,Yihan Xie,Jianxiang An,Boxiang Yun,Chenglin Yang,Jun Xiao,Guangyu Guo,Jiawen Yao,Wei Liu,Yuan Gao,Ke Yan,Weiwei Cao,Zhilin Zheng,Tony C. W. Mok,Kai Cao,Yu Shi,Jiuyu Zhang,Jian Zhou,Beng Chin Ooi,Yingda Xia,Ling Zhang
### Background
准确的肿瘤分析是临床放射学和精准肿瘤学的核心。早期发现、可靠的病灶特征描述以及病理级别的风险评估会指导诊断和治疗规划。链式思考（CoT）推理在这个背景下尤为重要，因为它可以实现从影像发现到临床印象再到病理结论的逐步解释，从而提高可追溯性并降低诊断错误率。该研究针对临床肿瘤分析任务，构建了一个大规模基准，集成了跨模态推理管道，包括发现、印象和病理预测。研究目的是创建TumorCoT数据集，并提出TumorChain框架以实现跨模态对齐和迭代交互式因果推理，达到增强视觉证据的结合、结论的汇总及多轮自我完善，以提高可追溯性并降低幻觉风险。
### Innovation
研究通过构建TumorCoT数据集，提出了TumorChain框架，该框架将3D成像编码、临床文本理解和器官级别的视觉语言对齐紧密结合起来。通过跨模态对齐和迭代交互式因果推理，TumorChain在病灶检测、印象生成和病理分类等方面表现出一致的改进，且在DeepTumorVQA基准测试中显示出强大的泛化能力。结果表明，多模态推理在临床肿瘤分析中具有可靠性和可解释性的潜力。
### Conclusion
研究表明，多模态推理对于临床实践中的可靠和可解释的肿瘤分析具有潜在价值。在我们的项目主页上可以获得更详细的项目信息。TumorChain框架展示了在病灶检测、印象生成和病理分类方面的优势，并成功地在DeepTumorVQA基准上进行了验证，体现了其在临床肿瘤分析中的应用前景。
