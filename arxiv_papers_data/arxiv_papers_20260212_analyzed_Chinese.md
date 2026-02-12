# 20260212
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 从几何视角衡量数据集多样性 [PDF](https://arxiv.org/pdf/2602.09340), [HTML](https://arxiv.org/abs/2602.09340)
### Authors
Yang Ba,Mohammad Sadeq Abolhasani,Michelle V Mancenido,Rong Pan
### Background
多样性可以被广泛定义为数据集中存在有意义的变异，这些变异可以从多个角度进行观察，如统计变异和数据集的几何结构丰富度。现有的多样性度量标准，如特征空间散度和度量空间大小，主要捕捉分布的变化或熵，而忽略数据集的几何结构。
### Innovation
本文提出了一种基于拓扑数据分析（TDA）和持续景观（PLs）的框架来提取并量化数据的几何特征。该方法为衡量多样性提供了一种既立足于理论基础又能超越熵的手段，能够捕捉数据集丰富的几何和结构属性。
### Conclusion
通过在多种模态的广泛实验，我们证明了我们提出的基于持续景观的多样性度量（PLDiv）是强大的、可靠的、可解释的，并直接将数据集的多样性与基础几何联系起来，是一种构建、增强和评估数据集的基础工具。
## 2. `cs.AI` - CoMMa: 基于博弈论视角的认知贡献医学多智能体系统 [PDF](https://arxiv.org/pdf/2602.09159), [HTML](https://arxiv.org/abs/2602.09159)
### Authors
Yichen Wu,Yujin Oh,Sangjoon Park,Kailong Fan,Dania Daye,Hana Farzaneh,Xiang Li,Raul Uppot,Quanzheng Li
### Background
近年来，多智能体框架极大地扩展了处理涉及动态、异构患者数据的肿瘤学决策支持任务的能力。现有的多智能体框架大多数依赖于基于随机叙事的推理，这些方法在任务处理上存在一定的局限性。
### Innovation
本文提出了一种名为CoMMa的认知贡献医学多智能体系统，这是一种去中心化的LLM智能体框架。CoMMa通过博弈论目标使专家在分区证据上操作并进行协调，利用确定性的嵌入投影来估计每个智能体的边际效用，实现显性的证据归因，并生成可解释性强、数学上可靠并具有良好稳定性的决策路径。
### Conclusion
CoMMa被评估于多种肿瘤学基准测试中，包括现实世界的多学科肿瘤董事会数据集，结果表明，CoMMa在准确性和稳定性方面都优于基于数据集中化和基于角色的多智能体基础模型。
## 3. `cs.AI` - 通过Dirichlet参数化实现的不确定性感知多模态情绪识别 [PDF](https://arxiv.org/pdf/2602.09121), [HTML](https://arxiv.org/abs/2602.09121)
### Authors
Rémi Grzeczkowicz,Eric Soriano,Ali Janati,Miyu Zhang,Gerard Comas-Quiles,Victor Carballo Araruna,Aneesh Jonelagadda
### Background
当前，多模态情绪识别（MER）技术正逐渐应用于各种场景，如医疗保健、人机交互等领域。然而，多模态情绪识别面临的主要挑战包括如何有效处理不同模态之间的不确定性，以及如何在边缘设备上实现高效且私人的情绪识别。针对这些挑战，该研究提出了一种轻量级且隐私保护的多模态情绪识别框架。
### Innovation
该框架引入了一种基于Dempster-Shafer理论和Dirichlet证据的模型和任务通用融合机制，直接在模型逻辑上工作，通过捕获预测不确定性来应对不同模态之间的不确定性，避免了额外的训练和联合分布估计，使其在多模态情绪识别领域具有广泛应用的可能性。
### Conclusion
通过对五个基准数据集的验证，该方法在保持高效性和鲁棒性的同时取得了与现有方法相匹敌的准确性，强调了模块化、可扩展性和实际可行性，为未来的不确定性感知多模态系统在医疗保健、人机交互等方面的应用铺平了道路。
## 4. `cs.AI` - 审计多智能体LLM推理树优于多数投票和LLM作为仲裁者 [PDF](https://arxiv.org/pdf/2602.09341), [HTML](https://arxiv.org/abs/2602.09341)
### Authors
Wei Yang,Shixuan Li,Heng Ping,Peiyu Zhang,Paul Bogdan,Jesse Thomason
### Background
多智能体系统（MAS）能够显著扩展大型语言模型（LLMs）的推理能力。然而，大多数框架仍使用多数投票来聚合智能体输出，这种启发式方法牺牲了推理轨迹的证据结构，在共构一致（confabulation consensus）情况下尤其脆弱，即智能体共享相关偏见并达成相同的错误推理方案。
### Innovation
作者引入了AgentAuditor，它通过路径搜索方法在推理树上替代了投票制度，明确表示智能体中的意见一致与分歧。AgentAuditor通过比较关键分歧阶段的推理分支来解析冲突，将全局仲裁转换为高效的局部验证。此外，提出了Anti-Consensus Preference Optimization（ACPO），旨在通过训练仲裁器以反对大多数失败案例，奖励基于证据的少数正确选择，而非流行的错误。该方法对于多种智能体环境是通用的，并且在5种流行的设置中，AgentAuditor相比大多数投票和使用LLM作为仲裁者分别提供了高达5%和3%的绝对准确性提升。
### Conclusion
随着AgentAuditor的应用，它可以有效提高多智能体系统中大型语言模型的推理准确性，特别是在应对共构一致的问题上，表明审计多智能体LLM推理树优于传统的多数投票和LLM作为仲裁者的方法。
## 5. `cs.AI` - FlyAOC：评估关于果蝇科学知识库的主动本体构建 [PDF](https://arxiv.org/pdf/2602.09163), [HTML](https://arxiv.org/abs/2602.09163)
### Authors
Xingjian Zhang,Sophia Moylan,Ziyang Xiong,Qiaozhu Mei,Yichen Luo,Jiaqi W. Ma
### Background
科学知识库通过将主要文献中的发现整理为结构化的查询格式，以供人类研究者和新兴的人工智能系统使用，加速了发现过程。维持这些资源需要专业知识库整理员搜索相关论文、在文档间统一证据并生成基于本体的注释。现有的基准测试集中在孤立子任务上，如命名实体识别或关系提取，无法捕捉这个工作流程。该研究引入了FlyBench，旨在评估人工智能代理在从科学文献中构建整合作本体的能力，并通过仅给定一个基因符号，让代理搜索并阅读16,898篇全文论文以产生结构化注释：描述基因功能、表达模式和历史同义词的基因本体论术语，以连接数十年来的命名规范。
### Innovation
FlyBench是专门为评估人工智能代理在科学文献中进行整合作本体的能力而设计的基准测试。它通过给定基因符号和全文献语料库，要求代理生成结构化注释。此基准还涵盖了来自FlyBase（果蝇知识库）的7,397个专家注释，涉及100个基因，并评估了四种基线代理架构：记忆、固定管道、单代理和多代理。研究表明，架构选择对性能有显著影响，多代理设计胜过更简单的替代方案，但随着骨干模型的扩展，这种收益将逐渐降低。所有基线都存在改进空间。
### Conclusion
我们的分析揭示了一些指导未来发展的关键发现，例如，代理主要使用检索来确认参数化知识而非发现新信息。我们希望FlyBench能推动增强检索的科学推理进展，这种能力在科学领域具有广泛的应用前景。
## 6. `cs.AI` - 人工智能时代的影像质量 [PDF](https://arxiv.org/pdf/2602.09347), [HTML](https://arxiv.org/abs/2602.09347)
### Authors
Jana G. Delfino,Jason L. Granstedt,Frank W. Samuelson,Robert Ochs,Krishna Juluru
### Background
人工智能（AI）技术在放射学领域的应用正在以惊人的速度进行。AI已被证明是一种杰出的工具，能够重建和增强影像，使影像更清晰、平滑、详细，并且可以更快地获取，从而让临床医生能够更快地进行检查。然而，AI的应用也引入了新的失败模式，并可能加剧对影像感知质量与信息内容之间差距的认知。了解AI增强影像的局限性对于安全和有效地使用这项技术至关重要。
### Innovation
本文旨在提高对使用AI进行影像重建或增强时的局限性认识，从而允许用户从技术中获取好处同时最小化风险。
### Conclusion
理解AI增强影像的局限性对于其安全有效应用至关重要。通过本文，作者希望提高临床医生及其他影像用户对AI技术潜在风险的认识，从而在应用AI技术时更加谨慎和明智。
## 7. `cs.AI` - 人类控制并非答案：积极AI社区中的早期监督分歧 [PDF](https://arxiv.org/pdf/2602.09286), [HTML](https://arxiv.org/abs/2602.09286)
### Authors
Hanjing Shi,Dominic DiFranzo
### Background
该论文研究的是监督创新型人工智能（agentic AI）的目标应该是人类控制，但早期采用可能会产生特定角色的期望。文章通过比较2026年1-2月两个新活跃的Reddit社区——r/OpenClaw（部署和操作）和r/Moltbook（代理中心的社会互动），探讨了不同社会和技术角色下的监督期望。
### Innovation
研究通过主题建模、粗粒度的监督主题抽象、基于参与的显著性以及分歧测试，发现了两个社区在监督期望上的显著差异：r/OpenClaw更加注重执行限制和恢复（行动风险），而r/Moltbook则更侧重于身份、合法性以及公共互动中的问责（意义风险）。这为设计和评估匹配代理角色的监督机制提供了一个可移植的视角，而不是简单的统一控制政策。
### Conclusion
论文得出结论，
## 8. `cs.AI` - Not-in-Perspective：针对反向否定攻击保护谷歌言论API的方法 [PDF](https://arxiv.org/pdf/2602.09343), [HTML](https://arxiv.org/abs/2602.09343)
### Authors
Michail S. Alexiou,J. Sukarno Mertoguno
### Background
网络安全问题中的网络欺凌现象在社交媒体平台上日益严重，亟需有效的方法来监控和管理在线互动。目前基于机器学习或深度学习的自动毒性检测系统已经存在，但这些基于统计的方法容易受到含有逻辑修改，如否定词的对抗攻击。针对这一问题，研究提出了一种形式化推理为基础的防护策略，可以作为预处理和后处理步骤，缓解否定词攻击问题，并大幅提高毒性评分的准确性和有效性。
### Innovation
研究提出了一种形式化推理为基础的防护策略，可以在预处理和后处理步骤中作为传统的机器学习毒性检测系统的包裹层，以解决对抗攻击中的否定词问题，提升毒性检测的准确性和有效性。研究成果展示了混合方法（形式化推理和机器学习）在各种纯统计方法中的优越性。
### Conclusion
实验结果表明，基于形式化推理和机器学习的混合方法在处理对抗攻击中的否定词问题方面优于纯统计方法，显著提高了毒性检测的准确性和有效性。
## 9. `cs.AI` - PABU: 进度感知信念更新以提高LLM代理的效率 [PDF](https://arxiv.org/pdf/2602.09138), [HTML](https://arxiv.org/abs/2602.09138)
### Authors
Haitao Jiang,Lin Ge,Hengrui Cai,Rui Song
### Background
现有的大型语言模型（LLM）代理会基于完整的操作观察历史来条件化行动，这种方式引入了与任务无关的信息，容易导致冗余的行动和更高的推理成本。
### Innovation
提出了一种名称为PABU（Progress-Aware Belief Update）的信念状态框架，该框架通过明确建模任务进度并选择性保留过去的操作和观察，紧凑地表示代理的状态。每个步骤中，代理预测自上一个回合以来的相对进度，并决定新遇到的交互是否应该被存储，从而只在未来决策中考虑保留的部分。PABU在八个环境中的AgentGym基准测试中，以相同的训练轨迹实现了81.0%的任务完成率，超越了全历史信念的先前最佳模型23.9%。此外，PABU的进度导向行动选择提高了效率，减少了平均交互步骤到9.5次，对应减少了26.9%。
### Conclusion
剥落研究显示，显式的进度预测和选择性保留对于稳健的信念学习和性能提升都是必要的。
## 10. `cs.AI` - 小规模系统实现自回归程序合成：为可控实验提供支持 [PDF](https://arxiv.org/pdf/2602.09112), [HTML](https://arxiv.org/abs/2602.09112)
### Authors
Russ Webb,Jason Ramapuram
### Background
当前的研究通常依赖大规模的语言模型（LLMs）来进行程序合成研究，这带来了诸如分布内外问题、微调效果理解、分词影响以及实验对计算和存储需求高的问题。此类大型模型的研究成本和实施难度较大。
### Innovation
该论文提出了一种名为Cadmus的系统，包括一个整数虚拟机（VM）、一个多样任务的正确程序数据集和一个在不到200美元计算成本下训练的自动回归变压器模型。Cadmus系统能够研究程序补全、分布外表示、归纳推理和指令遵循，允许研究人员有效和经济地控制培训分布，并对模型进行检查和监控。这对于进行更大模型可能无法支持的复杂推理任务提供了可能性。
### Conclusion
这些小规模的Cadmus模型，在简单任务如用域特定语言（DSL）补全正确的整数算术程序方面即使比GPT-5表现出更高的准确性（100% vs GPT-5的95%）。此外，研究还展示了GPT-5在解决相同任务时引入了未知先验知识，这表明在某些需要完全理解训练集与任务关系的研究中，使用大规模LLMs会有混淆因素，进而影响研究结果。
## 11. `cs.CV` - Controllable Dance Generation with Style-Guided Motion Diffusion [PDF](https://arxiv.org/pdf/2406.07871), [HTML](https://arxiv.org/abs/2406.07871)
### Authors
Hongsong Wang,Ying Zhu,Xin Geng,Liang Wang
### Background
舞蹈作为一种艺术形式和表达方式，在人类文化中扮演着重要角色，但自动生成舞蹈序列既具挑战性又非常重要。现有方法往往忽视了舞蹈生成中控制这一关键因素，并且未能充分建模音乐风格的细微影响，导致生成的舞蹈与所条件的音乐的表达特性缺乏对齐。
### Innovation
本文提出了一种名为Style-Guided Motion Diffusion (SGMD)的技术，该技术结合Transformer架构和Style Modulation模块。通过将音乐特征与用户提供的风格提示相结合，SGMD确保生成的舞蹈不仅匹配音乐内容，还能反映所需的风格特征。此外，还引入了一种空间-时间掩码机制，以实现对生成舞蹈的灵活控制。
### Conclusion
广泛的实验表明，我们的方法可以生成逼真且风格一致的舞蹈，同时还能够使用户根据不同的艺术和实践需求来创作舞蹈。代码可在Github上获得: this https URL
## 12. `cs.CV` - 恒定速率调度：基于分布变化优化扩散噪声计划的通用框架 [PDF](https://arxiv.org/pdf/2411.12188), [HTML](https://arxiv.org/abs/2411.12188)
### Authors
Shuntaro Okada,Kenji Doi,Ryota Yoshihashi,Hirokatsu Kataoka,Tomohiro Tanaka
### Background
本文提出了优化扩散模型中噪声计划的通用框架，适用于训练和采样。该方法在整个扩散过程中限制概率分布的变化率，并使用用户定义的不一致性度量来量化变化率。文章引入了三种这样的度量标准，可以根据领域和模型架构进行灵活选择或组合。
### Innovation
提出了恒定速率调度（Constant Rate Scheduling）框架，通过控制概率分布的变化率来优化扩散过程中的噪声计划。框架利用理论洞察但不提供完整的理论证明，而是专注于建立一个通用调度框架，并通过实验证明其有效性。该方法能够提高不同数据集、不同采样器以及不同函数评估次数（从5到250）下的像素空间和潜在空间扩散模型的性能。
### Conclusion
通过广泛的实验，本文的方法在训练和采样过程中显示了其优势，特别是在LSUN Horse 256×256数据集上达到了最先进的FID分数2.03，同时保持了模式覆盖。
## 13. `cs.CV` - RS-Agent: 通过智能代理自动化遥感任务 [PDF](https://arxiv.org/pdf/2406.07089), [HTML](https://arxiv.org/abs/2406.07089)
### Authors
Wenjia Xu,Zijian Yu,Boyang Mu,Zhiwei Wei,Yuanben Zhang,Guangzuo Li,Jiuniu Wang,Mugen Peng
### Background
多模态大型语言模型（MLLMs）的快速发展表明了其在通过语言和视觉输入与人类交互以执行诸如视觉问答和场景理解等下游任务方面的巨大潜力。然而，这些模型受限于基础的指令遵循或描述性任务，难以应对需要专门工具和知识的复杂现实世界遥感应用。
### Innovation
我们提出了RS-Agent，一种为远程感应应用设计的AI代理，能够自主使用专门模型解决实际应用需求。RS-Agent集成了四个关键组件：基于大型语言模型的中央控制器、用于工具执行的动态工具包、任务特定的专家解决方案空间和领域级推理的知识空间，以解释用户查询和协调工具以实现准确的遥感任务。提出的创新机制包括任务感知检索，通过专家引导规划提高工具选择准确性，以及DualRAG检索增强生成方法，增强知识的相关性并通过加权、双路径检索。
### Conclusion
经过9个数据集和18个遥感任务的广泛实验，RS-Agent显著优于最先进的MLLMs，实现95%以上的任务规划准确率，并在场景分类、物体计数和遥感视觉问答等任务中表现出优越性能。我们的工作为提高遥感分析中的智能自动化提供了鲁棒且可扩展的框架。
## 14. `cs.CV` - 基于重建变换图像的自监督学习方法以实现一致协调的特征表示 [PDF](https://arxiv.org/pdf/2503.18753), [HTML](https://arxiv.org/abs/2503.18753)
### Authors
Qin Wang,Alessio Quercia,Benjamin Bruns,Abigail Morrison,Hanno Scharr,Kai Krajsek
### Background
当前的自监督学习（SSL）方法在学习具有不变性的图像表示方面取得了显著成功，但同时也忽略了某些计算机视觉任务所需的变换信息。尽管最近的一些方法试图通过在特征空间中使用线性算子学习不变的特征来解决这一局限性，但它们对这些特征施加了限制条件，这会限制灵活性和泛化能力。
### Innovation
本文提出了一种更弱的变换关系定义，称为一致性-协调性（equivariance-coherence），并提出了一种新颖的SSL辅助任务来学习一致性-协调的表示，该任务通过中间变换重建来实现。该方法将特征向量分解为不变部分和可变部分，并通过标准SSL损失和重建损失分别进行训练。实验结果显示，该方法在合成的可变性基准上取得了显著改进，同时保持了在需要不变表示的下游任务上的竞争力。
### Conclusion
该方法无缝地与现有SSL方法（iBOT, DINOv2）集成，并且在多种任务场景下，包括语义分割、物体检测、深度估计和视频密集预测中，都展示了持续的性能提升。框架为增强SSL方法的可变性能力并保持不变性性能提供了一种实用的方法。
## 15. `cs.CV` - 通过物体运动感知实现视觉注意力的一种仿生方法 [PDF](https://arxiv.org/pdf/2502.06747), [HTML](https://arxiv.org/abs/2502.06747)
### Authors
Giulia D'Angelo,Victoria Clerico,Chiara Bartolozzi,Matej Hoffmann,P. Michael Furlong,Alexander Hadjiivanov
### Background
活动视觉能够动态感知视觉，提供了一种替代静态前馈架构的方案。前馈架构依赖于大量数据集和高计算资源，生物学上的选择性注意力机制使得智能体能够聚焦于重要区域，减少计算需求，同时保持实时响应。由哺乳动物视网膜启发的事件驱动摄像头能够捕捉场景变化，促进低延迟处理。为了在移动中区分移动物体，智能体需要一个运动分割机制来准确检测目标并将其置于视野中心。结合事件驱动传感器和神经形态算法，在Spiking神经网络中并行计算和适应动态环境是一种范式转变。
### Innovation
本文提出了一种基于物体运动敏感性仿生注意力系统，该系统利用动态视觉传感器整合至Speck神经形态硬件中，探测区域兴趣并通过全景倾斜单元转向。该系统在理想条纹和基准测试中表现优秀，多目标运动分割平均交并比达到82.2%，平均结构相似性达到96%。在办公室场景和低光照环境中，通过事件辅助低光视频物体分割数据集检测显著物体的准确性分别达到88.8%和89.8%。实时演示表明系统对动态场景的响应时间为0.12秒，无学习设计使得其在感知场景中具有鲁棒性，能够作为实时机器人应用的基础。
### Conclusion
该系统能在移动物体中有效区分子物体，并具有快速响应能力和高准确性。它简化了复杂架构的基础，并提供了强大的实时视觉处理能力。
## 16. `cs.CV` - Dual-IPO: 双迭代偏好优化以实现文本到视频生成 [PDF](https://arxiv.org/pdf/2502.02088), [HTML](https://arxiv.org/abs/2502.02088)
### Authors
Xiaomeng Yang,Mengping Yang,Jia Gong,Luozheng Qin,Zhiyu Tan,Hao Li
### Background
近年来，视频生成技术取得了显著进步，能够产生逼真的视频，但通常无法完全满足用户的真实需求和偏好，生成不满意、不一致的输出。
### Innovation
本文提出了双迭代优化（Dual-IPO）方法，这是一种逐步优化奖励模型和视频生成模型的迭代框架，以提高合成质量和更好的符合人类偏好，通过CoT引导的推理、基于投票的自我一致性以及偏好确定性估计确保奖励信号的可靠性和鲁棒性。
### Conclusion
全面的实验表明，所提出的Dual-IPO方法能够显著提升具有不同架构和规模的基础模型的视频生成质量，甚至帮助参数为2B的模型超越参数为5B的模型。此外，我们的系统设计分析和消融实验揭示了每个组件的有效性，并证明了该方法的有效性。
## 17. `cs.CV` - Story-Iter: 一种无需训练的长故事可视化迭代范式 [PDF](https://arxiv.org/pdf/2410.06244), [HTML](https://arxiv.org/abs/2410.06244)
### Authors
Jiawei Mao,Xiaoke Huang,Yunfei Xie,Yuanqi Chang,Mude Hui,Bingjie Xu,Zeyu Zheng,Zirui Wang,Cihang Xie,Yuyin Zhou
### Background
现有的长故事生成方法依赖固定参考图像来构建完整的故事，但这些方法存在一些局限性。Story-Iter 提供了一种新的训练-free 迭代范式，旨在增强长故事生成。与传统方法不同，它通过迭代融合所有参考图像的信息，逐步细化生成的每一幅图像，确保故事的语义一致性。
### Innovation
Story-Iter 引入了一个无需训练的全局参考交叉注意力 (GRCA) 模块，能够将所有参考图像使用全局嵌入进行建模，以确保整个长序列中的语义一致性。这种方法能够逐步融合整体的视觉上下文和文本约束，从而实现精确生成并优化故事可视化。
### Conclusion
实验结果表明，Story-Iter 在长故事可视化方面达到了最先进的性能（最多可达 100 帧），不仅在语义一致性方面表现出色，还在细节互动方面更具优势。
## 18. `cs.CV` - 基于图的多模态多视角对准在关键步识别中的应用 [PDF](https://arxiv.org/pdf/2501.04121), [HTML](https://arxiv.org/abs/2501.04121)
### Authors
Julia Lee Romero,Kyle Min,Subarna Tripathi,Morteza Karimzadeh
### Background
自中心视角视频记录佩戴者视角下的场景，导致背景动态、频繁运动和遮挡，这给准确的关键步识别带来了挑战。
### Innovation
提出了一种灵活的图学习框架，该框架能够有效地利用自中心视角视频中的长期依赖关系，并在训练期间利用自中心视角和外中心视角视频之间的对齐，从而在自中心视角视频的推理上取得更好的效果。该框架通过构建节点对应视频片段的图形，并通过多种策略定义这些节点之间的连接，将关键步识别转化为在构建的图上的节点分类任务。
### Conclusion
在Ego-Exo4D数据集上进行了广泛的实验，表明所提出的灵活的基于图的框架的准确率显著优于现有方法，高出12个百分点以上。此外，构建的图形稀疏且计算效率高。还研究了利用包括叙述、深度和对象类别标签在内的多种多模态特征在异构图上的应用，并讨论了它们对关键步识别性能的贡献。
## 19. `cs.CV` - 无条件先验很重要！提高微调扩散模型的条件生成 [PDF](https://arxiv.org/pdf/2503.20240), [HTML](https://arxiv.org/abs/2503.20240)
### Authors
Prin Phunyaphibarn,Phillip Y. Lee,Jaihoon Kim,Minhyuk Sung
### Background
Classifier-Free Guidance (CFG) 是训练条件扩散模型的基本技术。通常的做法是使用一个网络同时学习条件和无条件的噪声预测，采用少量的条件滴落率。然而，观察到在训练中无条件噪声的学习由于带宽有限而表现不佳，这导致无条件的噪声预测质量差。最重要的是，这些质量差的无条件噪声预测严重影响了有条件生成的质量。
### Innovation
文章提出了一种新的方法，通过将无条件噪声替换为由基础模型预测的噪声来显著提高条件生成。此外，文章表明即使不同训练的扩散模型也可以用于无条件噪声的替换，而不仅仅是微调模型训练时的模型。文章通过多种基于CFG的图像和视频生成模型进行了实验验证，包括Zero-1-to-3、Versatile Diffusion、DiT、DynamiCrafter和InstructPix2Pix。
### Conclusion
无条件生成的质量对条件生成结果有重大影响。通过改进无条件噪声的预测，可以显著提升模型的条件生成性能。
## 20. `cs.CV` - 在遮挡条件下评估3D人体姿态估计模型 [PDF](https://arxiv.org/pdf/2504.10350), [HTML](https://arxiv.org/abs/2504.10350)
### Authors
Filipa Lino,Carlos Santiago,Manuel Marques
### Background
3D人体姿态估计(HPE)涉及从视觉数据中检测和定位人体关键点。在3D HPE中，遮挡（身体的部分在图像中不可见）是准确重建姿态的主要挑战。本文旨在在现实世界遮挡条件下评估当前3D HPE模型的鲁棒性，使用BlendMimic3D合成数据集，该数据集包含真实可见点和遮挡标签，并针对现实世界场景中常见遮挡情况进行了模型集成。
### Innovation
本文通过引入一种基于真实检测器行为模拟遮挡的协议，评估了九种最先进的2D到3D HPE模型在BlendMimic3D数据集上的鲁棒性。这些模型涵盖卷积、基于变压器、基于图和基于扩散的架构，并在无需重新训练的情况下评估它们的泛化能力。此外，本文还进行了整体和每个关节的敏感性分析，揭示了所有模型在遮挡下的性能下降情况，尤其是基于扩散的模型（尽管它们具有随机性）和对末梢关节（如手腕、脚踝）的一致性脆弱性。
### Conclusion
本文突出了当前3D HPE模型在处理遮挡方面的关键局限性，并为进一步提高现实世界鲁棒性提供了见解。
