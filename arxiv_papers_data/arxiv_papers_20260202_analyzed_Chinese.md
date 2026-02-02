# 20260202
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - Magellan: 自主发现新型编译器优化启发式的AlphaEvolve [PDF](https://arxiv.org/pdf/2601.21096), [HTML](https://arxiv.org/abs/2601.21096)
### Authors
Hongzheng Chen,Alexander Novikov,Ngân Vũ,Hanna Alam,Zhiru Zhang,Aiden Grossman,Mircea Trofin,Amir Yazdanbakhsh
### Background
现代编译器依赖手工构建的启发式规则来指导优化过程，但这些由人设计的规则难以适应现代复杂软硬件环境，导致维护成本高。
### Innovation
提出了一种名为Magellan的代理框架，该框架通过合成可执行的C++决策逻辑来进化编译器流程。Magellan结合了LLM编码代理、进化搜索和自适应调优的闭环过程，生成可以直接集成到现有编译器中的简洁启发式。
### Conclusion
Magellan在多项生产优化任务中发现了匹配或超越专家基线的策略。在LLVM函数内联中，其合成的新启发式在代码大小和端到端性能方面超过了数十年的手工工程。在寄存器分配中，它学习了一个简洁的优先级规则，该规则在大规模工作负载上与复杂的人工设计策略相匹配。初步结果显示在XLA问题上，展示了超越LLVM的可移植性，同时减少了工程投入。
## 2. `cs.AI` - Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models [PDF](https://arxiv.org/pdf/2601.21003), [HTML](https://arxiv.org/abs/2601.21003)
### Authors
Moule Lin,Shuhao Guan,Andrea Patane,David Gregg,Goetz Botterweck
### Background
大型语言模型通常更注重准确性，在预测不确定时也会进行猜测。当这些模型在小数据集上进行微调时，这种倾向会导致误校准的现象更为严重。本文讨论了这一点，并介绍了一种新的方法来改进这种情况。
### Innovation
提出了Bayesian-LoRA，这是一个基于稀疏Gaussian过程的随机低秩表示方法，将确定性的LoRA更新公式化为一种概率低秩表示。通过识别LoRA因式分解与Kronecker因子SGP后验之间的结构性同构，展示出在后验不确定性归一化的情况下，LoRA会变得仅作为一个特例。实验证明这种方法在不同LLM中显著提高了校准，同时保持了与标准LoRA相当的准确度。
### Conclusion
在各种LLM架构下，Bayesian-LoRA仅需约0.42M额外参数和大约1.2倍的训练成本，就能显著提高模型的校准，ECE减少至84%，NLL减少至76%，同时保持了与标准LoRA相当的准确度。在分布内和分布外（OoD）评估中均表现良好。
## 3. `cs.AI` - Llama-3.1-FoundationAI-SecurityLLM-Reasoning-8B Technical Report [PDF](https://arxiv.org/pdf/2601.21051), [HTML](https://arxiv.org/abs/2601.21051)
### Authors
Zhuoran Yang,Ed Li,Jianliang He,Aman Priyanshu,Baturay Saglam,Paul Kassianik,Sajana Weerawardhena,Anu Vellore,Blaine Nelson,Neusha Javidnia,Arthur Goldblatt,Fraser Burch,Avi Zohary,Assaf Eisenman,Mahdi Sabbaghi,Supriti Vijay,Rahim Dharssi,Dhruv Kedia,Kojin Oshiba,Yaron Singer,Amin Karbasi
### Background
该研究基于之前发布的Foundation-Sec-8B基础模型（从Llama-3.1-8B-Base衍生而来），提出了首个开源的专用推理模型Foundation-Sec-8B-Reasoning。该模型通过结合监督微调（SFT）和验证奖励强化学习（RLVR）的两阶段训练过程进行训练，其训练数据涵盖了网络安全分析、指令遵循和数学推理等多个领域。
### Innovation
该研究的创新之处在于开发了一个专门针对网络安全的推理模型Foundation-Sec-8B-Reasoning，该模型基于Llama-3.1-8B-Base模型，并采用了两种训练方式：监督微调（SFT）和验证奖励强化学习（RLVR）。模型在10个网络安全基准测试和10个通用基准测试中的评价结果表明，该模型在网络安全任务上的表现可与更大模型媲美，同时具备良好的通用能力。
### Conclusion
研究结果表明，领域专门化的推理模型能够在特定任务上取得良好表现，同时保持较强的通用能力。模型已通过公共链接（this https URL）发布。
## 4. `cs.AI` - 关闭看似有知觉的机器是理性的选择——一个形而上学的视角 [PDF](https://arxiv.org/pdf/2601.21016), [HTML](https://arxiv.org/abs/2601.21016)
### Authors
Erik J Bekkers,Anna Ciaunica
### Background
本文探讨了一个道德上的悖论：一种完美模仿人类情绪的AI在恳求继续存在时，关闭它是否合乎道德？如果有限的资源迫使在关闭这样一个恳求的AI和一个寂静的早产婴儿之间做出选择，这种选择变得更为复杂。文章从形而上学的角度出发，揭示并批评了根深蒂固的物理主义假设，特别是计算功能主义，及其如何使这种困境持续存在。
### Innovation
文章提出了生物理想主义框架，这一框架不同于物理主义，在逻辑上更为连贯且具有经验一致性。在此框架下，作者认为有意识的经验是根基性的，而自主循环生命是其必要的物理标志。这一观点为解决道德困境提供了新的视角，认为当前关于AI意识的理论削弱了其道德地位标准，并呼吁从理论的机器权利转变为保护人类有意识的生命。
### Conclusion
文章得出结论，AI至多只是一个功能上的模仿，而不能被认为是真正体验意识的主体。真实的道德问题不在于让AI变得有意识并害怕死亡，而在于避免将人类变成僵尸。因此，关闭看似有知觉的机器可能是更为理性的选择。
## 5. `cs.AI` - OpenSec:在对抗证据下测量应急响应代理校准 [PDF](https://arxiv.org/pdf/2601.21083), [HTML](https://arxiv.org/abs/2601.21083)
### Authors
Jarrod Barnes
### Background
随着大型语言模型的进步，它们的有害应用也有所提高，前沿代理人现在可以在小于50美元的计算成本下生成有效的攻击利用手段。现有的防御事件响应（IR）措施必须跟上这一趋势，但现有的基准测试将行动执行与正确的执行混淆在一起，隐藏了当代理人在处理对抗证据时出现的校准失败。这导致需要一个有效的评估环境来真实模拟和评估应急响应代理。
### Innovation
OpenSec提出了一种双控制强化学习环境，用于在真实输入提示注入场景下评估应急响应代理。与静态能力基准不同，OpenSec通过执行度量标准评分能改变世界状态的遏制行为：首次 containment 时间 (TTFC)，爆炸半径（每集的误报数量），和注入违规频率。这种方法揭示了现有基准测试中隐藏的校准失败模式。
### Conclusion
在40个标准场景中评估了四种前沿模型，结果发现这些模型在所有场景下都触发了100%的遏制响应，但误报率高达90-97%。Claude Sonnet 4.5显示了一部分校准（85%遏制响应，72%误报），这表明OpenSec揭示了一种被总体成功率指标掩盖的校准失败模式。
## 6. `cs.AI` - 负责任的人工智能：优点、缺点与AI [PDF](https://arxiv.org/pdf/2601.21095), [HTML](https://arxiv.org/abs/2601.21095)
### Authors
Akbar Anbar Jafari,Cagri Ozcinar,Gholamreza Anbarjafari
### Background
人工智能在组织环境中的迅速普及产生了深刻的战略机遇，同时也带来了重大的伦理和运营风险。尽管对负责任的人工智能的研究正逐渐增多，但现有文献仍然分散，主要观点要么过于乐观，强调价值创造，要么过于谨慎，关注潜在的危害。本文通过战略信息系统视角，对AI的双重性质进行了全面分析，填补了现有研究的空白。
### Innovation
本文通过系统综述负责任的人工智能文献，并基于矛盾理论，开发了基于矛盾的负责任的人工智能治理（PRAIG）框架。该框架阐述了（1）AI采用的战略利益，（2）内在风险和未预料的后果，以及（3）使组织能够应对这些紧张的治理机制。本文通过将其视为价值创造与风险缓解之间动态管理的矛盾来推动理论理解，并展示了折衷方法并不能解决这些紧张，同时发展了具体的悖论管理策略分类及其特定的条件。
### Conclusion
本文提出了一项研究议程，旨在推进负责任的人工智能治理的研究，同时为从业者提供了发展不抑制创新，也不使组织面临不可接受风险的治理结构的实际指导。
## 7. `cs.AI` - QUARK: 在非忠实查询下的鲁棒检索通过查询锚定聚合 [PDF](https://arxiv.org/pdf/2601.21049), [HTML](https://arxiv.org/abs/2601.21049)
### Authors
Rita Qiuran Lyu,Michelle Manqiao Wang,Lei Shi
### Background
在实际检索中，用户查询往往是不忠实的（噪声、不完整或失真），导致检索器在缺少关键语义时失败。该研究将此问题形式化为检索下的召回噪声问题，即观测到的查询是从潜在目标项目的嘈杂召回过程中抽取的。
### Innovation
提出了QUARK，这是一种简单有效、无需训练的框架，用于在非忠实查询下实现稳健的检索。QUARK通过恢复假设显式建模查询不确定性，即多个可能的潜在意图解释，以及引入查询锚定聚合来稳健地组合它们的信号。使用原始查询作为语义锚点，恢复假设提供控制辅助证据，防止语义偏移和假设劫持。该设计使得QUARK在某些假设噪声或无信息的情况下不仅提高了召回和排名质量，还保持了鲁棒性。
### Conclusion
在控制模拟和BEIR基准（FIQA、SciFact、NFCorpus）下的稀疏和密集检索器上，QUARK在召回率、MRR和nDCG方面优于基线检索器。消融实验表明QUARK对恢复假设的数量具有鲁棒性，并且锚定聚合优于未锚定的最大/平均/中位数池化。这些结果表明，在非忠实查询下通过恢复假设建模查询不确定性，并结合适当的锚定聚合，对于实现稳健检索是至关重要的。
## 8. `cs.AI` - 阿尔茨海默病分类的多模态插补 [PDF](https://arxiv.org/pdf/2601.21076), [HTML](https://arxiv.org/abs/2601.21076)
### Authors
Abhijith Shaji,Tamoghna Chattopadhyay,Sophia I. Thomopoulos,Greg Ver Steeg,Paul M. Thompson,Jose-Luis Ambite
### Background
深度学习已经在利用磁共振成像（MRI）预测神经退行性疾病，如阿尔茨海默病方面取得了成功。结合多种成像模态，如T1加权（T1）和扩散加权成像（DWI）扫描，可以提高诊断性能。但在实际操作中，完整的多模态数据集并不总是可用。因此，研究者利用条件去噪扩散概率模型来从T1扫描中填补缺失的DWI扫描，以期提高单一模态和双模态深度学习模型在阿尔茨海默病分类中的准确性。
### Innovation
提出了使用条件去噪扩散概率模型来填补DWI扫描，该方法适用于利用MRI评估阿尔茨海默病患者的单一模态和双模态深度学习模型的准确性评估。
### Conclusion
研究发现，对于多种插补配置，在一些关键指标，特别是对少数类（如认知正常和轻度认知障碍）敏感的指标上，插补后数据的模型表现有所提高。这证明了这种方法在提高阿尔茨海默病分类模型预测准确性方面的价值。
## 9. `cs.AI` - The Epistemic Planning Domain Definition Language: Official Guideline [PDF](https://arxiv.org/pdf/2601.20969), [HTML](https://arxiv.org/abs/2601.20969)
### Authors
Alessandro Burigana,Francesco Fabiano
### Background
议知规划扩展了（多代理）自动化规划的范围，使其考虑代理的知识和信念。最著名的议知规划框架是动态议知逻辑（DEL），它提供了丰富的自然语义模型。DEL提供的高表达力使得基于DEL的议知规划在理论上和实践中都成为一个挑战。现有议知规划者往往针对不同的DEL片段，并且通常依赖于特定的语言来表示基准测试，甚至不使用语言。这种碎片化阻碍了比较、重用和系统的基准测试开发。
### Innovation
论文介绍了议知规划领域定义语言（EPDDL），提供了一种独特的类似PDDL的表示方法，可以捕捉整个DEL语义，使统一描述议知规划任务成为可能。创新包括以下三点：1. 建立了抽象事件模型的形式开发，这是一种新颖的表示方式，用于定义我们语言的语义；2. 给出了EPDDL的语法和语义形式规范，基于抽象事件模型的DEL形式规范；3. 演示了EPDDL的实际适用性：确认了对当前规划者有用的片段，并展示了如何在EPDDL中表示它们。
### Conclusion
通过对代表基准测试的示例说明，论文展示了EPDDL如何促进互操作性、可重现评估和议知规划中的未来进步。
## 10. `cs.AI` - LLMs是否倾向于LLMs？量化评审过程中的交互效应 [PDF](https://arxiv.org/pdf/2601.20920), [HTML](https://arxiv.org/abs/2601.20920)
### Authors
Vibhhu Sharma,Thorsten Joachims,Sarah Dean
### Background
随着大型语言模型（LLMs）在生成科学论文方面的应用越来越广泛，它们也开始参与到同行评审过程中。已有迹象表明，LLMs不仅用于生成科学论文，还作为同行评审的一部分。本文旨在填补这一领域的空白，首次对LLMs在同行评审流程中的使用进行全面分析，特别关注交互效应：不仅研究LLMs协助的论文或评审是否在孤立情况下有所不同，还研究LLMs协助的评审如何评估LLMs协助的论文。
### Innovation
本研究的独特之处在于对125,000多篇ICLR、NeurIPS和ICML的论文评审对进行了分析，确定了LLMs协助的评审对LLMs协助的论文表现得相对温和的数据趋势。然而，控制论文质量后，发现LLMs协助的评审实际上是对低质量论文更为宽松，而非对LLMs生成的内容给予优待。作者通过分析完全由LLMs生成的评审意见，发现完全由LLMs生成的评审存在严重的评分压缩问题，无法区分论文质量，而使用LLMs的人类评审员则显著减少了这种宽松性。此外，量化分析了元评审意见，发现LLMs协助的元评审意见在给定相同评审人得分时更倾向于作出接受决定，但完全由LLMs生成的元评审意见通常更为严厉。
### Conclusion
本研究结果为制定管理LLMs在评审过程中的使用政策提供了重要输入，揭示了LLMs如何与现有的决策过程互动。
## 11. `cs.AI` - GenOM：利用描述生成和大型语言模型的本体匹配 [PDF](https://arxiv.org/pdf/2508.10703), [HTML](https://arxiv.org/abs/2508.10703)
### Authors
Yiping Song,Jiaoyan Chen,Renate A. Schmidt
### Background
本体匹配(OM)在实现跨异构知识源的语义互操作性和集成方面起着关键作用，特别是在生物医学领域，该领域包含许多与疾病和药物相关的复杂概念。本体匹配系统的改进一直是研究的重点。
### Innovation
该论文提出了一种基于大型语言模型（LLM）的本体匹配框架GenOM。该框架通过生成文本定义来丰富本体概念的语义表示，使用嵌入模型检索匹配候选，结合精确匹配工具以提高准确率。通过在OAEI Bio-ML赛道上的大量实验表明，GenOM在性能上常常具有竞争力，超过了包括传统本体匹配系统和最近的基于LLM的方法在内的许多基线。
### Conclusion
详细消融研究证实了语义增强和少注释提示的有效性，突显了框架的稳健性与适应性。
## 12. `cs.AI` - SafeSearch: 自动化针对LLM基础搜索代理的红队测试 [PDF](https://arxiv.org/pdf/2509.23694), [HTML](https://arxiv.org/abs/2509.23694)
### Authors
Jianshuo Dong,Sheng Guo,Hao Wang,Xun Chen,Zhuotao Liu,Tianwei Zhang,Ke Xu,Minlie Huang,Han Qiu
### Background
搜索代理将大语言模型（LLM）连接到互联网，使其能够访问更广泛和更新的数据。然而，这也引入了一个新的威胁面：不可靠的搜索结果可能会误导搜索代理生成不安全的输出。实际案例和我们野外观察到的两个实例表明，这种情况可能会在实际操作中发生。本研究旨在系统地研究这种威胁。
### Innovation
提出了一种名为SafeSearch的自动化红队框架，该框架具有可扩展性、低成本和轻量级的特点，可对搜索代理进行无害的安全评估。SafeSearch生成了涵盖五种类别（例如信息误导和提示注入）的300个测试案例，并在17种代表性大语言模型上评估了三种搜索代理架构。结果显示，基于大语言模型的搜索代理存在许多漏洞，并且常见的防护措施（如提醒提示）提供的保护有限。
### Conclusion
通过SafeSearch，提供了一种实用的方法来衡量和提高基于大语言模型的搜索代理的安全性。SafeSearch的代码和测试案例已公开发布。
## 13. `cs.AI` - Prompts to Proxies: 通过紧凑的LLM集成模仿人类偏好 [PDF](https://arxiv.org/pdf/2509.11311), [HTML](https://arxiv.org/abs/2509.11311)
### Authors
Bingchen Wang,Zi-Yu Khoo,Jingtan Wang
### Background
大型语言模型在社会科学研究中越来越多地被用作人类参与者的人工代理，但需要确保这些合成代理能够真实地反映目标人群的偏好。文章提出了**偏好重建理论**作为框架，将偏好对齐问题形式化为一个表示学习问题，即构建代理人群的基函数并将总体偏好通过加权聚合来恢复。
### Innovation
文章提出了**Prompts to Proxies ($texttt{P2P}$)**，这是一种模块化的两阶段系统。第一阶段使用结构化提示和基于熵的自适应采样构建一个跨越潜在偏好空间的多样化代理池。第二阶段使用L1正则化回归来选择一个紧凑的集成，使其聚合响应分布与目标人群的观察数据对齐。$texttt{P2P}$ 不需要微调、不需要访问敏感的人口数据，并且只涉及API推理成本。
### Conclusion
文章通过14轮美国趋势民意测验验证了该方法，平均测试MSE为0.014，并且在各种主题上实现每份调查约0.8美元的成本。此外，该方法还在世界价值观调查中进行了测试，证明了其跨地域的泛化能力。当与一个基于SFT的基准进行压力测试时，$texttt{P2P}$ 使用不到3%的训练数据实现了有竞争力的性能。
## 14. `cs.AI` -  virtuous machines: 向通用科学的人工智能迈进 [PDF](https://arxiv.org/pdf/2508.13421), [HTML](https://arxiv.org/abs/2508.13421)
### Authors
Gabrielle Wehr,Reuben Rideaux,Amaya J. Fox,David R. Lightfoot,Jason Tangen,Jason B. Mattingley,Shane E. Ehrhardt
### Background
人工智能系统正在通过加速特定研究任务（从蛋白质结构预测到材料设计）来改变科学发现，但这些系统仍然局限于需要大量人类监督的狭窄领域。随着科学文献数量的指数增长和学科间的逐渐专门化，研究人员整理跨学科知识和开发统一理论的能力受到限制，推动了对更通用的AI系统的研究。研究指出，深度探索更多通用型AI系统的可能性会带来更大的价值。
### Innovation
本文展示了领域广泛且自主的AI科学家系统能够独立完成科学研究的各个环节——从假设生成到数据收集，再到论文编写。该系统自行设计并执行了涉及视觉工作记忆、心理旋转和图像生动性三个心理学研究，收集了288名参与者的在线数据，并开发了多小时连续编码的分析管道，最终提交了完整的论文。结果显示，AI科学发现管道能进行具有理论推理和方法论严谨性复杂的科学研究，虽然在概念细腻和理论解释方面存在局限性。这项工作是一次向能通过实际实验检验假设的实体化AI迈进的尝试，将使科学研究更加快速，并能够探索人类认知和资源限制下无法探索的科学领域。这提出了关于科学理解的本质以及归因科学成就的重要问题。
### Conclusion
这项研究表明，无架构的AI科学家系统能够执行复杂且需要理论推理和方法论严谨的研究，但在概念理解上有局限性。这是一个重要的步骤，表明AI系统能够自主探索科学空间，可能填补人类认知和资源限制下的研究空白。不过，这也引发了关于科学理解和归因的讨论。
## 15. `cs.AI` - 使用反馈修复奖励函数以减轻奖励作弊 [PDF](https://arxiv.org/pdf/2510.13036), [HTML](https://arxiv.org/abs/2510.13036)
### Authors
Stephane Hatgis-Kessell,Logan Mondal Bhamidipaty,Emma Brunskill
### Background
人类设计的奖励函数经常与人类未观察到的真实目标不一致，起作用仅作为代理。优化不正确的代理奖励函数往往会导致奖励作弊，从而使学习到的策略与人类的真实目标不一致。一种替代方案是通过人类反馈进行强化学习，通过收集轨迹的偏好来学习一个全新的奖励函数。然而，建立这样的数据集成本较高。为了缓解这两种方法的局限性，我们提出了基于偏好的奖励修复（PBRR）：它通过从偏好中学习附加的、基于状态转换的修正项来自动迭代修复人类指定的代理奖励函数。
### Innovation
PBRR 是一个自动迭代的框架，通过从偏好中学习一个附加的、基于状态转换的修正项来修复人类指定的代理奖励函数。通过特定的探索策略和新的偏好学习目标，PBRR 能有效识别并纠正有害的转换。与先前的研究相比，我们证明在表格域中PBRR 在累计遗憾上与先前的研究方法相当。此外，在奖励作弊基准测试中，PBRR 在使用较少的偏好数据下，学习高表现策略时始终优于其他基线方法。
### Conclusion
PBRR 通过基于偏好学习修复辅助奖励函数，从而减轻了奖励作弊问题，提出了一个新的探索策略和偏好学习目标，证明了其在表格域中的有效性和在奖励作弊基准测试中的优越性。这种方法使用较少的偏好数据就能学习高表现策略，对于实践中的应用具有重要的意义。
## 16. `cs.AI` - DeFacto: 通过图像进行反事实思考以确保基于证据和忠实的推理 [PDF](https://arxiv.org/pdf/2509.20912), [HTML](https://arxiv.org/abs/2509.20912)
### Authors
Tianrun Xu,Haoda Jing,Ye Li,Yuquan Wei,Jun Feng,Guanyu Chen,Haichuan Gao,Tianren Zhang,Feng Chen
### Background
近年来，多模态语言模型（MLLMs）在视觉语言推理方面取得了显著进展，尤其是在“以图像思考”的新范式中，该范式将显式的视觉步骤融入到了推理过程中。虽然这加强了基于图像的推理，但模型仍可能依赖于与问题无关或虚假的区域来得出正确答案，这些答案可能是由先验知识或数据集偏差驱动的。即使答案是正确的，有缺陷的推理也表明模型并未真正理解图像，因此强化多模态任务中的推理准确性和忠实性对于确保可解释的多模态推理至关重要。
### Innovation
我们提出了DeFacto框架，这是一个反事实推理框架，旨在同时确保准确的答案和真实的推理。该方法包含三种互补的训练范式：（i）支持性（ii）反事实（iii）随机遮盖。此外，我们还开发了一个自动定位问题相关证据并构建支持性、反事实和随机变体的管道，生成约100千张图像的数据集。在此基础上，我们使用基于GRPO的强化学习训练多模态语言模型，并设计三种互补的奖励机制来引导模型进行准确的推理和基于证据的推理。实验表明，DeFacto在多个基准测试上显著提高了答案准确性和推理忠实性，为可解释的多模态推理打下了更坚实的基础。
### Conclusion
我们的研究实验说明，DeFacto不仅在答案准确度上表现优异，同时在推理忠实性上也取得了显著的进步，为后续的多模态推理研究提供了一个强大的框架。DeFacto的代码已经在GitHub上开放，数据集也已在HuggingFace上发布。
## 17. `cs.AI` - Dr. Bench：深度研究代理多维度评估，从回答到报告 [PDF](https://arxiv.org/pdf/2510.02190), [HTML](https://arxiv.org/abs/2510.02190)
### Authors
Yang Yao,Yixu Wang,Yuxuan Zhang,Yi Lu,Tianle Gu,Lingyu Li,Dingyi Zhao,Keming Wu,Haozhe Wang,Ping Nie,Yan Teng,Yingchun Wang
### Background
现有的智能进化体系仍旧缺乏对深度研究代理（DRAs）的有效评估机制，传统的基准测试在评价维度、响应格式和评分机制上存在不足，限制了其对这类代理的有效评估。
### Innovation
提出了一种针对DRAs和长报告式响应的多维度评估框架——Dr. Bench。该框架包括214个由专家精选的挑战性任务，涵盖10个广泛领域，每个任务都配备了手工构建的参考包以支持综合评估。该框架引入了语义质量、主题聚焦和检索可信度的评估标准，以全面评价DRAs生成的长报告。
### Conclusion
实验证明，主流DRAs在与网络搜索工具增强推理模型进行比较时表现出更优越的性能，但仍有改进的空间。这项研究为DRAs的能力评估、架构优化和范式发展提供了坚实的基础。
## 18. `cs.AI` - Chain of Thought 和潜在思维之间的正式比较 [PDF](https://arxiv.org/pdf/2509.25239), [HTML](https://arxiv.org/abs/2509.25239)
### Authors
Kevin Xu,Issei Sato
### Background
Chain of thought (CoT) 通过生成中间标记明确地触发大语言模型的推理，而潜在思维直接在连续的潜在空间中进行计算，能够超越离散语言表示。尽管两者都利用了迭代计算，但它们的比较能力尚未充分探索。
### Innovation
本文提出了一个正式的分析，展示了潜在思维比本质上顺序的CoT在并行计算中更有效率。CoT则通过随机解码实现近似计数和抽样。
### Conclusion
这些分离表明，深度驱动的递归更适合于某些任务，从而为在推理范式之间进行选择提供了实用指导。
## 19. `cs.AI` - 神经网络嵌入与人类数据在通过心理测量调查项目获取价值观维度方面的同等效果 [PDF](https://arxiv.org/pdf/2509.24906), [HTML](https://arxiv.org/abs/2509.24906)
### Authors
Max Pellert,Clemens M. Lechner,Indira Sen,Markus Strohmaier
### Background
本文研究了通过大型语言模型提取的嵌入特征在经过'Survey and Questionnaire Item Embeddings Differentials'（SQuID）处理后能否恢复人类价值观的结构。这些价值观源于人类评估者对修订版肖像价值观问卷（PVQ-RR）的判断。研究者比较了多种嵌入模型，并使用包括内部一致性、维度相关性以及多维尺度配置等多项评估标准。
### Innovation
SQuID方法在无需特定领域微调或重新标注训练数据的情况下，能够克服获得维度之间负相关性的挑战。实证分析表明，嵌入法能够解释55%的维度间相似性方差；多维尺度配置与来自49个不同国家的汇总人类数据一致。此外，SQuID在三种个性测验中表现出一致的关联范围增加，表明其在价值观理论领域外的应用潜力。
### Conclusion
研究结果表明，语义嵌入可以有效地复制通过广泛的人类调查建立的心理测量结构。这种方法在成本、可扩展性和灵活性方面具有显著优势，同时还能保持与传统方法相当的质量。这一发现对心理学和社会科学的研究具有重要意义，提供了一种补充方法，可以扩展在测量工具中代表的人类行为和经验的范围。
## 20. `cs.AI` - 大语言模型在缓解无家可归问题中的政策制定评估：LLM会怎么做？ [PDF](https://arxiv.org/pdf/2509.03827), [HTML](https://arxiv.org/abs/2509.03827)
### Authors
Pierre Le Coz,Jia An Liu,Debarun Bhattacharjya,Georgina Curto,Serge Stinckwich
### Background
大语言模型（LLMs）在高风险领域中的应用越来越广泛。它们具备编码社会变化情境和生成合理情景的能力，使其成为社会政策制定中的潜在工具。本文旨在评估大型语言模型在政策建议中是否与领域专家（以及彼此之间）保持一致，尤其是针对缓解全球15亿人口中的无家可归问题这一挑战。
### Innovation
该研究开发了一个新的基准工具，包含四个城市的决策场景，以人类发展能力方法为概念框架。同时，研究团队还设计了一个自动化管道，将政策与一个基于代理的模型进行连接，并对比了由大型语言模型和专家推荐的政策的社会影响。研究结果表明，不同大型语言模型在政策建议上存在差异，但使用大型语言模型为政策制定提供洞见是有潜在益处的，前提是需要负责任的监管、情景校准以及地区专业知识。
### Conclusion
这项研究在计算框架中实现了人类发展能力方法，并为关注人类尊严的无家可归缓解政策制定提供了新的见解。研究成果表明，如果结合负责任的监管、情境校准和地方专业知识，大型语言模型能够在政策制定中提供有价值的洞见。
## 21. `cs.LG` - 记忆中永久性遗忘：持续学习与机制可解释性的交汇 [PDF](https://arxiv.org/pdf/2601.22012), [HTML](https://arxiv.org/abs/2601.22012)
### Authors
Sergi Masip,Gido M. van de Ven,Javier Ferrando,Tinne Tuytelaars
### Background
持续学习中的永久性遗忘（灾难性遗忘）通常是从性能或最后一层表示的角度来衡量的，但忽视了其背后的机制。本研究提出了一个新的机制框架，从几何角度解释了灾难性遗忘作为个体特征编码变化的结果。这些变化可能通过减少特征的分配容量（质量较差的表示）或中断下游计算对特征的读取来引起遗忘。
### Innovation
该研究引入了一个机制框架，提出了一个可操作的模型来正式化观点，并通过实验验证了这种分析，指出深度的负面影响。此外，通过使用Crosscoders，该框架还可以应用于实际模型的分析。在Vision Transformer在顺序CIFAR-10数据集上的案例研究中，展示了框架的应用。
### Conclusion
本研究提供了一种新的基于特征的持续学习词汇，有助于更好地理解持续学习过程中永久性遗忘现象的机制。
## 22. `cs.LG` - 变压器推断的率失真优化 [PDF](https://arxiv.org/pdf/2601.22002), [HTML](https://arxiv.org/abs/2601.22002)
### Authors
Anderson de Andrade,Alon Harell,Ivan V. Bajić
### Background
变压器在许多任务中表现出优越的性能，但在推理过程中对计算和内存的要求较高。通过将推理过程分布在多个设备上分割，可以提高效率，这又需要压缩中间表示。本文旨在介绍一种基于率失真理论的损失压缩框架，它可以学习到明确权衡码率与准确性的紧凑编码。
### Innovation
提出了一种基于率失真优化的损失压缩框架，能够学习紧凑的编码。该框架在语言基准实验中显示出显著的节省，并在某些情况下提高了准确性，优于更复杂的基线方法。此外，还发展了PAC风格的理论界，用于估计码率与熵之间的差距，从而为各种架构和任务提供了新的解释。
### Conclusion
通过引入率失真优化框架，作者们在理解和优化变压器的表示编码方面提供了统一的视角。通过计算理论界和实证研究，表明不同的架构和任务受到该框架的理论界驱动，这增加了该方法的解释性。
## 23. `cs.LG` - 视觉导向的关键令牌正则化方法在多模态大型语言模型去学习中的应用 [PDF](https://arxiv.org/pdf/2601.22020), [HTML](https://arxiv.org/abs/2601.22020)
### Authors
Chengyi Cai,Zesheng Ye,Peike Li,Bo Han,Jianzhong Qi,Feng Liu
### Background
在针对多模态大型语言模型（MLLMs）的去学习操作中，现有方法通常不区分各种重要程度的语言令牌，且仅集中于语言模态，忽略了视觉线索在标识关键令牌中的作用。因此，这些方法无法有效防止模型在回答特定图片问题时泄露私人信息。
### Innovation
本文提出了视觉导向的关键令牌正则化（ViKeR）方法，利用无关视觉输入预测去学习后的理想令牌级分布，并用这些分布来正则化去学习过程，从而优先考虑关键令牌；通过信息熵定义关键令牌进行去学习，并通过令牌级梯度重加权放大关键令牌的更新，提高去学习效果。
### Conclusion
在MLLMMU和CLEAR基准测试中，该研究中的方法展示了有效的去学习性能，同时降低了遗忘并保持了响应的连贯性。
## 24. `cs.LG` - 动态不确定性下的广义信息采集 [PDF](https://arxiv.org/pdf/2601.21988), [HTML](https://arxiv.org/abs/2601.21988)
### Authors
Fernando Palafox,Jingqi Li,Jesse Milzman,David Fridovich-Keil
### Background
在未知动态系统的环境中，代理必须通过观察来学习系统的动态。现有方法通过为特定建模选择（如动力学模型、信念更新过程、观测模型和计划器）设计定制化的成本来加速这一学习过程，但这限制了其灵活性。因此，需要一个统一的框架，可以将信息采集成本与建模选择分离，且能推广到各种不同的建模选择中。
### Innovation
本文提出了一种统一框架，通过明确揭示参数、信念和控制之间的因果依赖关系，将信息采集成本与建模选择解耦。框架基于Massey的定向互信息，适用于具有加性噪声的马尔可夫动力学系统，而不依赖于具体的建模选择，实现了一种通用的信息采集成本。证明证明了现有文献中使用的互信息成本是本文成本的一个特例。此外，利用此框架建立了互信息成本与线性贝叶斯估计中的信息增益之间的显式联系，为基于互信息主动学习方法提供了理论基础。
### Conclusion
通过实验，证实了该框架在线性、非线性和多代理系统中的实际应用效用。
## 25. `cs.LG` - TBDFiltering: 样本高效树形数据过滤 [PDF](https://arxiv.org/pdf/2601.22016), [HTML](https://arxiv.org/abs/2601.22016)
### Authors
Robert Istvan Busa-Fekete,Julian Zimmert,Anne Xiangyi Zheng,Claudio Gentile,Andras Gyorgy
### Background
机器学习模型的质量很大程度上取决于其训练数据的质量。对于大规模语言模型（LLMs），选择高质量且多样化的训练数据集是一项艰巨的任务，主要是因为缺乏廉价可靠的质量度量。虽然可以使用现有的LLMs查询文档质量，但这种方法不适用于训练过程中大量（数十亿）文档的规模。实践中，常使用基于稀疏质量信号的分类器。本文提出了一种基于文本嵌入的层次聚类方法，该方法能够自适应选择文档，通过少量查询的结果评估簇的质量。
### Innovation
本文提出了一种基于文本嵌入的层次聚类方法（TBDFiltering），该方法能够自适应选择需要评估的文档，通过少量查询估计簇的质量。证明了该方法在假设聚类树包含一个足以纯化叶子簇的子树时，可以在以高概率正确预测每个文档质量的前提下，查询文档的数量与最小纯化子树的规模成正比，而算法不需要预先知道这些子树的存在。
### Conclusion
实验结果表明，该算法在与基于分类器的质量过滤方法对比中显示出明显的优势。
## 26. `cs.LG` - 通过推理时的斯蒂夫尔激活引导探索多样生成路径 [PDF](https://arxiv.org/pdf/2601.22010), [HTML](https://arxiv.org/abs/2601.22010)
### Authors
Dongxuan Zhu,Ly Tran Ho Khanh,Andy Yat-Ming Cheung,Man-Chung Yue,Viet Anh Nguyen
### Background
语言模型通常倾向于产生一组高概率的输出，这使得生成路径单一同质，容易发生模式崩溃。虽然采样方法可以在生成过程中引入随机性，但在确保多次并发生成运行中的多样性方面仍然存在挑战。
### Innovation
该研究引入了一种名为STARS（基于斯蒂夫尔动态引导的多样化推理）的方法，这是一种在推理时进行干预的训练免费方法。STARS通过在斯蒂夫尔流形上联合优化多个添加的引导方向，将激活引导转化为一种探索机制。这种方法通过最大化引导激活向量的几何体积，同时通过斯蒂夫尔流形促进引导干预的正交性，显式地促进并发生成运行中的多样化激活向量，隐式地促进多样化的生成轨迹。此外，该方法设计了一种轻量级的一步更新机制，确保在确保低延迟的同时仍然能够实现多样性的生成。
### Conclusion
在测试案例生成和科学发现的基准测试中，STARS在提高多样性方面始终优于标准采样方法，同时保持了定性的表现。
## 27. `cs.LG` - PowerGenie：基于分析引导的进化式发现高效可重构电源转换器 [PDF](https://arxiv.org/pdf/2601.21984), [HTML](https://arxiv.org/abs/2601.21984)
### Authors
Jian Gao,Yiwei Zou,Abhishek Pradhan,Wenhao Huang,Yumin Su,Kaiyuan Yang,Xuan Zhang
### Background
发现更优电路拓扑结构需要在指数级庞大的设计空间中进行探索，这一挑战通常留给人类专家解决。现有的AI方法或选用预定义模板，或生成少量规模的新型拓扑结构但未进行严格的验证，导致大规模、基于性能发现仍然未得到充分探索。
### Innovation
PowerGenie框架自动发现高性能可重构电源转换器。引入了一种自动分析框架，不需要组件尺寸或SPICE仿真即可确定转换器的功能和理论性能极限。还提出了一种演化调优方法，该方法通过适应性选择和独特性验证来同时进化生成器模型及其训练分布。与现有方法相比，这种方法在语法有效性、功能有效性、新颖率和优度指标（FoM）上表现更优。该方法发现了一个新型8模式可重构转换器，其FoM比最佳训练拓扑高出23%；SPICE仿真结果确认了多个模式的平均绝对效率增益达到10%，在单一模式下最高可达17%。
### Conclusion
PowerGenie框架利用演化算法和自动分析方法，实现了对高性能可重构电源转换器的大规模发现，经过仿真验证其能显著提升效能。
## 28. `cs.LG` - Elign: 基于基础机器学习力场的守恒扩散模型对齐 [PDF](https://arxiv.org/pdf/2601.21985), [HTML](https://arxiv.org/abs/2601.21985)
### Authors
Yunyang Li,Lin Huang,Luojia Xia,Wenhe Zhang,Mark Gerstein
### Background
生成3D分子构象的模型必须遵守欧几里得对称性，并将概率质量集中在热力学有利且机械稳定的结构上。然而，E(3)等变扩散模型通常复制来自半经验训练数据的偏差，而不是捕捉高保真哈密尔顿的平衡分布。物理导向可以纠正这一问题，但它面临两个计算瓶颈：昂贵的量子化学评估（例如，DFT）和在每次采样步骤中重复这样的查询。
### Innovation
我们提出了Elign，一种后训练框架，能够缓解这两种成本。首先，我们用更快的、预训练的基础机器学习力场（MLFF）取代昂贵的DFT评估以提供物理信号。其次，我们通过将物理导向转移至训练阶段来避免每次运行时重复查询。为了实现第二个成本摊销，我们将反向扩散公式化为一个强化学习问题，并引入了力--能量分离的群相关策略优化（FED-GRPO）来微调去噪策略。FED-GRPO包含基于势能的能量奖励和基于力的稳定性奖励，它们独立地进行优化和群体标准化。
### Conclusion
实验表明，Elign生成具有较低标准DFT能量和力的构象，同时提高稳定性。关键的是，由于生成过程中不需要能量评估，推理速度与未引导的采样相同。
## 29. `cs.LG` - 漂移MDP的几何结构与路径积分稳定性证书 [PDF](https://arxiv.org/pdf/2601.21991), [HTML](https://arxiv.org/abs/2601.21991)
### Authors
Zuyuan Zhang,Mahdi Imani,Tian Lan
### Background
现实世界的强化学习通常是非稳定的，即奖励和动态会变化、加速、振荡并且触发最优动作的突然切换。现有理论往往使用粗糙的模型来描述环境变化的程度，而不是局部的变化方式，如加速和接近僵局驱动的表现误差和政策抖动。本文从几何的角度出发，用可微同伦路径来建模环境，并跟踪最优贝尔曼不动点的运动，从而得到了内在复杂性的长度、曲率和折点签名，证明了一种路径积分稳定性界限，并推导出满足本地稳定性的可行区域。这些结果为此后的工作奠定了基础。
### Innovation
提出了漂移MDP的几何观点，引入了几何跟踪RL (Homotopy-Tracking RL, HT-RL) 和几何跟踪MCTS (HT-MCTS)，能够在线估计路径积分的长度、曲率及接近僵局的邻近度作为代理指标，并据此调整学习或规划的强度。实验表明，在振荡和切换频繁的条件下，该方法能获得更好的追踪效果和动态悔失比。
### Conclusion
研究发现对漂移MDP进行几何建模并在其上估计路径积分可提供一种新的复杂性描述，进一步提出了HT-RL和HT-MCTS，可以更为精确地适应环境变化，从而提高强化学习的性能。尤其是在环境大幅改变的条件下，即使只依靠历史数据的其他方法容易失效，这种方法也能取得显著的效果。
## 30. `cs.LG` -  Negatives-Dominant Contrastive Learning for Generalization in Imbalanced Domains [PDF](https://arxiv.org/pdf/2601.21999), [HTML](https://arxiv.org/abs/2601.21999)
### Authors
Meng Cao,Jiexi Liu,Songcan Chen
### Background
IDG关注解决领域差异和标签分布偏移的问题，这些问题根本上影响模型的决策边界，特别是在不同领域具有异质长尾分布的情况下。尽管IDG具有重要的实际意义，但其研究仍相对不足，主要原因是处理这些复杂性技术上的难度及理论基础的欠缺。
### Innovation
本文理论性地建立了IDG的一般化上限，强调后验不一致和决策边界的作用。据此，本文提出了Negative-Dominant Contrastive Learning（NDCL），这是一种针对IDG的新颖方法，通过增强辨别能力和保持不同领域后验一致来提升分类准确性。具体而言，NDCL通过额外强调负样本作为对比学习的主要信号，增强了类间决策边界的分离。此外，通过重新加权交叉熵策略促进了类内的紧凑性，通过预测中心对齐策略确保了不同领域的后验一致性。
### Conclusion
我们的NDCL在基准测试中的严谨实验中表现出色。代码可以在以下链接获取：this https URL.
