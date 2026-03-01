# 20260301
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - FIRE: 一个全面的金融智能和推理评估基准 [PDF](https://arxiv.org/pdf/2602.22273), [HTML](https://arxiv.org/abs/2602.22273)
### Authors
Xiyuan Zhang,Huihang Wu,Jiayu Guo,Zhenlin Zhang,Yiwei Zhang,Liangyu Huo,Xiaoxiao Ma,Jiansong Wan,Xuewei Jiao,Yi Jing,Jian Xie
### Background
本文介绍了FIRE基准，旨在评估LLMs的理论金融知识以及解决实际商业场景的能力。该基准通过构建多样化的考试问题集和分类复杂的金融领域来评估LLMs在理论和实践方面的表现。
### Innovation
本文创新地提出了一种全面的评估矩阵，用于评估LLMs在金融领域的应用能力。该基准包括3000个金融场景问题，涵盖闭合式决策问题和需根据预定义评判标准评估的开放式问题。此外，该基准还通过公开发布问题和评价代码来促进未来的相关研究。
### Conclusion
本文对最新的LLM模型XuanYuan 4.0在金融应用中的能力边界进行了全面评估，结果提供了对当前LLM在金融应用中能力范围的系统分析。同时，本基准还将公开发布，以促进未来的研究工作。
## 2. `cs.AI` - 沿着作者合作网络走向灵感：结合检索增强生成方法的大型语言模型驱动的科学创新生成 [PDF](https://arxiv.org/pdf/2602.22215), [HTML](https://arxiv.org/abs/2602.22215)
### Authors
Pengzhen Xie,Huizhi Liang
### Background
大型语言模型（LLMs）在科学创意生成方面展现出潜力，但由于生成结果往往缺乏可控的学术背景和可追踪的灵感路径，这一领域仍存在明显不足。为此，本文结合作者知识图谱与检索增强生成（RAG）技术提出了一种新的科学创意生成系统GYWI，旨在为LLMs提供可控的学术背景和灵感路径的追溯能力。
### Innovation
该系统创新性地提出了基于作者中心的知识图谱构建方法和灵感来源采样算法，以构建外部知识库；提出了结合RAG和GraphRAG的混合检索机制，以实现深度和广度知识的检索；并且引入了结合强化学习原理的提示优化策略，以自动引导LLMs基于混合上下文优化生成结果。此外，文中还开发了一个全面的评估方法，包括多选题任务的自动评估、LLM评分、人工评估和语义空间可视化分析，从新颖性、可行性、清晰度、相关性和重要性五个维度对生成的创意进行评估。
### Conclusion
通过在包括GPT-4o、DeepSeek-V3、Qwen3-8B和Gemini 2.5在内的不同LLMs上的实验，研究结果表明，与主流LLMs相比，GYWI在新颖性、可靠性和相关性等多个指标上取得了显著的优势。
## 3. `cs.AI` - vibe 研究：技能型 AI 代理能否替代或增强社会科学工作者？ [PDF](https://arxiv.org/pdf/2602.22401), [HTML](https://arxiv.org/abs/2602.22401)
### Authors
Yongjun Zhang
### Background
本文探讨了AI代理在社会科学中的新兴作用。与过去仅能响应单一查询的聊天机器人不同，AI代理能够阅读文件、运行代码、查询数据库、搜索网络并运用领域专业知识，实现了整个研究流程的自主执行。作者提出了一种新的研究方法——vibe 研究，与前文提到的‘vibe 编码’类似，强调领域技能的应用。同时，作者利用一个包括21项研究技能的插件（scholar-skill，为 Claude Code 设计）来具体说明这个概念。
### Innovation
1. 提出了‘vibe 研究’的概念，作为‘vibe 编码’的现代对应。2. 制定了一个认知任务框架，通过编码能力和隐性知识需求两个维度来区分研究活动，并确定了认知而非顺序的分工边界。3. 强调AI代理在速度、覆盖范围和方法论支持方面的优势，但也在理论原创性和领域隐性知识方面存在局限。
### Conclusion
本研究指出AI代理可作为技能增强工具，但存在成果脆弱性、职业分层风险和教育危机等多重挑战。因此，提出了五项原则以促进负责任的vibe研究：明确分工、透明沟通、多学科合作、保护人力价值和强化教育革新。
## 4. `cs.AI` - 多层次因果嵌入 [PDF](https://arxiv.org/pdf/2602.22287), [HTML](https://arxiv.org/abs/2602.22287)
### Authors
Willem Schooltink,Fabio Massimo Zennaro
### Background
因果模型的抽象可以允许模型细化的过程中保持因果关系不变。以往的研究更倾向于研究两个模型之间的关系，而本论文探讨了一种框架，使得多个详细的模型可以映射到更粗略的因果模型的子系统中。这里的因果嵌入是一种抽象的一般化，并提出了一个更广泛的相容概念。通过定义一个多分辨率边际问题，我们展示了因果嵌入对于统计边缘问题和因果边缘问题的相关性，并进一步说明了其在合并来自不同表示模型的数据集方面的实际应用。
### Innovation
论文提出了因果嵌入的概念，这是一类更广泛的抽象，并且展示了如何通过多分辨率边缘问题将多个详细的模型映射到更粗略的因果模型的子系统中，从而解决了统计和因果边缘问题。此外，论文还提出了如何在不同表示模型的数据集中进行实际应用。
### Conclusion
通过定义多分辨率边缘问题，该研究展示了因果嵌入在统计和因果边缘问题中的相关性，并提出了一种方法来合并来自不同表示模型的数据集，从而推广了因果模型的抽象概念。
## 5. `cs.AI` - 在弱监督与强监督下，隐含推理方法的表现如何？ [PDF](https://arxiv.org/pdf/2602.22441), [HTML](https://arxiv.org/abs/2602.22441)
### Authors
Yingqian Cui,Zhenwei Dai,Bing He,Zhan Shi,Hui Liu,Rui Sun,Zhiji Liu,Yue Xing,Jiliang Tang,Benoit Dumoulin
### Background
隐含推理是一种新兴的推理框架，通过生成步骤来替代文本空间，而是在隐含空间进行多步推理，从而使推理能突破离散语言标记的限制，并在连续隐含空间中执行多步计算。尽管已经有许多研究关注如何改进隐含推理的表现，但其内部机制仍不完全清楚。
### Innovation
该研究对隐含推理方法进行了全面分析，以更好地理解隐含表示在整个过程中的作用和行为。研究发现了两种关键问题：第一，观察到普遍存在捷径行为，即这些方法在不依赖于隐含推理的情况下也能实现高准确性；第二，检验了隐含推理是否支持在隐含空间中的BFS-like探索，发现虽然隐含表示可以编码多种可能性，但推理过程并未忠实实施结构化搜索，而是表现出隐含的剪枝和压缩。
### Conclusion
研究发现监督强度存在权衡关系：强监督会减少捷径行为但限制隐含表示保持多样假设的能力；而弱监督则允许更丰富的隐含表示，但会增加捷径行为。
## 6. `cs.AI` - ArchAgent: 由代理人工智能驱动的计算机体系结构发现 [PDF](https://arxiv.org/pdf/2602.22425), [HTML](https://arxiv.org/abs/2602.22425)
### Authors
Raghav Gupta,Akanksha Jain,Abraham Gonzalez,Alexander Novikov,Po-Sen Huang,Matej Balog,Marvin Eisenberger,Sergey Shirobokov,Ngân Vũ,Martin Dixon,Borivoje Nikolić,Parthasarathy Ranganathan,Sagar Karandikar
### Background
敏捷硬件设计流程对于满足不断增长的计算需求至关重要。最近，代理生成型人工智能系统在算法设计、代码效率提升以及跨科学领域的发现方面取得了显著进展。本文的研究背景在于将代理生成型AI系统与硬件体系结构设计相结合。
### Innovation
提出了一个名为ArchAgent的自动计算机体系结构发现系统，基于AlphaEvolve构建。该系统展示了其自动设计并实现领先（SoTA）的缓存替换策略的能力，这些策略不仅调整参数，还设计新机制/逻辑。ArchAgent在短短两天内生成了一个策略，该策略在公共多核Google工作负载跟踪上实现了5.3%的IPC速度提升，相比之下，此前的SoTA策略仅在18天内实现0.9%的IPC速度提升。系统还实现了与特定工作负载（组合）更高度特化的性能提升，达到了2.4%的IPC速度提升。此外，还探讨了代理生成型AI对计算机体系结构研究的影响，包括发现仿真器漏洞的现象。
### Conclusion
ArchAgent 达到了与现有SoTA策略相比更快的速度提升，为计算机体系结构研究在人工智能代理时代提供了新的视角，尤其体现在研究工具设计考量上的深刻洞见。
## 7. `cs.AI` - 代理行为合同：可靠自主人工智能代理的正式规范与运行时强制 [PDF](https://arxiv.org/pdf/2602.22302), [HTML](https://arxiv.org/abs/2602.22302)
### Authors
Varun Pratap Bhardwaj
### Background
传统软件依赖于合同（如API、类型系统、断言）来指定和执行正确的行为。相比之下，基于提示和自然语言指令的AI代理没有任何形式的行为规范，这导致了漂移、治理失败和AI代理部署中频繁的项目失败。论文指出这一差距是漂移、治理失败和项目失败的根本原因。
### Innovation
论文提出了代理行为合同（ABC），这是一种结合设计合同原则的自主AI代理的正式框架。它通过包含预条件、不变量、治理政策和恢复机制等一阶可运行执行组件来规范行为。进一步定义了（p, delta, k）满意度，这是一种考虑了LLM非确定性和恢复的概率的合同合规性概念，并证明了一个漂移边界定理，表明具有恢复率γ>α的合同在期望上将行为漂移限制在D*=α/γ，在随机设置中具有高斯集中度。论文还建立了多代理链中安全合同组合的充分条件，并推导出了概率退化界。实现了ABC在AgentAssert运行时强制库，并在AgentContract-Bench上进行了评估，这是一个涵盖6家供应商7个模型的200个场景基准。研究表明，与未合同化的基线相比，合同化的代理在每会话中检测到5.2-6.8个软违规（p<0.0001，Cohen's d=6.7-33.8），实现88-100%的硬约束合规，并且将行为漂移限制在D*<0.27，所有模型的最大恢复率为100%，并在1980个会话中实现了可忽略不计的开销（<10 ms/动作）。
### Conclusion
代理行为合同提供了一种新的方法来规范自主AI代理的行为，通过定义精确的合同来确保其行为的一致性。该框架通过考虑即时执行和概率合规性，可以有效控制AI代理行为的漂移，且在实际应用场景中显示出了显著的性能提升。
## 8. `cs.AI` - 自主记忆代理的研究 [PDF](https://arxiv.org/pdf/2602.22406), [HTML](https://arxiv.org/abs/2602.22406)
### Authors
Xinle Wu,Rui Zhang,Mustafa Anis Hussain,Yao Lu
### Background
现有记忆代理通过提取个人经历和对话历史到外部存储中来改进大模型（LLMs），实现低开销的上下文构建和在线记忆更新，但这些方法较为被动和反应式。现有的记忆增长受限于可用信息，而记忆代理也不常在不确定性中主动寻求外部输入。
### Innovation
本文提出了主动记忆代理，具有主动获取、验证和整理知识的特性，并通过成本意识的知识提取级联和语义感知的Thompson采样机制来降低冷启动偏差。
### Conclusion
U-Mem 在可验证和不可验证基准测试中，持续优于先前的记忆基线，可以超越基于强化学习的优化方法，在HotpotQA 和 AIME25 上分别提升了14.6点和7.33点。
## 9. `cs.AI` - 通过认知抽象与推理语料库探究人类在抽象规则推理和问题解决中的行为 [PDF](https://arxiv.org/pdf/2602.22408), [HTML](https://arxiv.org/abs/2602.22408)
### Authors
Caroline Ahn,Quan Do,Leah Bakst,Michael P. Pascale,Joseph T. McGuire,Michael E. Hasselmo,Chantal E. Stern
### Background
人类在抽象推理方面表现出惊人的灵活性，能够快速从稀疏的示例中学习和应用规则。为了研究这种能力背后的认知策略，作者引入了Cognitive Abstraction and Reasoning Corpus (CogARC)，这是一份基于Abstraction and Reasoning Corpus (ARC)的人类适应子集，最初设计用于评估人工智能中的抽象推理能力。CogARC被用于对75个抽象视觉推理问题进行的人类实验，参与者能自由地生成解决方案，从而推导出输入到输出规则。
### Innovation
通过Cognitive Abstraction and Reasoning Corpus (CogARC)研究了人类在抽象规则推理和问题解决中的行为，收集了高时间分辨率的数据，包括实例查看、编辑序列和多次尝试提交。研究发现，即使在错误的解决方案中，错误的解决路径也可以高度收敛，这提供了关于人们如何在不确定性下推广、误推广和调整策略的见解。
### Conclusion
Cognitive Abstraction and Reasoning Corpus (CogARC)是一个丰富的行为环境，用于研究人类抽象推理，揭示了在任务过程中人们响应速度加快但准确率略有下降的现象，同时表明随着任务结构的熟悉度增加，规则学习能力并没有提高。
## 10. `cs.AI` - 知识过滤与集体幻觉：信心校准代理的陪审团理论 [PDF](https://arxiv.org/pdf/2602.22413), [HTML](https://arxiv.org/abs/2602.22413)
### Authors
Jonas Karge
### Background
本文探讨了一群异质学习者如何随着时间学习估计自身可靠性，并在必要时选择不投票。传统的集体智慧投票理论，例如库尔诺陪审团定理（CJT），假设参与者固定不变，但在实际情况中，允许代理人说“我不知道”往往更有益。本文提出了一个概率框架，在最终的信心门控阶段，学习者要决定是否投票或弃权之前，先进行一个校准阶段，更新他们关于自身固定能力的信念。
### Innovation
本文提出了一个新的框架，其中代理人在投票前经历一个校准阶段，更新他们关于自身确定性的信念，并在最后的信心门控阶段决定是否投票或弃权。研究证明了选择性参与情景的应用，该框架可以在序列化、信心门控设置中推广CJT的渐近保证，即使不是渐近的下限也得出了集体成功率的非渐近边界。并通过蒙特卡洛模拟验证了这些边界。此外，还讨论了该框架在AI安全中的潜在应用，尤其是降低了集体大型语言模型决策中的幻觉问题。
### Conclusion
本文研究了通过允许代理人性别地学习和自我评估来获得集体准确性的好处。提出了一个新框架，该框架可以推广经典理论的渐近保证，即使在非渐近情况下也能提高集体成功率。框架中代理人的选择性参与可以减少集体幻觉，这在AI决策中意义重大，尤其是在大型语言模型的集体决策中。这些发现为理解和改进集体决策的机制提供了新的见解。
## 11. `cs.AI` - PoSh: 使用场景图引导LLMs-as-a-Judge进行详细的图像描述 [PDF](https://arxiv.org/pdf/2510.19060), [HTML](https://arxiv.org/abs/2510.19060)
### Authors
Amith Ananthram,Elias Stengel-Eskin,Lorena A. Bradford,Julia Demarest,Adam Purvis,Keith Krut,Robert Stein,Rina Elster Pantalony,Mohit Bansal,Kathleen McKeown
### Background
视觉-语言模型（VLMs）在生成详细图像描述方面取得了显著进展，但评估这些描述仍然具有挑战性。传统的评估指标如CIDEr和SPICE是为短文本设计的，并且它们着重于识别现在不常见的错误，例如物体识别错误。然而，长文本的描述需要对属性和关系的精细评分，能够指出具体文本错误的范围。本文探讨了这些挑战，并提出了一种新的评估方法PoSh。
### Innovation
引入了PoSh，这是一种基于场景图的评估指标，用于指导LLMs-as-a-Judge，以产生基于细微错误（如组合理解错误）的评分。PoSh相对于现有指标（包括GPT4o-as-a-Judge），具有可复制性和可解释性，能够更好地作为人类评分者的代理。
### Conclusion
PoSh在DOCENT（一个包含艺术作品、专家编写的参考和模型生成描述的新基准）上的表现比现有最佳的评估指标要好。我们还使用PoSh评估了开放和封闭模型在DOCENT中的表现，发现前沿模型在描绘具有丰富场景动态的图像时难以达到完美无误的描述，这建立了衡量VLM进展的一个新挑战任务。PoSh和DOCENT的应用有望推动辅助文本生成等相关领域的发展。
## 12. `cs.AI` - RELOOP：使用多跳推理器和规划者的递归检索方法用于异构问答 [PDF](https://arxiv.org/pdf/2510.20505), [HTML](https://arxiv.org/abs/2510.20505)
### Authors
Ruiyi Yang,Hao Xue,Imran Razzak,Hakim Hacid,Flora D. Salim
### Background
当前的检索增强生成（RAG）方法在处理多步问题和异构证据源时仍然脆弱，它们在准确性与延迟和标记/工具预算之间进行权衡。
### Innovation
RELOOP介绍了结构意识框架，使用分层序列（HSEQ）来（i）将文档、表格和知识图谱线性化为轻量级结构标签的可逆分层序列，（ii）执行结构意识迭代来收集生成答案所需的适量证据，最后由头部代理组合规范化证据生成最终答案，可选的校正循环以解决检测到的矛盾。此外，RELOOP具有以下三个关键优势：一种格式通用的统一方法，可在一个策略中操作文本、表格和知识图谱，无需按数据集专门化；引导、预算意识迭代，减少不必要的跳转、工具调用和标记，同时保持准确性；及证据规范化，提高问答可靠性和答案一致性和可审计性。
### Conclusion
实验表明，RELOOP在用于多跳推理的单过、多跳和智能体RAG基线之上，显示出高效且一致的EM/F1优势，且展现了三个关键优势：格式通用的统一方法、引导和预算感知迭代及证据规范化。
## 13. `cs.AI` - 时序稀疏自编码器：利用语言的序列特性进行可解释性 [PDF](https://arxiv.org/pdf/2511.05541), [HTML](https://arxiv.org/abs/2511.05541)
### Authors
Usha Bhalla,Alex Oesterling,Claudio Mayrink Verdun,Himabindu Lakkaraju,Flavio P. Calmon
### Background
将模型内部表示和计算转化为人类可以理解的概念是模型可解释性的一个关键目标。虽然一些稀疏自编码器（SAEs）等字典学习方法提供了发现人类可理解特征的有希望途径，但这些方法往往只能恢复特定于标记的、嘈杂或高度局部的概念。这些方法的一个限制是忽视了语言的时间结构，其中语义内容通常在序列中平滑地发展。
### Innovation
引入了时序稀疏自编码器（T-SAEs），通过一个新颖的对比损失，鼓励高阶特征在相邻标记中的持续激活。这一简单的但强有力的方法使SAEs能够以半监督的方式解耦语义和句法特征。在多个数据集和模型上，T-SAEs 恢复了更平滑、更连贯的语义概念，同时不牺牲重构质量。尽管没有明确的语义信号，这些模型仍表现出明显的语义结构，为语言模型的无监督可解释性提供了新的途径。
### Conclusion
时序稀疏自编码器揭示了更连贯的语义概念，同时保持了重建质量。它们展示了在缺乏明确语义信号的情况下，仍然可以实现明显的语义结构，为无监督可解释性开辟了新途径。
## 14. `cs.AI` - DropVLA: 一种针对视觉-语言-动作模型的动作级别后门攻击 [PDF](https://arxiv.org/pdf/2510.10932), [HTML](https://arxiv.org/abs/2510.10932)
### Authors
Zonghuan Xu,Xiang Zheng,Xingjun Ma,Yu-Gang Jiang
### Background
Vision-Language-Action (VLA) 模型能够将多模态感知和语言指令映射到可执行的机器人动作。由于这些模型高度依赖于训练数据，因此它们很容易受到行为后门攻击。攻击者在训练中引入隐藏触发器，可以导致不可预期的物理动作，同时保持名义任务性能的完整性。现有研究主要集中在无目标攻击或任务级别的劫持上，较少关注对各个动作的精细控制。
### Innovation
本文设计了一种名为 DropVLA 的动作级别后门攻击，能够在真实的管道黑盒设置下，利用少量的数据污染访问，使可重用的动作原语（如 open_gripper）在攻击者选择的时间点执行。采用区块精细调整中的窗口一致重标记方案，实现隐藏触发器的有效注入。实验证明，仅使用视觉污染即可在 LIBERO 平台上的 OpenVLA-7B 达到 98.67%-99.83% 的攻击成功率，同时保持 98.50%-99.17% 的任务保留率，且在 25 步控制步骤内诱发目标动作。
### Conclusion
该研究揭示了 VLA 模型在最小化污染和未观察到性能下降的情况下，能够被隐蔽地引导，对其进行关键安全动作的精细控制。视觉污染比仅文字触发器更稳定，将视觉与文字结合没有显著提高攻击成功率，但攻击后门在适度触发变化下仍然稳健，并且能够跨不同评估套件迁移。进一步在带有 pi0-fast 的 7 自由度 Franka 手臂上验证了物理世界的可行性，表明在相机相对移动下，图像平面触发器漂移能够实现非平凡的攻击效果。
## 15. `cs.AI` - 瓦特智能：衡量本地AI的智能效率 [PDF](https://arxiv.org/pdf/2511.07885), [HTML](https://arxiv.org/abs/2511.07885)
### Authors
Jon Saad-Falcon,Avanika Narayan,Hakki Orhun Akengin,J. Wes Griffin,Herumb Shandilya,Adrian Gamarra Lafuente,Medhya Goel,Rebecca Joseph,Shlok Natarajan,Etash Kumar Guha,Shang Zhu,Ben Athiwaratkun,John Hennessy,Azalia Mirhoseini,Christopher Ré
### Background
当前的语言模型查询主要由集中式的云基础设施中的前沿模型处理。随着需求迅速增长，云提供商难以迅速扩展基础设施。两项进展促使我们重新思考这一范式：小型语言模型（参数量<=200亿）在许多任务上的表现已与前沿模型相当，并且本地加速器（如苹果M4 Max）可以在交互式延迟下运行这些模型。这引发了一个问题：本地推理能否有效地重新分配对集中式基础设施的需求？
### Innovation
本文提出了瓦特智能（IPW，Task Accuracy per Unit Power）作为评估本地推理能力与效率的指标，计算方式为任务准确率除以单位功耗。通过大规模实验，研究了20多个最先进的本地语言模型和8种加速器在真实世界的100万个单轮对话和推理查询中的表现，包括准确率、能耗、延迟和功耗等指标。
### Conclusion
研究表明，本地语言模型可以准确回答88.7%的查询，且IPW在2023-2025年提高了5.3倍，本地查询覆盖从23.2%增加到71.3%。本地加速器在相同的模型上实现了至少1.4倍的IPW降低，显示出显著的优化空间。这些发现证明本地推理可以重新分配来自集中式基础设施的一部分需求，IPW成为跟踪这一转变的关键指标。
## 16. `cs.AI` - 监督式强化学习：从专家轨迹到逐步推理 [PDF](https://arxiv.org/pdf/2510.25992), [HTML](https://arxiv.org/abs/2510.25992)
### Authors
Yihe Deng,I-Hung Hsu,Jun Yan,Zifeng Wang,Rujun Han,Gufeng Zhang,Yanfei Chen,Wei Wang,Tomas Pfister,Chen-Yu Lee
### Background
大型语言模型（LLMs）在需要多步骤推理的问题上往往表现不佳。对于小型开源模型，强化学习带可验证奖励（RLVR）在多次尝试后仍难于正确回答时失败，而监督式微调（SFT）则容易过度拟合长示例，通过逐个模仿令牌进行僵硬的学习。
### Innovation
我们提出了监督式强化学习（SRL）框架，将问题解决重新定义为生成一系列逻辑“动作”的序列。SRL 训练模型在执行每个动作之前生成内部推理说明。此框架提供了基于模型动作与从SFT数据集中提取的专家动作的相似性逐步递进的奖励。这种监督提供了即使所有卷积都不正确也能提供更丰富的学习信号，同时鼓励在专家演示指导下的灵活推理。SRL使小模型能够学习以前无法通过SFT或RLVR学习的复杂问题；而且，在RLVR细化前通过SRL初始化训练可获得最强的整体性能。
### Conclusion
SRL不仅适用于推理基准，还能有效地推广到主动软件工程任务，证明其作为一种强大且多功能的推理导向LLM训练框架的适用性。
## 17. `cs.AI` - 在潜在空间中的扩散模型用于医学图像分割任务 [PDF](https://arxiv.org/pdf/2512.01292), [HTML](https://arxiv.org/abs/2512.01292)
### Authors
Huynh Trinh Ngoc,Toan Nguyen Hai,Ba Luong Son,Long Tran Quoc
### Background
医学图像分割对于临床诊断和治疗规划至关重要。传统方法通常产生单个分割掩码，无法捕捉内在不确定性。近年来，生成模型能够创建每幅图像多个合理的掩码，模拟多位临床医生的联合解释。然而，这些方法仍计算密集型。
### Innovation
提出了一种基于扩散的框架MedSegLatDiff，结合变分自编码器（VAE）和潜在扩散模型进行高效的医学图像分割。使用VAE将输入压缩到低维潜空间，减少噪声并加速训练，而扩散过程直接操作此紧凑表示。此外，用加权交叉熵替换传统的MSE损失以更好地保留微小结构例如小结节。在ISIC-2018（皮肤病变）、CVC-Clinic（息肉）和LIDC-IDRI（肺结节）数据集上评估MedSegLatDiff，同时产生多样性的分割假设和置信图，相比确定性基线提供增强的可解释性和可靠性。
### Conclusion
MedSegLatDiff在分割和信心图生成方面达到了最先进的或极具竞争力的Dice和IoU分数，同时增强了临床部署的可解释性和可靠性。
## 18. `cs.AI` - Q²: 低比特量化感知的梯度平衡与注意力对齐 [PDF](https://arxiv.org/pdf/2511.05898), [HTML](https://arxiv.org/abs/2511.05898)
### Authors
Zhaoyang Wang,Dong Wang
### Background
量化感知训练（QAT）在低比特（≤4比特）量化分类网络中取得了显著成功。然而，在应用于更复杂的视觉任务，如目标检测和图像分割时，性能仍会出现显著下降。文献中对这一现象的主要原因之一——特征融合阶段的梯度不平衡却有所忽视。
### Innovation
本文从新视角重新审视这一现象，指出关键失败因素是由于累积量化误差引起的特征融合阶段的梯度不平衡。基于此诊断，提出Q²框架，包括：1）量化感知梯度平衡融合（Q-GBFusion），一个闭环机制，在特征融合过程中动态重新平衡梯度贡献；2）量化感知注意力分布对齐（Q-ADA），一种无需参数的监督策略，通过语义相关性和量化敏感性重建监督分布，提供更稳定可靠的监督，以稳定训练和加速收敛。
### Conclusion
大量的实验表明，我们的方法作为一种即插即用和通用策略，可以集成到各种最新的QAT管道中，分别在目标检测上实现平均+2.5%的mAP增益和图像分割上实现+3.7%的mDICE改进。特别地，这个方法仅在训练阶段使用且不引入推理时开销，使其适用于实际部署。
## 19. `cs.AI` - 从正确示例中学习回答 [PDF](https://arxiv.org/pdf/2510.15464), [HTML](https://arxiv.org/abs/2510.15464)
### Authors
Nirmit Joshi,Gene Li,Siddharth Bhandari,Shiva Prasad Kasiviswanathan,Cong Ma,Nathan Srebro
### Background
该研究关注的问题是生成问题的适当答案或完成，其中可能会有多个正确答案，只要测试时选择其中之一即可。学习基于训练过程中每个问题给出某些正确答案的演示，类似于有监督的微调。研究通过将问题视为在上下文臂中的模仿学习（即徒工学习）来正式化该问题，并利用未显式观察到的专家（最优或非常优秀的）策略的离线演示。与之前假设演示者属于有限复杂性策略类的工作不同，该研究提出仅仅依赖于底层奖励模型（即指定哪些答案是正确的）属于有限复杂性类是一个更弱的假设。
### Innovation
该研究展示了最大似然方法在该环境下可能失效，并提出了一种方法，通过样本复杂性与奖励类基数呈对数关系来学习近乎与演示者一样好的答案。该方法类似于 Syed 和 Schapire 2007 的工作，适应到上下文臂（即一步）设置，但是一种简单的单次在线方法，具有“乐观率”（即当演示者最优时为 $1/theta$，而在 Syed 和 Schapire 中为 $1/theta^2$），即使面对任意适应性演示也能适用。
### Conclusion
该研究通过仅依赖底层奖励模型并采用一种新的学习方法，展示了在基于正确示例的学习问题上的严格改进，并巧妙地处理了适应性演示器的问题。
## 20. `cs.AI` - 工具体决斗：评估多元化、现实及长期任务执行的语言代理 [PDF](https://arxiv.org/pdf/2510.25726), [HTML](https://arxiv.org/abs/2510.25726)
### Authors
Junlong Li,Wenshuo Zhao,Jian Zhao,Weihao Zeng,Haoze Wu,Xiaochen Wang,Rui Ge,Yuxuan Cao,Yuzhen Huang,Wei Liu,Junteng Liu,Zhaochen Su,Yiyang Guo,Fan Zhou,Lueyang Zhang,Juan Michelini,Xingyao Wang,Xiang Yue,Shuyan Zhou,Graham Neubig,Junxian He
### Background
现有的语言代理基准大多集中在狭窄的领域或简化任务上，缺乏评估代理真实世界性能所需的多样性、现实性和长时间复杂性的能力。语言代理必须处理跨多种应用的复杂、多步骤工作流，在真实环境中执行任务，需要综合多个应用程序的功能，例如邮件管理、数据监控等。
### Innovation
引入了名为Tool Decathlon（简称Toolathlon）的基准测试，这是一个旨在评测语言代理在多元化、现实及长期任务执行中的能力的基准测试。Toolathlon涵盖了32款软件应用和604个工具，具有真实的初始环境状态和可靠的执行评估。不同于过去的基准测试仅确保功能上的现实性但环境状态多样性有限，该基准提供了来自真实软件的应用场景，例如带有数十名学生的Canvas课程或实际财务表单。
### Conclusion
全面评估了当前最先进的模型，揭示了它们在实际任务执行中的重大不足。最好的模型Claude-4.5-Sonnet仅在20.2轮工具调用后成功率为38.6%，而顶级开放权重模型DeepSeek-V3.2-Exp的成功率为20.1%。预计Toolathlon将推动开发出更适合现实世界、长时间任务执行的语言代理。
## 21. `cs.LG` - 超越归因：统一的概念级解释 [PDF](https://arxiv.org/pdf/2410.12439), [HTML](https://arxiv.org/abs/2410.12439)
### Authors
Junhao Liu,Haonan Yu,Xin Zhang
### Background
随着模型泛适用解释技术与基于概念的方法相整合的需求增加，模型泛用解释技术可以跨越不同架构来解释模型，而基于概念的方法则使解释更加忠实和易于理解给最终用户。然而，现有的基于概念的模型泛用解释方法范围有限，主要集中在属性归因解释上，而忽略了许多不同的形式如充分条件和反事实，从而限制了其应用范围。
### Innovation
本文提出了一种通用框架 UAELE，旨在将现有的局部模型泛用技术提升为提供基于概念的解释。我们发现可以通过使用大规模预训练模型扰动来统一扩展现有的局部模型泛用方法，以提供统一的概念级解释。该框架被应用于三种形式的解释：归因、充分条件和反事实，并应用于流行的文本、图像和多模态模型。
### Conclusion
我们的评估结果显示，UAELE 提供的解释比现有的最佳基于概念的解释方法更加忠实，且提供了更丰富、更能满足各种用户需求的解释形式。
## 22. `cs.LG` - 使用最小阻力路径解释深度网络 [PDF](https://arxiv.org/pdf/2502.12108), [HTML](https://arxiv.org/abs/2502.12108)
### Authors
Sina Salek,Joseph Enguehard
### Background
Integrated Gradients (IG) 是一种广泛使用的基于路径的归因方法，它通过沿基线到输入的直线路径集成模型梯度来为输入特征分配重要性分数。尽管在某些情况下有效，但研究表明直线路径可能导致归因错误。
### Innovation
本文识别了这些错误归因的原因，并提出了一种新方法——Geodesic Integrated Gradients (GIG)，该方法赋予输入空间以模型诱导的黎曼度量，并通过沿此度量下的测地线计算梯度积分，来避免归因错误。此外，还提出了No-Cancellation Completeness (NCC) 这一新公理，即测地路径下的路径归因中，NCC成立当且仅当积分路径为测地线。
### Conclusion
通过在合成和真实图像分类数据上的实验，本文提供了支持其理论分析的实验证据，表明GIG方法在所考虑的基准上产生比现有方法更忠实的归因，包括IG。
## 23. `cs.LG` - 通过无监督几何深度学习揭示全球图特征 [PDF](https://arxiv.org/pdf/2503.05560), [HTML](https://arxiv.org/abs/2503.05560)
### Authors
Mirja Granfors,Jesús Pineda,Blanca Zufiria Gerbolés,Joana B. Pereira,Carlo Manzo,Giovanni Volpe
### Background
图提供了强大的框架来建模复杂系统，但其结构变化性为分析和分类带来了重大挑战。
### Innovation
提出了一种新的无监督几何深度学习框架GAUDI，旨在捕捉局部细节和全局结构。GAUDI采用具有层次聚合并采样层的创新沙漏架构，通过跳跃连接保持在编码-解码过程中关键连通性信息。
### Conclusion
通过跨多个应用（如小型世界网络建模、超分辨率显微镜中蛋白质组装的表征、维克塞数学分析、以及脑连接随年龄变化的识别），证明了GAUDI在分析复杂图方面的优越性能，并提供了不同科学领域中涌现现象的全新见解。
## 24. `cs.LG` - 密集奖励差分学习实现Few-步扩散模型的对齐 [PDF](https://arxiv.org/pdf/2411.11727), [HTML](https://arxiv.org/abs/2411.11727)
### Authors
Ziyi Zhang,Li Shen,Sen Zhang,Deheng Ye,Yong Luo,Miaojing Shi,Dongjing Shan,Bo Du,Dacheng Tao
### Background
少几步的扩散模型能够高效地合成高分辨率图像，但难以与特定的下游目标对齐，这是因为现有低步骤范围内的强化学习（RL）方法在状态空间有限和样本质量不佳的情况下表现不足。
### Innovation
本文提出了步进扩散策略优化（SDPO）框架，这是一种专为少几步扩散模型设计的新型RL框架。SDPO引入了双重状态轨迹采样机制，每步跟踪噪点和预测的干净状态，提供密集的回报反馈，并启用低方差、混合步骤的优化。此外，通过基于潜空间相似性的密集回报预测策略进一步提高效率，减少密集回报查询的成本。
### Conclusion
实验结果表明，SDPO能够持续提供在多种少几步设置和任务中与密集回报差学习对齐的优异结果。
## 25. `cs.LG` - 神经符号AI在常微分方程分析解中的应用 [PDF](https://arxiv.org/pdf/2502.01476), [HTML](https://arxiv.org/abs/2502.01476)
### Authors
Orestis Oikonomou,Levi Lingsch,Dana Grund,Siddhartha Mishra,Georgios Kissas
### Background
解析解的微分方程提供了准确且可解释的见解，但这些解析解很少存在，因为找到它们需要专家直觉或在组合空间中的详尽搜索。现有的方法在这方面存在局限性。
### Innovation
我们引入了SIGS，这是一种神经符号框架，用于自动化此过程。SIGS采用形式语言生成仅含有语法上有效的构建块，将这些表达式嵌入到连续空间中，并在此空间中搜索，通过最小化基于物理的剩余来组装、评分和改进候选闭式解。该设计将符号推理与数值优化统一；语法使候选解决方案构建块在构建时即为恰当，而隐空间使探索变得可处理且无需数据。
### Conclusion
总体而言，SIGS在标准基准上的精度和效率都比现有符号方法提高了多个数量级。SIGS是第一个能够（i）解析解决耦合的非线性PDE系统，（ii）在语法不准确的情况下发现解决方案，以及（iii）为缺乏已知闭形式解的PDE生成准确的符号近似的方法。
## 26. `cs.LG` - Mixing It Up: 探索 Mixer 网络在不规则多元时间序列预测中的应用 [PDF](https://arxiv.org/pdf/2502.11816), [HTML](https://arxiv.org/abs/2502.11816)
### Authors
Christian Klötergens,Tim Dernedde,Lars Schmidt-Thieme,Vijaya Krishna Yalavarthi
### Background
在医疗保健、气候科学和生物学等领域，不规则采样且包含缺失值的多元时间序列的预测是一项基本挑战。虽然近期的图像和时间序列预测进步表明，基于轻量级的MLP架构（如MLP-Mixer、TSMixer）能在准确性和效率上与基于注意力模型相当，但它们在处理不规则和稀疏时间序列方面的应用尚未被探索。
### Innovation
本文提出了一种名为IMTS-Mixer的新型架构，适用于IMTS场景。它包含以下两个关键组件：(1)ISCAM，一种通道编码器，利用简单的MLP将不规则观察值转换为固定大小的向量；(2)ConTP，连续时间解码器，支持在任意时间点进行预测。在标准基准数据集上，该模型在预测准确性和推理时间上都达到了最先进的效果，并且参数量更少。
### Conclusion
我们的研究表明，IMTS-Mixer在部分不规则且稀疏时间序列的预测中展现出显著的优势。
