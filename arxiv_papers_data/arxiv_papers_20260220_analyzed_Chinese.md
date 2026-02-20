# 20260220
[![Subscribe_Visitors](https://visitor-badge.laobi.icu/badge?page_id=nituchao.latest_arxiv_analyze_ai_rss)](https://github.com/nituchao/latest_arxiv_analyze_ai)

## 1. `cs.AI` - 革新AI中的长期记忆：高容量高效率存储的新前景 [PDF](https://arxiv.org/pdf/2602.16192), [HTML](https://arxiv.org/abs/2602.16192)
### Authors
Hiroaki Yamanaka,Daisuke Miyashita,Takashi Toi,Asuka Maki,Taiga Ikeda,Jun Deguchi
### Background
该研究基于“通过记忆提升世界”的使命，探讨了实现人工智能超级智能（ASI）所需的核心记忆设计概念。在传统的‘提取然后存储’方法中，相关信息被提取并保存，但这种方法存在丢失有价值信息的风险。
### Innovation
提出了‘存储然后按需提取’的方法来保留原始体验，并根据需要灵活应用。此外，还强调通过深入研究大规模概率体验集合以发现更深层次的见解，以及通过分享存储体验来提高体验收集效率。
### Conclusion
尽管这些方法看似有效，但简单的实验表明它们确实在实践中有显著效果。然而，也提到了制约这些方向进一步研究的主要挑战，并提议了相关研究方向以解决这些问题。
## 2. `cs.AI` - 从人类反馈中学习个性化代理 [PDF](https://arxiv.org/pdf/2602.16173), [HTML](https://arxiv.org/abs/2602.16173)
### Authors
Kaiqu Liang,Julia Kruk,Shengyi Qian,Xianjun Yang,Shengjie Bi,Yuanshun Yao,Shaoliang Nie,Mingyang Zhang,Lijuan Liu,Jaime Fernández Fisac,Shuyan Zhou,Saghar Hosseini
### Background
现代AI智能体虽然强大，但却经常未能与个体用户的独特、不断变化的偏好保持一致。先前的方法通常依赖静态数据集，通过互动历史训练隐式偏好模型或在外部记忆中编码用户简介。但是这些方法在处理新用户和随时间变化的偏好方面存在问题。因此，需要一种新的方法来解决这些挑战。
### Innovation
介绍了Personalized Agents from Human Feedback (PAHF)框架，该框架通过实时交互使用明确的用户内存进行在线学习。PAHF提出了一个三步循环：（1）在动作前寻求澄清以解决歧义，（2）将动作定位在从内存检索的偏好中，（3）通过收集动作后的反馈来更新内存以应对偏好的漂移。为评估该能力，开发了一个四阶段协议和两个基准，分别为实体操作和在线购物。这些基准量化了代理从零开始学习初始偏好并随后适应个性转变的能力。
### Conclusion
理论分析和实验结果表明，结合明确的内存与双反馈通道是至关重要的：PAHF学习速度显著提高，并且持续优于没有内存和单一通道的基线，减少了初始个性化误差并且能够快速适应偏好变化。
## 3. `cs.AI` - 在路由问题中通过神经求解器实现高效约束处理 [PDF](https://arxiv.org/pdf/2602.16012), [HTML](https://arxiv.org/abs/2602.16012)
### Authors
Jieyi Bi,Zhiguang Cao,Jianan Zhou,Wen Song,Yaoxin Wu,Jie Zhang,Yining Ma,Cathy Wu
### Background
神经求解器在解决简单路由问题方面取得了显著进展，特别是在计算效率上表现出色。然而，它们在处理复杂约束方面仍面临挑战，当前用于处理约束的方案，如可行性掩码或隐式可行性意识方法，对于硬约束可能是低效的或不适用的。
### Innovation
本文提出了一种名为Construct-and-Refine（CaR）的新的通用且高效的约束处理框架，适用于基于神经路由求解器的复杂约束情况。与以往通过大幅度改进来减少最优性差距但仍难以处理硬约束的混合求解器设计相比，CaR 通过设计一个联合训练框架来指导构建模块生成多样且高质量的解决方案，这些解决方案能够很好地适用于轻量级改进过程，例如，CaR 使用的改进过程只需10步，而先前的工作可能需要5000步。此外，CaR 首次使用了构建过程和改进过程共享的表示方法，这使得跨框架潜在的知识共享在更复杂的约束场景中成为可能。
### Conclusion
我们在典型的硬约束路由问题上评估了CaR，结果表明，与经典的和神经网络的最先进的求解器相比，CaR 在满足约束、解的质量和效率方面表现更优。
## 4. `cs.AI` - 通过自然语言反馈提高互动式上下文学习 [PDF](https://arxiv.org/pdf/2602.16066), [HTML](https://arxiv.org/abs/2602.16066)
### Authors
Martin Klissarov,Jonathan Cook,Diego Antognini,Hao Sun,Jingling Li,Natasha Jaques,Claudiu Musat,Edward Grefenstette
### Background
人类学习尤其是协作学习过程中，能够根据纠正性反馈调整思维过程是一种重要的能力。然而，当前的大语言模型训练方法主要依赖于大规模的静态语料库建模，这种方法虽然有助于知识获取，但未能体现模型根据上下文动态调整的能力。本文研究了这种与反馈互动的上下文学习能力的不足，并提出了一种新型框架，将其视为一种可训练的技能。
### Innovation
引入了一种可扩展的方法，将单轮可验证任务转化为由信息不对称驱动的多轮教诲式互动。这种方法首次证明了当前主流模型在处理复杂推理任务时难以整合纠正性反馈。研究还展示了通过此方法训练的模型，其在互动学习方面的提升显著，具体表现是小型模型在多轮交互中接近大型模型的表现。同时，这种互动训练在数学问题上的应用也能推广到不同的领域，如编程、益智题和迷宫导航。定量分析显示，这种改进归因于上下文中的增强可塑性。最后，证明了该范式为自我改进提供了一条统一路径，通过让模型预测教师的批评，模仿外部信号的自我修正机制，使模型能够在没有老师的情况下自我纠正。
### Conclusion
本文提出的框架将互动式上下文学习视为一种可训练的技能，而不是一种自发的属性，通过这种方法，模型能够根据信息不对称进行多轮学习和自我修正，这种能力不仅限于数学问题，在其他领域也有良好的表现，为大语言模型的自我改进提供了一条新的途径。
## 5. `cs.AI` - LLM-Grade-不确定性的度量基准 [PDF](https://arxiv.org/pdf/2602.16039), [HTML](https://arxiv.org/abs/2602.16039)
### Authors
Hang Li,Kaiqi Yang,Xianxuan Long,Fedor Filippov,Yucheng Chu,Yasemin Copur-Gencturk,Peng He,Cory Miller,Namsoo Shin,Joseph Krajcik,Hui Liu,Jiliang Tang
### Background
大型语言模型（LLMs）的兴起正在改变教育自动评估的格局。尽管这些系统在多种题目类型和多种输出格式的适应性和灵活性方面表现出显著优势，但它们也带来了与输出不确定性相关的挑战，这源于LLMs固有的概率本质。确保评估结果的可靠性至关重要，因为它们在提供学生反馈和指导教学决策方面发挥着关键作用。因此，可靠或校准不佳的不确定性估计可能会导致不稳定的下游干预措施，从而干扰学生的学习过程并产生意外的不利后果。
### Innovation
本研究通过广泛的不确定性量化方法基准测试，全面分析了多个评估数据集、LLM家族和生成控制设置下的不确定性行为，从而表征了LLMs在评分场景中的不确定性模式。该研究评估了不同不确定性指标的优缺点，并分析了关键因素，如模型家族、评估任务和解码策略，对不确定性估计的影响。这些发现提供了关于LLM基于自动评估中不确定性特征的实际见解，并为未来发展更可靠和有效的不确定性意识评分系统奠定了基础。
### Conclusion
本研究提供了一项关于LLM基于自动评估中不确定性的基准，基于此，可以推进更多可靠的、有效的不确定性感知评分系统的开发。
## 6. `cs.AI` - 自主代理工作流在临床症状检测中的优化不稳定现象 [PDF](https://arxiv.org/pdf/2602.16037), [HTML](https://arxiv.org/abs/2602.16037)
### Authors
Cameron Cagan,Pedram Fard,Jiazi Tian,Jingya Cheng,Shawn N. Murphy,Hossein Estiri
### Background
自主代理工作流能够通过迭代改进自身的策略，显示出巨大的潜力。然而，这些系统的失败模式仍然没有得到充分的理解和研究。特别是在持续自主优化过程中可能会出现优化不稳定现象，这种现象会 paradoxically 地降低分类器的性能。
### Innovation
研究团队使用开源框架 Pythia 对自主代理工作流中的优化不稳定现象进行了研究。他们通过观察三种不同流行程度的临床症状（呼吸困难、胸痛和长期 COVID 头晕）的表现，发现验证敏感性在迭代中不稳定地波动，并识别了一种新的失效模式：在低流行率下，系统可能达到高准确率但未能检测到任何阳性病例。研究还测试了两种干预措施，发现回溯选择代理比主动干预更为有效，特别是在低流行率的分类任务中。
### Conclusion
这些研究结果描述了一种自主 AI 系统的关键失效模式，并证明回溯选择对于低流行率分类任务的稳定化优于主动干预。
## 7. `cs.AI` - GPSBench：大型语言模型理解GPS坐标吗？ [PDF](https://arxiv.org/pdf/2602.16105), [HTML](https://arxiv.org/abs/2602.16105)
### Authors
Thinh Hung Truong,Jey Han Lau,Jianzhong Qi
### Background
随着大型语言模型（LLMs）越来越多地应用于与物理世界互动的应用领域，如导航、机器人学或制图，稳健的空间地理推理成为了关键能力。尽管如此，LLMs 在处理GPS坐标和现实世界地理方面的能力探索仍然不足。
### Innovation
本文引入了GPSBench，这是一个包含57,800个样本和17个任务的基准数据集，用于评估LLMs的空间地理推理能力，涵盖了几何坐标操作（如距离和方位角计算）和将坐标与世界知识结合的推理。通过关注模型内在能力而不依赖于工具使用，本文评估了14种最新模型，并发现GPS推理仍然具有挑战性，任务间存在显著差异：模型在现实世界地理推理方面的可靠性普遍高于几何计算；地理知识表现出等级衰减，国家层面表现出色但城市层面定位较弱；对坐标的噪音具有高稳健性表明模型具备真正的坐标理解能力而非记忆。
### Conclusion
GPS坐标的增强能够提升下游空间地理任务的表现，微调会在几何计算能力提高和世界知识丧失之间产生权衡。本文的数据集和可重复代码可在提供的链接中找到。
## 8. `cs.AI` - 基于证据的亚专业推理：评估定制化临床智能层在2025内分泌科考试中的表现 [PDF](https://arxiv.org/pdf/2602.16050), [HTML](https://arxiv.org/abs/2602.16050)
### Authors
Amir Hosseinian,MohammadReza Zare Shahneh,Umer Mansoor,Gilbert Szeto,Kirill Karlin,Nima Aghaeepour
### Background
大型语言模型在普遍医学考试中表现出色，但在亚专科临床推理方面仍面临挑战，因为这需要处理快速更新的指南和复杂的证据层次。因此，对于亚专科的临床推理，急需一个根基在证据上的系统来改善现有技术的局限性。
### Innovation
研究评估了一个名为January Mirror的证据驱动的临床推理系统，该系统整合了一个精选的内分泌学和心血管代谢证据库，并构建了一个结构化的推理框架，以生成证据链接的输出。与前沿的LLMs（GPT-5, GPT-5.2, Gemini-3-Pro）相比，Mirror在内分泌学板式考试中表现更佳，特别是在最困难的30个问题上。
### Conclusion
Mirror系统通过提供证据追溯性确保了其结果的可靠性。在74.2%的输出中，至少引用了一个指南级别的来源，在手动验证中达到了100%的引用准确性。这表明，对于亚专科临床推理，针对特定领域的证据收集比泛用的网络检索更加有效，也能为临床部署提供审计能力。
## 9. `cs.AI` - 向量可验证奖励：多轮工具调用大语言模型代理的代理状态评估 [PDF](https://arxiv.org/pdf/2602.16246), [HTML](https://arxiv.org/abs/2602.16246)
### Authors
Yun-Shiuan Chuang,Chaitanya Kulkarni,Alec Chiu,Avinash Thangali,Zijie Pan,Shivani Shekhar,Yirou Ge,Yixi Li,Uma Kona,Linsey Pang,Prakhar Mehrotra
### Background
多轮对话和多步工具调用的大语言模型（LLM）代理在生产中越来越普及。评估这些代理的基准必须既可靠地比较模型，又能提供训练数据。现有的代理基准依赖完全确定性的后端，这不仅成本高，迭代也困难。
### Innovation
提出了一种基于代理状态的评估方法（Proxy State-Based Evaluation），这是一个由LLM驱动的仿真框架，解决了无需确定性数据库的问题。具体而言，场景指定了用户目标、用户/系统事实、预期最终状态和预期代理行为，LLM 状态追踪器从完整的交互记录中推断结构化的代理状态。LLM 推断者通过场景约束验证目标完成情况，并检测工具/用户幻觉。实验证明该基准可以产生稳定的、区分模型的排名，并在看不到的场景中提供转移监督。精细的场景指定在消融研究的支持下，能实现几乎零的仿真幻觉率。该框架还支持针对用户人设的敏感性分析。人类-LLM 推断者的一致性超过90%，表明自动评估的可靠性。
### Conclusion
代理状态评估为工业LLM代理提供了实用且可扩展的替代方案，与确定性代理基准相比，实现了一种更加灵活且可靠的评估机制。
## 10. `cs.AI` - 企业健身房Corecraft：在高保真强化学习环境中训练通用型代理 [PDF](https://arxiv.org/pdf/2602.16179), [HTML](https://arxiv.org/abs/2602.16179)
### Authors
Sushant Mehta,Logan Ritchie,Suhaas Garre,Nick Heiner,Edwin Chen
### Background
研究团队发现，将AI代理训练在高保真强化学习环境中可以使其获得超越训练分布范围的能力。论文介绍了一个名为Corecraft的新环境，它是Surge AI企业级代理强化学习环境套件中的首个环境。Corecraft是一个全面的企业模拟，该模拟涵盖了超过2,500个实体，并由14种实体类型和23种独特工具组成，旨在测试AI代理是否能够完成多步骤的专业工作，这些任务对于真实工作来说具有挑战性。
### Innovation
论文提出了一种新的环境Corecraft，并展示了通过在该环境中训练AI代理可以使其在任务多样性及挑战性方面表现出色。具体来说，研究团队创立了Corecraft，它具有以下特点：以其为中心构建的世界设计以优化多样的挑战性任务；专家撰写的评分标准能够实现可靠的奖励计算；模拟的企业工作流反映了现实的职场模式。此外，研究团队还发现在Corecraft环境中，经过单一训练周期后，GLM~4.6模型的表现有了显著提升，该模型的未见任务通过率从25.37%提高到了36.76%。重要的是，这些进展也体现在不同分布基准上。
### Conclusion
实验结果显示了环境的质量、多样性和真实性是实现可推广的代理能力的关键因素。研究结果展示了通过创建高质量、多样化且现实的环境来训练AI代理的重要性，为强化学习领域提供了新的见解。
## 11. `cs.CL` - 零样本TTS中的声音印象控制 [PDF](https://arxiv.org/pdf/2506.05688), [HTML](https://arxiv.org/abs/2506.05688)
### Authors
Kenichi Fujita,Shota Horiguchi,Yusuke Ijima
### Background
声音中的非言语信息对于塑造听众的印象至关重要。尽管零样本文本转换语音（TTS）在实现高speaker忠实度方面取得了成功，但在控制感知声音特征（即印象）方面，调节细微的非言语信息依然面临挑战。
### Innovation
开发了一种零样本TTS声音印象控制方法。该方法利用低维向量表示不同的声音印象对（例如，暗-亮），并能够通过大型语言模型自动生成目标印象向量，从而无需手动优化。
### Conclusion
客观和主观评估结果均证明了该方法在印象控制方面的有效性，并在演示页面上提供了音频示例。
## 12. `cs.CL` - 语言与经验：复杂任务中社会学习的计算模型 [PDF](https://arxiv.org/pdf/2509.00074), [HTML](https://arxiv.org/abs/2509.00074)
### Authors
Cédric Colas,Tracey Mills,Ben Prystawski,Michael Henry Tessler,Noah Goodman,Jacob Andreas,Joshua Tenenbaum
### Background
人类通过结合他人的语言指导和直接经验来快速发展和学习新环境。人们如何整合这两种知识来源，以及AI系统能否做到这一点？
### Innovation
文章提出了一个计算框架，将社会学习建模为基于传感器数据和语言数据的结构化执行世界模型的概率联合推理过程。此模型能将预训练的语言模型转化为基于其信念如何分享建议的概率模型，使得智能体不仅能生成建议，还能理解语言输入作为证据来推进贝叶斯推理。
### Conclusion
研究通过行为实验和模拟10款视频游戏，展示了语言指导如何影响探索和加速学习，减少风险互动，加快关键发现。此外，通过迭代学习实验展示了知识在代际间的累积，并证明了人类与模型间知识的转移，揭示了结构化、语言兼容的表示如何可能促进人类与机器的合作学习。
## 13. `cs.CL` - 远在天边：评估语言模型在澳式英语和印式英语俚词理解上的表现 [PDF](https://arxiv.org/pdf/2602.15373), [HTML](https://arxiv.org/abs/2602.15373)
### Authors
Deniz Kaya Dilsiz,Dipankar Srirag,Aditya Joshi
### Background
现有语言模型在处理非标准语言变体时表现出了系统性的性能差距，尤其是在理解特定于方言的俚语方面，对于多种语言来说仍是一个相对未被充分探索的领域。本文通过全面评估印度英语和澳大利亚英语中俚语的识别能力，对这一现象进行了深入探讨。
### Innovation
文章构建了两个互补数据集WEB和GEN，用于评估最新的七种语言模型在理解和使用俚语方面的性能。通过三个任务——目标词预测（TWP）、引导目标词预测（TWP*）和目标词选择（TWS），揭示了语言模型在解决不同语言变体中方言俚语问题上的差异性和表现。
### Conclusion
研究结果发现：(1) 在目标词选择任务中，模型的表现高于目标词预测和引导目标词预测任务。(2) 对于WEB数据集，模型的性能优于GEN数据集。(3) 印度英语任务的整体表现优于澳大利亚英语，尤其是在目标词选择任务中，差距最大。这些发现强调了不同变体的语言生成和分类能力之间的基本差异，尤其是在俚语表达方面，即便是在技术丰富如英语的语言中也是如此。
## 14. `cs.CL` - EconEvals: 基准和边缘测试用于LLM代理的经济决策 [PDF](https://arxiv.org/pdf/2503.18825), [HTML](https://arxiv.org/abs/2503.18825)
### Authors
Sara Fish,Julia Shephard,Minkai Li,Ran I. Shorrer,Yannai A. Gonczarowski
### Background
鉴于大规模语言模型（LLMs）在经济决策中的潜在应用，本文旨在评估这些模型在经济决策制定方面的能力和倾向。研究者基于经济学中的关键问题（如采购、调度和定价）来开发基准测试方法，以及通过设计一个多目标决策任务框架来量化LLMs的选择行为。此外，通过综合评估多种前沿的LLM模型，探索了模型在时间和能力上的变化，为推进LLMs在经济决策中的应用提供了基础。
### Innovation
本文的创新在于提出了EconEvals方法和框架。它包括开发基于经济学核心问题的基准测试方法来测试LLMs从环境学习的能力，以及设计了一种边缘测试框架，通过定量评估LLMs在带有多个冲突目标的决策任务中的行为来衡量模型的决策行为、可靠性和能力。这项工作的创新为评估不断集成到经济决策中的LLM代理提供了基础。
### Conclusion
本文通过评估多种前沿的LLM模型，研究了模型在经济决策中的能力随时间的变化，得出了具有经济学意义的洞察。同时验证了所提出的边缘测试框架的自洽性、稳健性和可推广性，为LLMs在经济决策领域的进一步整合提供了坚实的基础。
## 15. `cs.CL` - RoboSpatial: 教授2D和3D视觉-语言模型机器人领域的空间理解 [PDF](https://arxiv.org/pdf/2411.16537), [HTML](https://arxiv.org/abs/2411.16537)
### Authors
Chan Hee Song,Valts Blukis,Jonathan Tremblay,Stephen Tyree,Yu Su,Stan Birchfield
### Background
机器人需要具备空间理解能力以便感知周围环境、推理环境并与其有意义地互动。现代机器人技术越来越多地依赖于视觉-语言模型来提供这些功能。然而，这些模型在进行空间推理任务时面临巨大挑战，因为它们的训练数据主要基于通用图像数据集，这些数据集通常缺乏精细的空间理解能力。例如，这些数据集往往没有捕捉到参考框架理解的能力，而有效的空间推理需要理解是从自身、世界还是物体的视角进行推理。
### Innovation
为了解决这一问题，作者引入了RoboSpatial，这是一个大规模的空间理解数据集，专注于机器人领域。该数据集包含真实的大规模室内和桌面场景，其中3D扫描和第一视角图像被捕获并附有丰富且与机器人相关的空间信息。数据集包括100万张图像、5000个3D扫描和300万标注的空间关系。数据集的2D第一视角图像与3D扫描配对使其同时适用于2D和3D场景。实验结果表明，使用RoboSpatial进行训练的模型在下游任务，如空间功能预测、空间关系预测和机器人操作方面优于基线模型。
### Conclusion
RoboSpatial数据集显著提高了视觉-语言模型在机器人领域的空间理解能力，特别是在空间功能预测、空间关系预测和机器人操作等下游任务上表现突出。
## 16. `cs.CL` - SNAP-UQ: 自监督下一激活预测在TinyML中的单次通过不确定性估计 [PDF](https://arxiv.org/pdf/2508.12907), [HTML](https://arxiv.org/abs/2508.12907)
### Authors
Ismail Lamaakal,Chaymae Yahyati,Khalid El Makkaoui,Ibrahim Ouahbi,Yassine Maleh
### Background
在TinyML（在设备上运行的机器学习）中，微控制器需要检测故障、分布迁移或准确度下降，但通常的方法需要多个推理步骤、额外分支或需占用大量内存的额外状态，这些都不适用于功率和存储空间有限的设备。现有的不确定性估计方法如深度集成、MC蒙特卡洛丢弃、早退出和时间缓冲等方式在这些场景下并不适用。
### Innovation
本文提出了一种新颖且实用的方法，即SNAP-UQ，用于基于深度卷积层预测下一激活可能性的单次通过、无标签的不确定性估计。SNAP-UQ使用一个小的骨干网络层集和小型的int8头部来预测下一激活均值和方差，并通过轻量级单调校准器将标准化的预测误差转换为深度层面的惊讶信号。该设计不需要时间缓冲或辅助出口，且只增加了少量几十千字节的部署空间。
### Conclusion
SNAP-UQ在多种视觉和音频骨干网络中相对于早期退出和深度集成基准在Flash和延迟上减少了约40-60%和25-35%，且在受污染的数据流中提高了准确度下降事件检测的性能，并保持了强大的故障检测能力。通过基于层间动力学而不是单纯依赖输出信心，SNAP-UQ为鲁棒TinyML监控提供了一种新颖且资源友好的基础。
## 17. `cs.CL` - 大型语言模型中基于内容的网络安全拒绝决策框架 [PDF](https://arxiv.org/pdf/2602.15689), [HTML](https://arxiv.org/abs/2602.15689)
### Authors
Noa Linder,Meirav Segal,Omer Antverg,Gil Gekker,Tomer Fichman,Omri Bodenheimer,Edan Maor,Omer Nevo
### Background
大型语言模型和基于LLM的代理越来越多地用于网络安全任务，这些任务本身具有二用性。现有方法通常依赖于广泛的基于主题的禁令或以攻击为重点的分类系统来阻止内容，这会导致不一致的决策、过度限制合法防御者的行为，并在混淆或请求分割时表现脆弱。
### Innovation
提出了一个基于内容的框架，用于设计和审核网络拒绝政策，使其明确刻画了攻击-防御权衡。该框架根据请求的技术实质，从五个维度对请求进行表征：攻击行动贡献、攻击风险、技术复杂性、防御效益和合法用户预期频率，而非仅仅依赖于声明的意图或攻击分类。
### Conclusion
该内容导向的方法解决了当前前沿模型行为中的不一致性，并使组织能够构建可调节、风险意识强的拒绝政策。
## 18. `cs.CL` - STAPO: 通过静默罕见错误标记 Token 稳定大语言模型的强化学习 [PDF](https://arxiv.org/pdf/2602.15620), [HTML](https://arxiv.org/abs/2602.15620)
### Authors
Shiqi Liu,Zeyu He,Guojian Zhan,Letian Tao,Zhilong Zheng,Jiang Wu,Yinuo Wang,Yang Guan,Kehua Sheng,Bo Zhang,Keqiang Li,Jingliang Duan,Shengbo Eben Li
### Background
强化学习（RL）在大型语言模型（LLMs）的推理方面取得了显著进步，但现有的RL微调方法高度依赖于诸如熵正则化和重新加权等启发式技术来保持稳定性。这些方法在实践中经常导致后期性能崩溃，导致推理质量下降，训练不稳。研究表明，RL中的令牌级策略梯度幅度与其概率和局部策略熵呈负相关。训练不稳可能是由于一小部分约0.01％的标记，我们称之为‘错误标记’（spurious tokens）所引起的。当这些标记出现在正确响应中时，它们对推理结果的贡献很小，但继承了整个序列级别的奖励，导致异常放大梯度更新。
### Innovation
本文设计了一种称为S2T（Silencing Spurious Tokens）机制，通过特征信号识别低概率、低熵和正优势的错误标记，进而抑制它们在优化过程中的梯度扰动。该机制整合进组目标中，提出了错误标记感知策略优化（Spurious-Token-Aware Policy Optimization, STAPO）框架，促进了稳定而有效的大型模型精炼。研究表明，STAPO在六个数学推理基准上，对GRPO、20-Entropy和JustRL分别平均实现了7.13%（ρT=1.0, top-p=1.0）和3.69%（ρT=0.7, top-p=0.9）的性能改进。
### Conclusion
STAPO通过认识和解决错误标记问题，增强了RL方法在LLMs上的稳定性，提高了模型的推理质量，展现了在多个基准上的显著性能提升。
## 19. `cs.CL` - GDGB: 用于生成动态文本标注图学习的基准 [PDF](https://arxiv.org/pdf/2507.03267), [HTML](https://arxiv.org/abs/2507.03267)
### Authors
Jie Peng,Jiarui Ji,Runlin Lei,Zhewei Wei,Yongchao Liu,Chuntao Hong
### Background
动态文本标注图（DyTAGs）同时整合了结构、时间和文本属性，对于建模复杂的真实世界系统至关重要。然而，大多数现有的DyTAG数据集文本质量较差，严重限制了其在需要丰富语义输入的生成DyTAG任务中的应用。此外，先前的研究主要集中在DyTAG上的判别任务，缺乏为DyTAG生成标准化的任务定义和评价协议。
### Innovation
本文提出了生成DyTAG基准（GDGB），包含八个高质量文本特征的DyTAG数据集，弥补了之前的限制。在此基础之上，定义了两种新的生成DyTAG任务：自编码动态图生成（TDGG）和归纳动态图生成（IDGG）。设计了多维度的评估指标来综合评估生成的DyTAG的质量。进一步提出了GAG-General，这是一种基于LLM的多代理生成框架，用于生成DyTAG的可重复和稳健基准测试。实验结果表明，GDGB能够严格评价TDGG和IDGG，揭示了结构和文本特征在DyTAG生成中的关键作用。
### Conclusion
GDGB为生成DyTAG研究提供了一个基础资源，促进了在生成DyTAG领域的进一步实用应用。该数据集和源代码可在以下链接获取。
## 20. `cs.CL` - 无需标签的演化语言模型：多数决定选择，新颖促进多样性 [PDF](https://arxiv.org/pdf/2509.15194), [HTML](https://arxiv.org/abs/2509.15194)
### Authors
Yujun Zhou,Zhenwen Liang,Haolin Liu,Wenhao Yu,Kishan Panaganti,Linfeng Song,Dian Yu,Xiangliang Zhang,Haitao Mi,Dong Yu
### Background
大型语言模型（LLMs）越来越多地通过可验证奖励的强化学习（RLVR）进行训练，但在实际部署中，需要能自我改进而无需标签或外部评判员的模型。现有自我改进方法主要依赖自我确认信号（如置信度、熵或一致性）生成奖励，这导致模型倾向于产生自信和大众偏好的解决方案，进而引发熵塌缩，降低通过特定标准的准确率和推理复杂度。
### Innovation
提出了EVOL-RL，一种无标签框架，该框架模仿了进化的平衡选择与多样性的原则。具体而言，EVOL-RL将多数投票的答案作为稳定性锚点，但添加了一种新颖度感知的奖励，根据其推理与同时生成的响应有多大不同来为每个采样解决方案打分。这种多数决定稳定性+新颖促进探索的规则模仿了变异选择的原则：选择防止漂移，新颖性防止塌缩。实验结果表明，EVOL-RL在一无标签基准AIME24的数据集上优于仅多数基准，例如，训练Qwen3-4B-Base模型的结果从基线的4.6%提升到16.4%的pass@1，从18.5%提升到37.9%的pass@16。EVOL-RL不仅防止领域内多样性塌缩，还提高了领域外的一般化能力（从数学推理延伸到更广泛的任务，例如MMLU-Pro和BBEH）。
### Conclusion
EVOL-RL在提高模型的多样性、稳定性和领域间迁移能力方面表现出色，为无需标签的自我改进提供了一个有效框架。
