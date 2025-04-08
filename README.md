# LLM-inference-optimization-paper

Summary of some awesome works for optimizing  LLM inference

This summary will including three parts:

1. some **repositories** that you can follow
2. some representative **person** or **labs** that you can follow
3. some important **works** in the different research interests

## Repositories

For example, [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) contains many excellent articles, and is keeping updating (which I believe is the most important for a paperlist). [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) and [Awesome_LLM_Accelerate-PaperList](https://github.com/galeselee/Awesome_LLM_Accelerate-PaperList/) are also worth reading.

Besides, [awesome-AI-system](https://github.com/lambda7xx/awesome-AI-system) works also very well. And you can find other repositories in its content.  

The log ["Large Transformer Model Inference Optimization"](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) helps me a lot at the beginning.  

This log [OpenAI Keynote on Building Scalable AI Infrastructure](https://www.servethehome.com/openai-keynote-on-building-scalable-ai-infrastructure/) seems to be a laeding guidance.  

## Person/Lab

**Follow others' research, and find yourself's idea.**  

It is not my intention to judge the work of these pioneers, and I understand that the shortness of my knowledge will lead me to leave out many important people.
If you have a different opinion, please feel free to communicate with me through the issue.  
**In no particular order!!**  
**Damn, I can't remember the names of foreigners.**  

[Zhihao JIA](https://www.cs.cmu.edu/~zhihaoj2/): FlexFlow and other imporessive work, important role in MLSys, affiliated with CMU  
[Tianqi CHEN](https://tqchen.com/): TVM, XGBoost, and other imporessive work, important role in Machine Learning System and ML compilers, affiliated with CMU  
[Song HAN](https://hanlab.mit.edu/songhan): many important work in efficient ML including sparsity and quantization. btw, the class [*TinyML and Efficient Deep Learning Computing*](https://efficientml.ai) is highly recommanded, affiliated with MIT
[Zhen DONG](https://dong-zhen.com/): many important work in quantization and high-performance ML, affiliated with UCB  
[Tri DAO](https://tridao.me/): author of FlashAttention, affiliated with Princeton  
[Ce ZHANG](https://zhangce.github.io/): famous in efficient MLsys, affiliated with UChicago  
[Ion Stoica](https://people.eecs.berkeley.edu/~istoica/): Alpa, Ray, Spark, et.al.  

[SPCL](https://spcl.inf.ethz.ch/Publications/): Scalable Parallel Computing Lab, affiliated with ETHz  
[Luo MAI](https://luomai.github.io/): affiliated with University of Edinburgh

[IPADS](https://ipads.se.sjtu.edu.cn/zh/publications/): focus more on PURE systems, buut also make great progress in MLSys, affiliated with SJTU  
[EPCC](http://epcc.sjtu.edu.cn/): Emerging Parallel Computing Center, parallel computing and MLSys are Naturally combined, affiliated with SJTU

[Xin JIN](https://xinjin.github.io/): FastServe and LLMCad are impressive work, affiliated with PKU  
[Bin CUI](https://cuibinpku.github.io/): important role in MLSys including DL, GNN, and MoE, affiliated with PKU  
[Jidong ZHAI](https://pacman.cs.tsinghua.edu.cn/~zjd/): leading many important work in MLSys, affiliated with THU  
[Lingxiao MA](https://xysmlx.github.io/): with many important work in MLSys on Top-Conference, affiliated with MSRA  
[Cheng LI](http://staff.ustc.edu.cn/~chengli7/): high performce system and MLSys, affiliated with USTC  
[Xupeng Miao](https://hsword.github.io/): SpotServe, SpecInfer, HET, et.al

[Chuan WU](https://i.cs.hku.hk/~cwu/): with some important work in distributed machine learning systems, affiliated with HKU
[James CHENG](https://www.cse.cuhk.edu.hk/~jcheng/index.html): affiliated with CUHK  
[Kai CHEN](https://www.cse.ust.hk/~kaichen/): database works well with MLSys, affiliated with HKUST  
[Lei CHEN](https://scholar.google.com/citations?hl=zh-CN&user=gtglwgYAAAAJ&view_op=list_works&sortby=pubdate): database works well with MLSys, many papers so I recommand u to focus on his Top-Conference paper, affiliated with HKUST  
[Yang YOU](https://scholar.google.com/citations?hl=en&user=jF4dPZwAAAAJ&view_op=list_works&sortby=pubdate): leader of Colossal-AI, affiliated with NUS  
[Wei WANG](https://www.cse.ust.hk/~weiwa/): work in System and MLSys, affiliated with HKUST

## Work

I hope to conlude these impressive works based on their research direction.  
But my summary must not be informative enough, and I am looking forward to your addition.  

**Perhaps someone should write a detailed survey.**  

**Periodically check the "cited by" of the papers with ⭐ will be helpful.**  
**Paragraphs with 💡 are not perfect.**

### Survey/Evaluations/Benchmarks 💡

- [ ] ⭐ [Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models](https://arxiv.org/abs/2311.03687): evaluations helps you find the bottleneck  
- [ ] ⭐ [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017): a survey by UCB  
- [x] ⭐ [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234): worth a read
- [ ] ⭐ [Deep Learning Workload Scheduling in GPU Datacenters: A Survey](https://dl.acm.org/doi/full/10.1145/3638757): survey for GPU Datacenters DL Workload Scheduling
- [ ] ⭐ [Towards Efficient and Reliable LLM Serving: A Real-World Workload Study](https://arxiv.org/abs/2401.17644): a benchmark for LLM serving
- [ ] ⭐ [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363): both survey and analysis
- [ ] [A SURVEY OF RESOURCE-EFFICIENT LLM AND MULTIMODAL FOUNDATION MODELS](https://arxiv.org/pdf/2401.08092.pdf): worth reading
- [ ] [Training and Serving System of Foundation Models: A Comprehensive Survey](https://arxiv.org/abs/2401.02643)
- [ ] [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748)
- [ ] ⭐ [Towards Coarse-to-Fine Evaluation of Inference Efficiency for Large Language Models](https://arxiv.org/abs/2404.11502)
- [ ] ⭐ [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294): worth reading
- [ ] [Beyond the Speculative Game: A Survey of Speculative Execution in Large Language Models](https://arxiv.org/abs/2404.14897)
- [ ] ⭐ [Navigating Challenges and Technical Debt in Large Language Models Deployment](https://dl.acm.org/doi/abs/10.1145/3642970.3655840): important
- [ ] [The CAP Principle for LLM Serving](https://arxiv.org/abs/2405.11299): anothor angle
- [ ] [Demystifying Data Management for Large Language Models](https://dl.acm.org/doi/abs/10.1145/3626246.3654683): talking about database in LLM, by Xupeng MIAO, accepted by SIDMOD'24
- [ ] [Benchmarking LLM Inference Backends: vLLM, LMDeploy, MLC-LLM, TensorRT-LLM, and TGI](https://bentoml.com/blog/benchmarking-llm-inference-backends): with [code](https://github.com/bentoml/llm-bench/tree/main)
- [ ] [A Survey on Mixture of Experts](https://arxiv.org/pdf/2407.06204)
- [ ] [Analyzing LLM performance: The impact of high-bandwidth memory on model inference](https://www.micron.com/content/dam/micron/global/public/documents/products/product-flyer/llm-inference-engineering-report.pdf): analyze of inference
- [ ] [Inference Optimization of Foundation Models on AI Accelerators](https://arxiv.org/abs/2407.09111)
- [ ] [LLM Inference Serving: Survey of Recent Advances and Opportunities](https://arxiv.org/abs/2407.12391): newest
- [ ] [A Survey on Mixture of Experts](https://arxiv.org/abs/2407.06204)
- [ ] [LLM Inference Serving: Survey of Recent Advances and Opportunities](https://arxiv.org/abs/2407.12391): better than nothing
- [ ] [Contemporary Model Compression on Large Language Models Inference](https://arxiv.org/abs/2409.01990): survey in model compression
- [ ] ⭐ [Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning](https://www.computer.org/csdl/proceedings-article/sc/2024/529100b314/21HUWpZ2Xgk): bring insights for MLSys
- [ ] [Resource-efficient Algorithms and Systems of Foundation Models: A Survey](https://dl.acm.org/doi/abs/10.1145/3706418)
- [ ] ⭐ [A Survey on Inference Optimization Techniques for Mixture of Experts Models](https://arxiv.org/abs/2412.14219): asurvey on MoE models
- [ ] [Deploying Foundation Model Powered Agent Services: A Survey](https://arxiv.org/abs/2412.13437): survey for AI agent service
- [ ] [Resource-efficient Algorithms and Systems of Foundation Models: A Survey](https://dl.acm.org/doi/abs/10.1145/3706418)

Make useful benchmark or evaluation is helfpul.  

- [ ] [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549): [inference github](https://github.com/mlcommons/inference), a well-known benchmark
- [ ] [llmperf](https://github.com/ray-project/llmperf): evaluate both performance and correctness, but based on ray
- [ ] [The Importance of Workload Choice in Evaluating LLM Inference Systems](https://dl.acm.org/doi/abs/10.1145/3642970.3655823): important angles in LLM inference systems
- [ ] [Vidur: A Large-Scale Simulation Framework For LLM Inference](https://arxiv.org/abs/2405.05465): test the performance of LLM inference
- [ ] [Metron: Holistic Performance Evaluation Framework for LLM Inference Systems](https://arxiv.org/abs/2407.07000): an evaluation framework
- [ ] [LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale](https://arxiv.org/abs/2408.05499): a Simulator
- [ ] [LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators](https://arxiv.org/abs/2411.00136): inference + hardware
- [ ] [Towards Efficient Large Multimodal Model Serving](https://arxiv.org/abs/2502.00937): a survey on mm serving, and a decoupled serving architecture that enables independent resource allocation and adaptive scaling for each stage

- [ ] [LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf): a performance evaluation framework, can be used to estimate the time cost
- [ ] [Predicting LLM Inference Latency: A Roofline-Driven ML Method](https://neurips.cc/virtual/2024/103606): predict inference performance based on Roofline
- [ ] [GUIDE: A Global Unified Inference Engine for Deploying Large Language Models in Heterogeneous Environments](https://arxiv.org/abs/2412.04788): a work for predict LLMSys performance
- [ ] [TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415): simulator provide some performance analysis

### Technical reports of the enterprise

- [ ] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): deepseek: mla + moe
- [ ] [Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs](https://arxiv.org/abs/2503.05139): moe training with lower-specification hardware

### Interesting *NEW* Frameworks in Parallel Decoding

[Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads](https://sites.google.com/view/medusa-llm), [pdf](https://arxiv.org/pdf/2401.10774.pdf)  

prior paper: [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)

[Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/): by lookahead decoding  

Both frameworks use parallel decoding, and deserve a more detailed research.  

### Benchmark LLM Inference framework

- [vllm-project/aibrix](https://github.com/vllm-project/aibrix)

#### Papers for Parallel Decoding

There are some interesting papers about parallel decoding.  

- [ ] [Fast Chain-of-Thought: A Glance of Future from Parallel Decoding Leads to Answers Faster](https://arxiv.org/abs/2311.08263)  
- [ ] [Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding](https://arxiv.org/abs/2307.15337)  
- [ ] [ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding](https://arxiv.org/abs/2402.13485)
- [ ] [APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding](https://arxiv.org/abs/2401.06761): how to make it auto-parallel?

### Complex Inference

In fact, I'm not so familiar with with topic. But perhaps OpenAI 4o1 used this...  
Spend more time inferencing than pre-training  

- [ ] ⭐ [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787): Starter material, apply repeated sampling
- [ ] ⭐ [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314): Starter material, scaling LLM Test-Time to improve accuracy
- [ ] [Strategic Chain-of-Thought: Guiding Accurate Reasoning in LLMs through Strategy Elicitation](https://arxiv.org/abs/2409.03271): seems fewer people have explore the efficiency of CoT; a two-stage method gives me some throught
- [ ] [Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/abs/2410.20290): optimize alignment in inference, accepted by NIPS'24
- [ ] [S*: Test Time Scaling for Code Generation](https://arxiv.org/abs/2502.14382): perhaps can do some acceleration on Test Time Scaling

#### GPT-o1

This topic is about GPT-o1, aka the strawberry.  

- [ ] ⭐ [Reverse engineering OpenAI’s o1](https://www.interconnects.ai/p/reverse-engineering-openai-o1): a leading blog for introduction in OpenAI’s o1
- [ ] ⭐ [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903): base work
- [ ] [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601): a improment based on CoT
- [ ] [Large Language Model Guided Tree-of-Thought](https://arxiv.org/abs/2305.08291): also a ToT
- [ ] [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050): verify by step can be helpful
- [ ] [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406): what is Language Agent Tree Search (LATS)? accepted by ICML'24
- [ ] [Critique-out-Loud Reward Models](https://arxiv.org/abs/2408.11791)
- [ ] [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240): a verifier, by DeepMind

### Speculative Decoding

Also named as Speculative Sampling, model collaboration.  

- [x] ⭐ [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318): opening of *Speculative Decoding*, by DeepMind
- [x] ⭐ [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192): work of similar period with the upper one, by Google, accepted by ICML'23
- [x] [SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification](https://dl.acm.org/doi/10.1145/3620666.3651335): paper under guidance of Zhihao JIA, use Tree decoding and a set of draft models  
- [x] [LLMCad: Fast and Scalable On-device Large Language Model Inference](https://arxiv.org/pdf/2309.04255.pdf): paper under guidance of Xin JIN, speculative decoding for on-device LLM inference based on tree decoding and other optimizations  
- [ ] [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863): similar to speculative decoding, accepted in NIPS'23  
- [ ] [Online Speculative Decoding](https://arxiv.org/abs/2310.07177): update draft model online  
- [ ] [Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding](https://arxiv.org/pdf/2307.05908.pdf): the trade-off analyse deserves a reading
- [ ] [The Synergy of Speculative Decoding and Batching in Serving Large Language Models](https://arxiv.org/abs/2310.18813): analyse for combining the spec decoding with batching  
- [ ] [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252): use retrieval for spec decoding, some familiar names in the authors list  
- [ ] [Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462): by UIUC
- [ ] [Multi-Candidate Speculative Decoding](https://arxiv.org/abs/2401.06706): multiple draft models
- [ ] ⭐ [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851): survey for Speculative Decoding
- [ ] [BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models](https://arxiv.org/abs/2401.12522)
- [ ] [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [ ] [GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding](https://arxiv.org/abs/2402.02082): a work with Yang YOU's name
- [ ] [Decoding Speculative Decoding](https://arxiv.org/abs/2402.01528): provide some insight into the selection of draft models
- [ ] [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting](https://arxiv.org/abs/2402.13720): perhaps tree specualtive decoding?
- [ ] ⭐ [Speculative Streaming: Fast LLM Inference without Auxiliary Models](https://arxiv.org/abs/2402.11131): a promising method for speculative decoding
- [ ] [Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding](https://arxiv.org/abs/2402.12374): accelerating spec decoding
- [ ] [Chimera: A Lossless Decoding Method for Accelerating Large Language Models Inference by Fusing all Tokens](https://arxiv.org/abs/2402.15758): accelerate spec decoding with Fusing all tokens
- [ ] [Minions: Accelerating Large Language Model Inference with Adaptive and Collective Speculative Decoding](https://arxiv.org/abs/2402.15678): using several SSMs, adaptive SSM prediction length, pipelining SSM decode and LLM verify
- [ ] [Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://arxiv.org/abs/2403.09919)
- [ ] [Optimal Block-Level Draft Verification for Accelerating Speculative Decoding](https://arxiv.org/abs/2403.10444)
- [ ] [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/abs/2308.04623): token tree and a second stage of speculative decoding
- [ ] [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [ ] [TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding](https://arxiv.org/abs/2404.11912): combine KV cache with spec decoding
- [ ] [EMS-SD: Efficient Multi-sample Speculative Decoding for Accelerating Large Language Models](https://arxiv.org/abs/2405.07542): algorithm optimization in spec decoding
- [ ] [SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices](https://arxiv.org/abs/2406.02532): any difference with specinfer?
- [ ] [Optimizing Speculative Decoding for Serving Large Language Models Using Goodput](https://arxiv.org/abs/2406.14066): model the speculative decoding length
- [ ] [MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding](https://arxiv.org/abs/2408.11049): spec decoding for long-context
- [ ] [QSpec: Speculative Decoding with Complementary Quantization Schemes](https://arxiv.org/abs/2410.11305): spec decoding with quantization, a novel A+B
- [ ] [Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement](https://arxiv.org/abs/2410.13344): optimization ob Medusa
- [ ] [The N-Grammys: Accelerating autoregressive inference with learning-free batched speculation](https://www.amazon.science/publications/the-n-grammys-accelerating-autoregressive-inference-with-learning-free-batched-speculation): use learning-free, negligible-cost draft strategies, namely N-grams obtained from the model weights and the context
- [ ] [EdgeLLM: Fast On-device LLM Inference with Speculative Decoding](https://ieeexplore.ieee.org/abstract/document/10812936): seem a extended work of LLMCad
- [ ] [AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decodin](https://arxiv.org/abs/2501.12162): a speculation-and-selection scheme, that first constructs candidate token trees for each request and then dynamically selects tokens to meet individual SLO constraints
- [ ] [SpecServe: Efficient and SLO-Aware Large Language Model Serving with Adaptive Speculative Decoding](https://arxiv.org/abs/2503.05096): dynamically adjusts speculative strategies according to real-time request loads and system configurations
- [ ] [ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts](https://arxiv.org/abs/2503.13565): combining multi-level speculative decoding with MXFP4 quantized drafts, simple but work
- [ ] [SPIN: Accelerating Large Language Model Inference with Heterogeneous Speculative Models](https://arxiv.org/abs/2503.15921): using multiple heterogeneous SSMs with a learning-based algorithm for SSM selection, request decomposition method to minimize batching overhead during LLM verification, pipelining speculation and verification phases on GPU

#### different model collaboration  

- [ ] [Think Big, Generate Quick: LLM-to-SLM for Fast Autoregressive Decoding](https://arxiv.org/abs/2402.16844): use both LLM and SLM

#### Skeleton-of-Thought

- [ ] [Adaptive Skeleton Graph Decoding](https://arxiv.org/abs/2402.12280): successor of Skeleton-of-Thought

### 3D Parallelism 💡

Some knowledege about data parallel, model tensor parallel, and model pipeline parallel will help in this track.  

- [x] ⭐ [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102): use model parallel to accelerating inference, by Google, in MLSys'23
- [ ] [HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment](https://arxiv.org/abs/2311.11514):  a distributed inference engine that supports asymmetric partitioning of the inference computation
- [ ] [InternEvo: Efficient Long-sequence Large Language Model Training via Hybrid Parallelism and Redundant Sharding](https://arxiv.org/abs/2401.09149): Efficient Long-sequence training
- [x] [Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference](https://dl.acm.org/doi/abs/10.1145/3627535.3638466): accepted by PPoPP'24
- [ ] [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627): full-stack approach of LLM training
- [ ] [DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](https://arxiv.org/abs/2403.10266): sequence parallel by Yang YOU
- [x] [LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism](https://arxiv.org/abs/2404.09526): Elastic Sequence Parallelism?
- [ ] [GraphPipe: Improving Performance and Scalability of DNN Training with Graph Pipeline Parallelism](https://arxiv.org/abs/2406.17145): this could be potential in inference
- [ ] [TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models](https://proceedings.mlr.press/v139/li21y.html): pipeline parallism
- [ ] [QUART: Latency-Aware FaaS System for Pipelining Large Model Inference](https://ieeexplore.ieee.org/document/10631006): pipeline in serving and fast expanding
- [ ] [Mnemosyne: Parallelization Strategies for Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations](https://arxiv.org/abs/2409.17264): optimize sequence parallel
- [ ] [CSPS: A Communication-Efficient Sequence-Parallelism based Serving System for Transformer based Models with Long Prompts](https://arxiv.org/abs/2409.15104): optimize sequence parallel
- [ ] ⭐ [PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation](https://www.computer.org/csdl/proceedings-article/sc/2024/529100a624/21HUVz57ZgQ): pipeline parallelism and speculation, accepted by SC'24
- [ ] [HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment](https://arxiv.org/abs/2502.07903): algorithm analyse for resource allocation, parallel strategy and kv transfer in disaggreagting llm system
- [ ] [ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput](https://arxiv.org/abs/2503.04253): explores design spaces to suggest architectures that meet the requirements of both vendors and users
- [ ] [Seesaw: High-throughput LLM Inference via Model Re-sharding](https://mlsys.org/virtual/2025/poster/2974): dynamic model re-sharding, facilitates the dynamic reconfiguration of parallelization strategies across prefill-decode stages, accepted by MLSYS'25
- [ ] [PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training](https://mlsys.org/virtual/2025/poster/3004): fill the bubbles with other GPU workload

#### Communication Overlap

- [ ] [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959): overlap comm with comp, similar to Liger
- [ ] [Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning](https://dl.acm.org/doi/10.1145/3620666.3651379): accepted by ASPLOS'24
- [ ] [T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives](https://dl.acm.org/doi/10.1145/3620665.3640410): many work about overlap in LLM, accepted by ASPLOS'24
- [x] [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858): Fine-grained decomposition, perhaps provide some experiment result
- [ ] [Kraken: Inherently Parallel Transformers For Efficient Multi-Device Inference](https://arxiv.org/abs/2408.07802): modify the model design for fast decoding, based on comm-comp overlapping
- [x] [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757): overlaping based on nano-batch, with some interesting engineer implemntation
- [ ] [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/abs/2409.15241): overlapping, provided by Deepspeed team
- [ ] [PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving](https://arxiv.org/abs/2501.08192): overlap communication with model-weights/KV-cache prefetch
- [ ] [Concerto: Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning](https://dl.acm.org/doi/10.1145/3669940.3707223): use compilation to schedule overlap, accepted by ASPLOS'25

#### Prefill-Decode disaggregation

Ignore some of the earliest papers and focus on the latest work to optimize this.

- [ ] ⭐ [Seesaw: High-throughput LLM Inference via Model Re-sharding](https://arxiv.org/abs/2503.06433): dynamic model re-sharding to facilitates the dynamic reconfiguration of parallelization strategies across stages, reduce the overhead caused by frequent stage transitions (seems like Elastic Scheduling)
- [ ] [DynamicAttention: Dynamic KV Cache for Disaggregate LLM Inference](https://ieeexplore.ieee.org/abstract/document/10890367): DynamicAttention, it allocates a continuous virtual GPU memory space at startup, but does not actually allocate physical GPU memory?

### Prune & Sparsity 💡

An enduring topic in efficient machine learning.  
We mainly focus on Semi-structured and Structured pruning becasue they can accelerate computing.  

- [ ] ⭐ [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378): use N:M sparsity to fully utilize the hardware for accelerating, by Nvidia
- [ ] ⭐ [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://proceedings.mlr.press/v202/liu23am.html): interesting paper in using sparsity, under guidence of Tri DAO and Ce ZHANG, accepted in ICML'23
- [ ] [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers](https://arxiv.org/abs/2305.15805)
- [ ] [Dynamic N:M Fine-Grained Structured Sparse Attention Mechanism](https://dl.acm.org/doi/abs/10.1145/3572848.3577500): accepted by PPoPP'23
- [x] ⭐ [PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation](https://dl.acm.org/doi/10.1145/3600006.3613139): A novel way to deal with dynamic sparsity may be used for GNN and MoE, accepted by SOSP'23
- [ ] [DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving](https://arxiv.org/abs/2403.01876): seem a follow-up work of Deja Vu, also focus on KV-Cache

- [ ] [FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inferenc](https://arxiv.org/abs/2401.04044): sparsity in FFN
- [ ] [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516): a simple and effective sparsification method named "ProSparse"
- [ ] [Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters](https://arxiv.org/abs/2406.05955): work for powerinfo
- [ ] [Pruning Large Language Models to Intra-module Low-rank Architecture with Transitional Activations](https://arxiv.org/abs/2407.05690): pruning for LLM
- [ ] [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://arxiv.org/abs/2407.02490): inference framework based on sparse attention, by Microsoft
- [ ] [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://arxiv.org/abs/2310.04564): use ReLU to imporve Sparsity, just like powerinfer
- [ ] [CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation](https://arxiv.org/abs/2410.18311): algorithm optimization that can utilize sparsity to accelerate inference
- [ ] [Star Attention: Efficient LLM Inference over Long Sequences](https://arxiv.org/abs/2411.17116): a two-phase block-sparse approximation

- [ ] [Lexico: Extreme KV Cache Compression via Sparse Coding over Universal Dictionaries](https://arxiv.org/abs/2412.08890): use Sparse Coding over Universal Dictionaries to compress KV cache, it's novelty
- [ ] [SHARP: Accelerating Language Model Inference by SHaring Adjacent layers with Recovery Parameters](https://arxiv.org/abs/2502.07832): algorithm to replace a layer with the previous Adjacent layer and Recovery Parameters(based on finetune), to decrease memory overhead
- [ ] [Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking](https://mlsys.org/virtual/2025/poster/2972): accepted by MLSYS'25
- [ ] [SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs](https://dl.acm.org/doi/10.1145/3689031.3717481): Tensor-Core-Aware Bitmap Encoding (TCA-BME) and sparse Gemm kernel, make unstructured pruning's theoretical advantages translate into practical performance gains, EuroSys'25
- [ ] [Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores](https://dl.acm.org/doi/10.1145/3689031.3717455): EuroSys'25

### Quantization 💡

Low-precision for memory and computing efficiency.  

- [ ] [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948)
- [ ] ⭐ [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339): by UW  
- [ ] ⭐ [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438): paper under guidance of Song HAN  
- [ ] ⭐ [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978): paper under guidance of Song HAN  
- [x] [Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](https://arxiv.org/abs/2310.19102): paper under guidance of Tianqi CHEN, quantization is not important, designing how to quantify is important, in review of MLSys'24
- [ ] [FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs](https://arxiv.org/abs/2308.09723)
- [ ] [QUIK: Towards End-to-End 4-Bit Inference on Generative Large Language Models](https://arxiv.org/abs/2310.09259)  
- [ ] [Understanding the Impact of Post-Training Quantization on Large Language Models](https://arxiv.org/abs/2309.05210): tech report will help  
- [ ] ⭐ [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836): by HKUST, accepted in EMNLP'23
- [ ] ⭐ [Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization](https://arxiv.org/pdf/2311.16442.pdf): by SJTU, accepted in DAC'24
- [ ] [INT4 Wight + FP8 KV-Cache: optimization for LLM inference](https://zhuanlan.zhihu.com/p/653735572): INT4 Wight + FP8 KV-Cache + Continues batching
- [ ] [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- [ ] [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079): quant KV cache
- [ ] [QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference](https://arxiv.org/abs/2402.10076): simple and crude optimization work
- [ ] [LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization](https://arxiv.org/abs/2403.01136): for Heterogeneous Clusters and Adaptive Quantization, under guidence of Chuan WU, accepted by PPoPP'24(poster)  
- [ ] [IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact](https://arxiv.org/abs/2403.01241): use pivot token
- [ ] [QAQ: Quality Adaptive Quantization for LLM KV Cache](https://arxiv.org/abs/2403.04643)
- [ ] [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532): quantization in inference, under guidence of Song HAN
- [ ] [Does compressing activations help model parallel training?](https://arxiv.org/abs/2301.02654): analyse in compressing(including pruning and quantization) in MP training, accepted by MLSys'24
- [ ] [Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression](https://arxiv.org/abs/2405.12591): compress KV cache with quantization
- [ ] [Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs](https://arxiv.org/abs/2405.14428): with targeted activate function
- [ ] [FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design](https://arxiv.org/abs/2401.14112): FPx quantization, accepted by ATC'24
- [ ] [Demystifying the Compression of Mixture-of-Experts Through a Unified Framework](https://arxiv.org/abs/2406.02500): combine quantization with MoE
- [ ] [Does Compressing Activations Help Model Parallel Training?](https://proceedings.mlsys.org/paper_files/paper/2024/hash/71381211d0abef73ed1887b83c4547b1-Abstract-Conference.html): quantization Activation?
- [ ] [PQCache: Product Quantization-based KVCache for Long Context LLM Inference](https://arxiv.org/abs/2407.12820): apply quantization and Maximum Inner-Product Search for KV Cache compression
- [ ] [Fast Matrix Multiplications for Lookup Table-Quantized LLMs](https://arxiv.org/abs/2407.10960): provide efficient kernels for lookup quantization
- [ ] [Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation](https://www.usenix.org/conference/osdi24/presentation/wang-lei): a computation optimization for Low-Precision
- [ ] [Quant-LLM: Accelerating the Serving of Large Language Models via FP6-Centric Algorithm-System Co-Design on Modern GPUs](https://www.usenix.org/conference/atc24/presentation/xia): a computation optimization for 6-bit LLM
- [ ] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417): quantization on MoE models
- [ ] [Zero-Delay QKV Compression for Mitigating KV Cache and Network Bottlenecks in LLM Inference](https://arxiv.org/abs/2408.04107): compress the KV Cache
- [ ] [ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for Large Language Models](https://arxiv.org/abs/2408.08554): quantization matrix multiplication of arbitrary precision combinations based on BTC (Binary TensorCore) equivalents
- [ ] [Progressive Mixed-Precision Decoding for Efficient LLM Inference](https://arxiv.org/abs/2410.13461): gradual lowering of precision deeper in the generated sequence, together with a spectrum of precision-switching schedulers
- [ ] [COMET: Towards Partical W4A4KV4 LLMs Serving](https://arxiv.org/abs/2410.12168): provide quantization algorithm, quantization kernel and SM schedule method
- [ ] [MixQ: Taming Dynamic Outliers in Mixed-Precision Quantization by Online Prediction](https://www.computer.org/csdl/proceedings-article/sc/2024/529100b161/21HUWiUMiqI): quantization with outliers, optimization on AWQ, accepted by SC'24
- [ ] [Flash Communication: Reducing Tensor Parallelization Bottleneck for Fast Large Language Model Inference](https://arxiv.org/abs/2412.04964): low-bit compression to accelerate communication
- [ ] [Unifying KV Cache Compression for Large Language Models with LeanKV](https://arxiv.org/abs/2412.03131): combine quantization and sparity to compress KV cache
- [ ] [MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design](https://arxiv.org/abs/2412.14590): mix quantization, effectively assigning the larger bit-width to output features that need it most to achieve good accuracy with low memory consumption
- [ ] [KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference](https://arxiv.org/abs/2502.04420): KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference
- [ ] [HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference](https://arxiv.org/abs/2502.03589): quantization to decrease kvc transfer overhead in disaggregation and eliminate kv dequantization
- [ ] [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743): Mixed-precision Auto-Regressive LINear kernels, accepted by PPoPP'25
- [ ] [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://mlsys.org/virtual/2025/poster/2987): augments highly quantized MoEs with a mixture of low-rank compensators, provide 3-bit tensorcore kernels, accepted by MLSYS'25
- [ ] [PacQ: A SIMT Microarchitecture for Efficient Dataflow in Hyper-asymmetric GEMMs](https://arxiv.org/abs/2502.18627): accelerator design, but may be helpful
- [ ] [Cocktail: Chunk-Adaptive Mixed-Precision Quantization for Long-Context LLM Inference](https://arxiv.org/abs/2503.23294): based on mixed-precision quantization to the key-value (KV) cache in LLMs based on token granularity, do quantization on KV cache chunk-level
- [ ] [Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization](https://arxiv.org/abs/2503.18599): employs an online-offline hybrid approach, setting outlier thresholds offline, which are then used to determine the quantization scale online
- [ ] [SQuat: Subspace-orthogonal KV Cache Quantization](https://arxiv.org/abs/2503.24358): a more efficient quantization algorithm(?)

### Batch Processing

Perhaps the most important way for improving the throughput in LLM inference.  
This blog [Dissecting Batching Effects in GPT Inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) helps me a lot at the beginning.  

*Update2023/12/12: I'd like to use `Continues Batching` to take place of the `Dynamic Batching` I used before.* The name `Dynamic Batching` is more likely to be used in [Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/examples/jetson/concurrency_and_dynamic_batching/README.html).  

- [x] ⭐ [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu): Continues batch processing without redundant computing, accepted in OSDI'23  
- [x] [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920): considering Job Completion Time(JCT) in LLM serving, paper under guidance of Xin JIN  
- [ ] [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144): schedule based on response length prediction by LLM, paper under guidance of Yang YOU  
- [ ] [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput](https://arxiv.org/abs/2306.06000): idea similar to above, by Harvard University  
- [x] [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369): blocking the prefill phase and reduce pipeline bubbles, by MSRIndia  
- [ ] [Flover: A Temporal Fusion Framework for Efficient Autoregressive Model Parallel Inference](https://arxiv.org/abs/2305.13484): accepted by HiPC'23  
- [ ] [Handling heavy-tailed input of transformer inference on GPUs](https://dl.acm.org/doi/10.1145/3524059.3532372): accepted by ICS'22  
- [ ] [CoFB: latency-constrained co-scheduling of flows and batches for deep learning inference service on the CPU–GPU system](https://link.springer.com/article/10.1007/s11227-023-05183-6): Some form of inference service  
- [ ] [TCB: Accelerating Transformer Inference Services with Request Concatenation](https://dl.acm.org/doi/10.1145/3545008.3545052): perhaps similar to ByteTransformer, accepted by ICPP'22  
- [ ] [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588): under guidence of Ion Stoica, accepted by OSDI'24
- [ ] [Characterizing and understanding deep neural network batching systems on GPUs](https://www.sciencedirect.com/science/article/pii/S2772485924000036): benchmarking is important
- [ ] [Hydragen: High-Throughput LLM Inference with Shared Prefixes](https://arxiv.org/abs/2402.05099)
- [ ] [RelayAttention for Efficient Large Language Model Serving with Long System Prompts](https://arxiv.org/abs/2402.14808): think about the memory access of KV cache
- [ ] [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310): follow-up work of sarathi
- [ ] [Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction](https://arxiv.org/abs/2404.08509): predict length
- [ ] [LiveMind: Low-latency Large Language Models with Simultaneous Inference](https://arxiv.org/abs/2406.14319): perform inferences with incomplete prompts, to take advantage of streaming prompt
- [ ] [A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length](https://arxiv.org/abs/2407.05347): theoretical analysis of latency
- [ ] [ElasticBatch: A Learning-Augmented Elastic Scheduling System for Batch Inference on MIG](https://ieeexplore.ieee.org/abstract/document/10605084)
- [ ] [Prepacking: A Simple Method for Fast Prefilling and Increased Throughput in Large Language Models](https://arxiv.org/abs/2404.09529): seems similar to ORCA or bytetransformer?
- [ ] [BATON: Enhancing Batch-wise Inference Efficiency for Large Language Models via Dynamic Re-batching](https://arxiv.org/abs/2410.18701): optimization on ORCA, dynamic re-batching
- [ ] [EcoServe: Maximizing Multi-Resource Utilization with SLO Guarantees in LLM Serving](https://arxiv.org/abs/2411.06364): A fusion monster with a variety of optimization techniques
- [ ] ⭐ [AcceLLM: Accelerating LLM Inference using Redundancy for Load Balancing and Data Locality](https://arxiv.org/abs/2411.05555): what's Redundancy
- [ ] [Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching](https://arxiv.org/abs/2503.05248): formalize as an optimization problem and adjust the batch size based on this

### Computing Optimization

This part include some impressive work optimizing LLM computing by observing the underlying computing properties. Such as FlashAttention, et.al.

#### FlashAttention Family

- [ ] ⭐ [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): one of the most important work these years, both simple and easy to use, by Tri DAO
- [ ] ⭐ [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691): you'd better not ignore it  
- [ ] ⭐ [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html): you'd better not ignore it, too  
- [ ] ⭐ [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285): successor to FlashAttention in inference, accepted by VLDB'24
- [x] ⭐ [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282): worth reading, FLashDecoding follow-up  
- [ ] [SubGen: Token Generation in Sublinear Time and Memory](https://arxiv.org/abs/2402.06082)
- [ ] [DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference](https://arxiv.org/abs/2404.00242)
- [ ] [Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers](https://arxiv.org/abs/2405.10480): modification in self-attention
- [ ] [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [ ] [Flex Attention: A Programming Model for Generating Optimized Attention Kernels](https://arxiv.org/abs/2412.05496): auto-generated attention kernel

#### Optimization focus on Auto-regressive Decoding

- [x] [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677): splitting prefill and decode in a map-reduce style, by UW and Microsoft
- [x] [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670): also split the prefill and decode, accepted by OSDI'24
- [x] [Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/abs/2401.11181): seems a combination of SARATHI and Splitwise
- [ ] [ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference](https://dl.acm.org/doi/10.1145/3620665.3640383): similar to splitwise, accepted by ASPLOS'24
- [ ] [Splitwiser: Efficient LLM Inference with Constrained Resources](https://asadaali.com/assets/pdf/paper_splitwiser.pdf)
- [ ] [ToEx: Accelerating Generation Stage of Transformer-based Language Models via Token-adaptive Early Exit](https://ieeexplore.ieee.org/abstract/document/10535998): Token-adaptive Early Exit

#### Kernels Optimization

- [ ] [Automatic Task Parallelization of Dataflow Graphs in ML/DL models](https://arxiv.org/abs/2308.11192)
- [ ] [MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures](https://www.usenix.org/conference/osdi24/presentation/zhuang): compilation optimization on compuataion graph
- [ ] [POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference](https://arxiv.org/abs/2410.18038): optimize attention kernel in mix-batching
- [ ] [Focus: High-Performant and Customizable Attention Engine for LLM Serving](https://mlsys.org/virtual/2025/poster/2980): flexible attention engine, advised by Chen Tianqi and accepted by MLSYS'25
- [ ] [ML-Triton, A Multi-Level Compilation and Language Extension to Triton GPU Programming](https://arxiv.org/abs/2503.14985): Multi-level Triton

### Memory Manage

This part is inspired by PagedAttention of vLLM. And there are many Top-Conference paper discussing the memory management in DL computing on GPUs.  

- [x] ⭐ [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): memory page management for the KV-Cache in Attention-type model, accepted by SOSP'23 (many papers will cite the vLLM project instead of their paper, which makes it harder for us to find its *citated by*)
- [ ] ⭐ [AutoScratch: ML-Optimized Cache Management for Inference-Oriented GPUs](https://proceedings.mlsys.org/paper_files/paper/2023/hash/627b5f83ffa130fb33cb03dafb47a630-Abstract-mlsys2023.html): cache management for inference, accepted by MLSys'23
- [ ] [Improving Computation and Memory Efficiency for Real-world Transformer Inference on GPUs](https://dl.acm.org/doi/full/10.1145/3617689): block-based data layout, accepted by TACO'October-2023
- [ ] [AttMEMO : Accelerating Transformers with Memoization on Big Memory Systems](https://arxiv.org/abs/2301.09262): a unique observation that there is rich similarity in attention computation across inference sequences
- [ ] [BPIPE: memory-balanced pipeline parallelism for training large language models](https://dl.acm.org/doi/10.5555/3618408.3619090): memory balance perhaps can work well in inferencce, by SNU, accepted by ICML'23
- [ ] [Improving Large Language Model Throughput with Efficient LongTerm Memory Management](https://people.eecs.berkeley.edu/~kubitron/courses/cs262a-F23/projects/reports/project1010_paper_64287652274076362722.pdf): perhaps a new view
- [ ] [CacheGen: Fast Context Loading for Language Model Applications](https://arxiv.org/abs/2310.07240)
- [x] [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669)
- [ ] [Quantized Side Tuning: Fast and Memory-Efficient Tuning of Quantized Large Language Models](https://arxiv.org/abs/2401.07159): consider the memory consumption in fine-tuning
- [ ] [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/abs/2402.09398)
- [ ] [Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/abs/2403.09636): compress KV Cache
- [ ] [LLM as a System Service on Mobile Devices](https://arxiv.org/abs/2403.11805): LLM as a service on Mobile devices
- [ ] [DistMind: Efficient Resource Disaggregation for Deep Learning Workloads](https://ieeexplore.ieee.org/abstract/document/10414009): by Xin JIN, accepted by ToN'Jan24
- [ ] [ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching](https://arxiv.org/abs/2403.17312): sparsity in KV Cache, accepted by ISCA'24
- [ ] [AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving](https://arxiv.org/abs/2403.19708): a hierarchical KV caching system that leverages cost-effective memory/storage mediums to save KV caches for all requests
- [ ] [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437): improve PagedAttention
- [ ] [Layer-Condensed KV Cache for Efficient Inference of Large Language Models](https://arxiv.org/abs/2405.10637): only computes and caches the KVs of a small number of layers
- [ ] [MiniCache: KV Cache Compression in Depth Dimension for Large Language Models](https://arxiv.org/abs/2405.14366): compress KV cache
- [ ] [CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444): very popular idea recently
- [ ] [Block Transformer: Global-to-Local Language Modeling for Fast Inference](https://arxiv.org/abs/2406.02657): build KV Cache block from many tokens' KV Cache
- [ ] [MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool](https://arxiv.org/abs/2406.17565): KV Cache management in P/D disaggregation arch
- [ ] [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost): multi-round chat and memory management, accepted by ATC'24
- [ ] [Stateful Large Language Model Serving with Pensieve](https://arxiv.org/abs/2312.05516): similar to cachedattention
- [ ] [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079): P/D disaggregation archtecture and KV Cache management
- [ ] [P/D-Serve: Serving Disaggregated Large Language Model at Scale](https://arxiv.org/abs/2408.08147): a P/D based system, with D2D access optimization
- [ ] [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://www.usenix.org/conference/osdi24/presentation/lee): offload KV Cache
- [ ] [Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption](https://arxiv.org/abs/2407.18003): a survey for optimizing KV Cache
- [ ] [vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving](https://arxiv.org/abs/2407.15309): tensor management especially for llm inference
- [ ] [Token-Picker: Accelerating Attention in Text Generation with Minimized Memory Transfer via Probability Estimation](https://arxiv.org/abs/2407.15131): remove unimportant tokens in KV Cache  
- [ ] [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://arxiv.org/abs/2310.07240): compression and streaming transfering of KV Cache, accepted by SIGCOMM'24
- [ ] [Compute Or Load KV Cache? Why Not Both?](https://arxiv.org/abs/2410.03065): recompute and load together for long context
- [ ] [LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management](https://arxiv.org/abs/2410.00428): manage KV Cache by layers
- [ ] [Harnessing Your DRAM and SSD for Sustainable and Accessible LLM Inference with Mixed-Precision and Multi-level Caching](https://arxiv.org/abs/2410.14740): compress KV cache and multi-level memory
- [ ] [EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models](https://arxiv.org/abs/2410.15332): better prefix-cache
- [ ] [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://arxiv.org/abs/2410.21465): Low-rank KV cache and dynamic rebuild KV cache
- [ ] ⭐ [VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration](https://arxiv.org/abs/2410.23317): the first work I see that optimize KV cache in vision models
- [ ] [ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction](https://openreview.net/forum?id=4oAt5L4lYe): KV cache page evict and recall, accepted by NIPS'24
- [ ] [SpeedLoader: An I/O efficient scheme for heterogeneous and distributed LLM operation](https://openreview.net/forum?id=Y2I0Fy4sm7): Optimization on Zero? redesign the data flow of heterogeneous hardware and sharded model training to minimize the excessive communication overhead, accepted by NIPS'24
- [ ] ⭐ [KunServe: Elastic and Efficient Large Language Model Serving with Parameter-centric Memory Management](https://arxiv.org/abs/2412.18169): memory management for KV cache and parameter, seems a novel work considering the weights migration
- [ ] [SYMPHONY: Improving Memory Management for LLM Inference Workloads](https://arxiv.org/abs/2412.16434): dynamically migrates K,V caches to enable finegrained scheduling of inference requests
- [ ] [Mell: Memory-Efficient Large Language Model Serving via Multi-GPU KV Cache Management](https://arxiv.org/abs/2501.06709): efficiently migrate requests and their KV cache among GPUs
- [ ] [Efficient LLM Inference with Activation Checkpointing and Hybrid Caching](https://arxiv.org/abs/2501.01792): recompute+cache for KV cache management, only recompute attention(no projection)

- [ ] [Memory Offloading for Large Language Model Inference with Latency SLO Guarantees](https://arxiv.org/abs/2502.08182): offload kv cache to CPU memory
- [ ] [Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving](https://arxiv.org/abs/2503.00392): sparse attention is hot recently, dynamic kvcache budget and efficient kvc loading from CPU
- [ ] [Efficient and scalable huge embedding model training via distributed cache management](https://link.springer.com/article/10.1007/s00778-025-00908-w): staleness and skewed popularity distributions based cache
- [ ] [BaKlaVa -- Budgeted Allocation of KV cache for Long-context Inference](https://arxiv.org/abs/2502.13176): different kv heads have different importance, then offload and compress
- [ ] [Fast State Restoration in LLM Serving with HCache](https://arxiv.org/abs/2410.05004): cache for offloading kvc to CPU, accepted by EuroSys'25
- [ ] [Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/abs/2503.08311): use model replication to improve serving throughput and GPU utilization?
- [ ] [Characterizing the Behavior and Impact of KV Caching on Transformer Inferences under Concurrency](https://inria.hal.science/hal-04984000/): instrument vLLM to measure and analyze fine-grain metrics (token throughput, KV cache memory access patterns, load balancing of the forward passes), during different inference stages (prefill, decode, batching and KV cache eviction policies) in several scenarios
- [ ] [Mitigating KV Cache Competition to Enhance User Experience in LLM Inference](https://arxiv.org/abs/2503.13773): mitigating KV Cache competition with several technology
- [ ] [Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache](https://arxiv.org/abs/2503.14647): KV cache reusing is able to save cloud cost across a range of workloads with long context
- [ ] [KVSort: Drastically Improving LLM Inference Performance via KV Cache Compression](https://sc24.supercomputing.org/proceedings/poster/poster_files/post189s2-file3.pdf): error-bounded lossy compression on sorted KV vectors
- [ ] [FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework](https://arxiv.org/abs/2503.08461): dynamic batching and kv cache pool in MM kv cache compression, guided by Jidong ZHAI
- [ ] [Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management](https://dl.acm.org/doi/abs/10.1145/3676641.3716245): multi-level KV cache management(an idea lack innovation) and request reorder, accepted by ASPLOS'25
- [ ] [Aqua: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains](https://dl.acm.org/doi/abs/10.1145/3676641.3715983): memory management framework for a sudden increase in the number of inference requests to a cloud-hosted LLM, accepted by ASPLOS'25
- [ ] ⭐ [Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://arxiv.org/abs/2503.18292): optimization on PagedAttention, targeted at heterogeneous embeddings in LLMs

#### Prefix Sharing

note: some papers about prefix sharing is not in this section

- [ ] [LLM Query Scheduling with Prefix Reuse and Latency Constraints](https://arxiv.org/abs/2502.04677): balancing prefix reuse and fairness in query scheduling

### Inference on hardware: GPUs, CPUs or based on SSD

- [ ] [Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective](https://arxiv.org/abs/2410.04466): a helpful survey
- [ ] [ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput](https://arxiv.org/abs/2503.04253): balances throughput and latency under different hardware

#### Underlying optimization for GPU

- [ ] [Reducing shared memory footprint to leverage high throughput on Tensor Cores and its flexible API extension library](https://dl.acm.org/doi/abs/10.1145/3578178.3578238): implement some APIs to reduce the shared memory footprint, accepted in HPC Asia'23
- [ ] [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499): help us understand GPUs
- [ ] [SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving](https://arxiv.org/abs/2408.05235): optimizing energy consuming based on lower GPU frequency
- [ ] [Foreseer: Knowledge-Driven Acceleration of Memory-Bound Matrix Multiplications for Large Language Model Inference](https://dl.acm.org/doi/abs/10.1145/3688351.3689153): similar to cutlass, optimization on intel GPU
- [ ] [Tackling the Dynamicity in a Production LLM Serving System with SOTA Optimizations via Hybrid Prefill/Decode/Verify Scheduling on Efficient Meta-kernels](https://arxiv.org/abs/2412.18106): for Ascend GPU (perhaps also work for NVIDIA?)
- [ ] [MEPipe: Democratizing LLM Training with Memory-Efficient Slice-Level Pipeline Scheduling on Cost-Effective Accelerators](https://dl.acm.org/doi/10.1145/3689031.3717469): maybe inference on RTX4090?

#### CPUs or based on SSD

Heterogeneous scenarios or single PC are becoming increasingly important.  

Making optimization for the calculating on CPU or SSD will have different methods.  

- [ ] [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502): LLMs with quantization on CPUs, by Intel, accepted by NIPS'23
- [ ] [Inference Performance Optimization for Large Language Models on CPUs](https://arxiv.org/abs/2407.07304): xFasterTransformer, LLMs inference optimization on CPUs, by Intel
- [ ] [Distributed Inference Performance Optimization for LLMs on CPUs](https://arxiv.org/abs/2407.00029): similar work to above, by Intel

- [ ] [Exploiting Intel Advanced Matrix Extensions (AMX) for Large Language Model Inference](https://ieeexplore.ieee.org/abstract/document/10538369): inference on CPU based on advanced hardware
- [ ] [TURNIP: A "Nondeterministic" GPU Runtime with CPU RAM Offload](https://arxiv.org/abs/2405.16283): free to run operations such as GPU kernel calls in many different orders
- [ ] [Improving Throughput-oriented Generative Inference with CPUs](https://dl.acm.org/doi/abs/10.1145/3609510.3609815): cooperate of CPUs and GPU, accepted by APSys'23  
- [ ] [Chrion: Optimizing Recurrent Neural Network Inference by Collaboratively Utilizing CPUs and GPUs](https://arxiv.org/abs/2307.11339): execute the operators on the CPU and GPU in parallel, by SJTU
- [ ] [EdgeNN: Efficient Neural Network Inference for CPU-GPU Integrated Edge Devices](https://ieeexplore.ieee.org/document/10184528): inference on edge devices, accepted by ICDE'23
- [ ] [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456): by SJTU IPADS
- [ ] [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514): by Apple

- [ ] [Efficient LLM inference solution on Intel GPU](https://arxiv.org/abs/2401.05391): intel GPU is interesting
- [x] [FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines](https://openreview.net/forum?id=GahfuPsGw2): efficient serving with CPU-GPU system
- [ ] [Efficient and Economic Large Language Model Inference with Attention Offloading](https://arxiv.org/abs/2405.01814): similar to FastDecode
- [ ] [Glinthawk: A Two-Tiered Architecture for High-Throughput LLM Inference](https://arxiv.org/abs/2501.11779): similar to fastdecode: cpu for attention and gpu for others
- [ ] [Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188): looks like heterogeneous resources are being utilized

- [ ] [Efficient Inference on CPU](https://huggingface.co/docs/transformers/v4.34.0/en/perf_infer_cpu)
- [ ] [CPU inference](https://huggingface.co/docs/transformers/en/perf_infer_cpu)
- [ ] [NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention](https://arxiv.org/abs/2403.01273)
- [ ] ⭐ [A Quantitative Analysis and Guidelines of Data Streaming Accelerator in Modern Intel Xeon Scalable Processors](https://dl.acm.org/doi/10.1145/3620665.3640401): use CPU for DL, accepted by ASPLOS'24
- [ ] [LM-Offload: Performance Model-Guided Generative Inference of Large Language Models with Parallelism Control](https://pasalabs.org/papers/2024/llm_offload_2024.pdf): based on offload
- [ ] [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088): computation on CPU with quantization
- [ ] [TBA: Faster Large Language Model Training Using SSD-Based Activation Offloading](https://arxiv.org/abs/2408.10013): how to use SSD?
- [ ] [InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference](https://arxiv.org/abs/2409.04992): offload KV Cache to CSD(Computational Storage Drive)
- [ ] [TwinPilots: A New Computing Paradigm for GPU-CPU Parallel LLM Inference](https://jiangs.utasites.cloud/pubs/papers/Yu24-TwinPilots.pdf): some idea in using CPU
- [ ] [Improving Throughput-oriented LLM Inference with CPU Computations](https://dl.acm.org/doi/abs/10.1145/3656019.3676949): pipeline in CPU-GPU inference
- [ ] [Understanding Performance Implications of LLM Inference on CPUs](https://seonjinna.github.io/assets/pdf/iiswc24_CPULLM.pdf): analyse of using CPU for inference
- [ ] [GPUs, CPUs, and... NICs: Rethinking the Network's Role in Serving Complex AI Pipelines](https://arxiv.org/abs/2502.15712): NIC can be important, especially in communication

- [ ] [Pie: Pooling CPU Memory for LLM Inference](https://arxiv.org/abs/2411.09317): use CPU memory to enlarge batchsize to improve throughput, by Ion Stoica
- [ ] [NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference](https://arxiv.org/abs/2411.01142): offload KV cache and attention to CPU for larger batchsize, similar to fastdecode, by Ion Stoica
- [ ] [Task Scheduling for Efficient Inference of Large Language Models on Single Moderate GPU Systems](https://arxiv.org/abs/2411.15715): more likely inference on personal device
- [ ] [Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation](https://arxiv.org/abs/2411.17089): use recomputation and transfer to re-produce KV cache; can use their run-time and split parallelism

#### Inference on personal device

Inspired by AI PC, open up a new area.  
Including edge systems now.  

- [ ] [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865): inference a 30B model with a 16GB GPU, accepted by ICML'23
- [ ] [LLM as a System Service on Mobile Devices](https://arxiv.org/abs/2403.11805): an intro for LLM on private devices
- [ ] [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456): based on sparsity in NN Layers
- [ ] ⭐ [LLM for Mobile: An Initial Roadmap](https://arxiv.org/abs/2407.06573): a road map
- [ ] [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/abs/2406.06282): work on smartphone
- [ ] [Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM](https://arxiv.org/abs/2409.15654): on edge devices, accepted by MICRO'24
- [ ] ⭐ [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators](https://www.preprints.org/manuscript/202501.0901/v1): features on mobile SoCs, tensor partition strategy, to do Heterogeneous AI inference
- [ ] [PICE: A Semantic-Driven Progressive Inference System for LLM Serving in Cloud-Edge Networks](https://arxiv.org/abs/2501.09367): cloud(LLM)-edge(SmallLM) collaboration
- [ ] [FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference](https://arxiv.org/abs/2503.03777): offloading based framework, asynchronous prefetching, balanced memory locking, and flexible tensor preservation
- [ ] [Fast On-device LLM Inference with NPUs](https://dl.acm.org/doi/10.1145/3669940.3707239): chunked prefill, offload outlier to CPU/GPU, schedule computation to NPU/CPU/GPU, accepted by ASPLOS'25
- [ ] [FlexInfer: Flexible LLM Inference with CPU Computations](https://mlsys.org/virtual/2025/poster/2955): offload kvc and weights to CPU, accepted by MLSYS'25
- [ ] [An Adaptive and Scalable Framework for Resource-Efficient Deployment of Mixture of Experts in LLM-Based Intelligent IoT Networks](https://ieeexplore.ieee.org/abstract/document/10945759): deploy MoE on IoT, but the strategies are commonly used
- [ ] [A Novel Hat-Shaped Device-Cloud Collaborative Inference Framework for Large Language Models](https://arxiv.org/abs/2503.18989): can learn the edge-cloud serving from this paper, based on speculation decode
- [ ] [HERA: Hybrid Edge-cloud Resource Allocation for Cost-Efficient AI Agents](https://arxiv.org/abs/2504.00434): assign sub-tasks of LLM agent to local SLM and cloud-side LLM

#### Heterogeneous or decentralized environments

- [ ] [FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs](https://arxiv.org/abs/2309.01172): decentrailized system on consumer-level GPUs, through there will be some problems
- [ ] [Distributed Inference and Fine-tuning of Large Language Models Over The Internet](https://arxiv.org/abs/2312.08361): some techs in this paper will be instructive

- [ ] ⭐ [HeteGen: Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices](https://arxiv.org/abs/2403.01164): heterogeneous parallel computing using CPUs and GPUs
- [ ] [Metis: Fast Automatic Distributed Training on Heterogeneous GPUs](https://www.usenix.org/conference/atc24/presentation/um): accepted by ATC'24
- [ ] [Helix: Distributed Serving of Large Language Models via Max-Flow on Heterogeneous GPUs](https://dl.acm.org/doi/abs/10.1145/3669940.3707215): we can get performance model for Heterogeneous GPUs cluster and learn the algorithm analyse
- [ ] [Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity](https://arxiv.org/abs/2404.14527): making heterogeneity-aware GPU provisioning decisions for LLM serving

### Algorithm Optimization 💡

In this part, researchers provide some algorithm-based method to optimizing LLM inference.  

- [x] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048): accepted by NIPS'23
- [ ] [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time](https://arxiv.org/abs/2305.17118): consider the different importance of tokens in KV Cache, similar to H2O
- [ ] ⭐ [SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference](https://arxiv.org/abs/2307.02628): skipping maybe an useful method like spec decoding
- [ ] [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487): also a potential optimization
- [ ] [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453): streaming LLM for infinite sequence lengths, by MIT and under guidence of Song HAN
- [ ] [Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference](https://proceedings.mlsys.org/paper_files/paper/2024/hash/48fecef47b19fe501d27d338b6d52582-Abstract-Conference.html): also important tokens, just like H2O, accepted by MLSys'24
- [ ] [Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache](https://proceedings.mlsys.org/paper_files/paper/2024/hash/bbb7506579431a85861a05fff048d3e1-Abstract-Conference.html): an optimization to H2O, accepted by MLSys'24
- [ ] [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](https://arxiv.org/abs/2409.10516): use approximate nearest neighbor search to search the most relevant KV cache
- [ ] [CritiPrefill: A Segment-wise Criticality-based Approach for Prefilling Acceleration in LLMs](https://arxiv.org/abs/2409.12490): based on observation: adjacent query tokens tend to focus on similar subsets of the past Key-Value (KV) cache
- [ ] [TidalDecode: Fast and Accurate LLM Decoding with Position Persistent Sparse Attention](https://arxiv.org/abs/2410.05076): sparse attention
- [ ] [SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation](https://arxiv.org/abs/2410.03960): algorithm optimization for less KV Cache
- [ ] [Activation Sequence Caching: High-Throughput and Memory-Efficient Generative Inference with a Single GPU](https://dl.acm.org/doi/abs/10.1145/3656019.3676945): use characterization results to optimize KV Cache management

### Industrial Inference Frameworks 💡

- [ ] ⭐ [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032): you must know DeepSpeed  
- [ ] [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
- [ ] [DeepSpeed Model Implementations for Inference (MII)](https://github.com/microsoft/DeepSpeed-MII)
- [x] [ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs](https://arxiv.org/abs/2210.03052): developed by ByteDance, accepted by IPDPS'23
- [ ] [TurboTransformers: an efficient GPU serving system for transformer models](https://dl.acm.org/doi/10.1145/3437801.3441578): by Tencent Inc, accepted by PPoPP'21  
- [ ] [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/?utm_content=273712248&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024): a blog in PyTorch, use only PyTorch code, [gpt-fast](https://github.com/pytorch-labs/gpt-fast)
- [ ] [FlexFlow Serve: Low-Latency, High-Performance LLM Serving](https://github.com/flexflow/FlexFlow): based on FlexFlow
- [ ] [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
- [ ] [Efficiently Programming Large Language Models using SGLang](https://arxiv.org/abs/2312.07104): we can get some optimization from here
- [ ] [Inferflow: an Efficient and Highly Configurable Inference Engine for Large Language Models](https://arxiv.org/abs/2401.08294): different parallel, by Tencent

### LLM Serving 💡

LLM server providers will focus on this part. Engineering practices are just as important as algorithm optimization.  

- [ ] ⭐ [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665): accepted by OSDI'23
- [ ] ⭐ [STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining](https://arxiv.org/abs/2207.05022): Elastic will be important in the future, accepted by ASPLOS'23  
- [ ] [INFaaS: Automated Model-less Inference Serving](https://www.usenix.org/conference/atc21/presentation/romero): accepted by ATC'21  
- [ ] [Tabi: An Efficient Multi-Level Inference System for Large Language Models](https://dl.acm.org/doi/abs/10.1145/3552326.3587438): under guidence of Kai CHEN, accepted by EuroSys'23  
- [ ] [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176): cost is the service provider cares most  
- [ ] [FaaSwap: SLO-Aware, GPU-Efficient Serverless Inference via Model Swapping](https://arxiv.org/abs/2306.03622)
- [ ] [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning](https://www.usenix.org/conference/nsdi23/presentation/zheng): accepted by NSDI'23
- [ ] [Cocktail: A Multidimensional Optimization for Model Serving in Cloud](https://www.usenix.org/conference/nsdi22/presentation/gunasekaran): model ensembling, accepted by NSDI'22

- [ ] [SLA-Driven ML INFERENCE FRAMEWORK FOR CLOUDS WITH HETEROGENEOUS ACCELERATORS](https://mlsys.org/virtual/2022/poster/2034): accepted by MLSys'22
- [ ] [FaST-GShare: Enabling Efficient Spatio-Temporal GPU Sharing in Serverless Computing for Deep Learning Inference](https://dl.acm.org/doi/abs/10.1145/3605573.3605638): accepted by ICPP'23
- [ ] [Flashpoint: A Low-latency Serverless Platform for Deep Learning Inference Serving](https://uwspace.uwaterloo.ca/handle/10012/19748)
- [ ] [Serving deep learning models in a serverless platform](https://arxiv.org/abs/1710.08460)
- [ ] [BATCH: Machine Learning Inference Serving on Serverless Platforms with Adaptive Batching](https://ieeexplore.ieee.org/document/9355312): accepted by SC'20
- [ ] [MArk: exploiting cloud services for cost-effective, SLO-aware machine learning inference serving](https://dl.acm.org/doi/abs/10.5555/3358807.3358897): accepted by ATC'19
- [ ] ⭐ [MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters](https://www.usenix.org/conference/nsdi22/presentation/weng): challenges and solutions in real-world scenarios, accepted by NSDI'22
- [ ] [SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads](https://arxiv.org/pdf/2312.16733.pdf): under the guidence of Ion Stoica
- [ ] [Learned Best-Effort LLM Serving](https://arxiv.org/abs/2401.07886): a best-effort serving system of UCB

- [ ] [Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences](https://www.usenix.org/conference/osdi22/presentation/han): accepted by OSDI'22, enables microsecond-scale kernel preemption and controlled concurrent execution in GPU scheduling
- [ ] [PipeSwitch: fast pipelined context switching for deep learning applications](https://dl.acm.org/doi/10.5555/3488766.3488794): PipeSwitch, a system that enables unused cycles of an inference application to be filled by training or other inference applications, accepted by OSDI'20

- [ ] ⭐ [Paella: Low-latency Model Serving with Software-defined GPU Scheduling](https://dl.acm.org/doi/abs/10.1145/3600006.3613163): how the tasks are scheduled to GPUs, accepted by SOSP'23
- [ ] [OTAS: An Elastic Transformer Serving System via Token Adaptation](https://arxiv.org/abs/2401.05031): elastic in serving while considering SLO
- [ ] [DeltaZip: Multi-Tenant Language Model Serving via Delta Compression](https://arxiv.org/abs/2312.05215): Multi-Tenant is interesting
- [x] [ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](https://arxiv.org/abs/2401.14351): find different problems in serving LLMs
- [ ] [Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access](https://dl.acm.org/doi/10.1145/3552326.3567508): accepted by EuroSys'23
- [ ] [Towards Pareto Optimal Throughput in Small Language Model Serving](https://arxiv.org/abs/2404.03353): Small Language Model Serving
- [ ] [MOPAR: A Model Partitioning Framework for Deep Learning Inference Services on Serverless Platforms](https://arxiv.org/abs/2404.02445)
- [ ] [Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services](https://arxiv.org/abs/2404.16283): idea of QoE

- [ ] [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789): how to find novel questions?
- [ ] [Deferred Continuous Batching in Resource-Efficient Large Language Model Serving](https://dl.acm.org/doi/abs/10.1145/3642970.3655835): similar to FlexLLM
- [ ] [LLMServingSim: A Simulation Infrastructure for LLM Inference Serving Systems](https://openreview.net/forum?id=LI2IUfI8km): provide some features about LLM serving
- [ ] [Slice-Level Scheduling for High Throughput and Load Balanced LLM Serving](https://arxiv.org/abs/2406.13511): Improvements to ORCA(SLS) and FastServe(ILS)
- [ ] [Offline Energy-Optimal LLM Serving: Workload-Based Energy Models for LLM Inference on Heterogeneous Systems](https://arxiv.org/abs/2407.04014): consider serving efficiency from energy view
- [ ] [Power-aware Deep Learning Model Serving with μ-Serve](https://www.usenix.org/conference/atc24/presentation/qiu): consider energy
- [ ] [Eloquent: A More Robust Transmission Scheme for LLM Token Streaming](https://dl.acm.org/doi/abs/10.1145/3672198.3673797): a new token transmission scheme, useful in chatbot
- [ ] [Responsive ML inference in multi-tenanted environments using AQUA](https://arxiv.org/abs/2407.21255): serving several LLMs based on time-sharing GPUs cycles, in multi-tenanted environments
- [ ] [Towards SLO-Optimized LLM Serving via Automatic Inference Engine Tuning](https://arxiv.org/abs/2408.04323): effect of hyper-parameters in inference engine

- [ ] [Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Scheduling](https://arxiv.org/abs/2408.13510?utm_source=pocket_shared): request schedule
- [ ] [Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/abs/2408.15792): rank request based on output length predict and schedule
- [ ] [Responsive ML inference in multi-tenanted environments using AQUA](https://arxiv.org/abs/2407.21255): offload context to other GPUs in multi-tenant environment
- [ ] [UELLM: A Unified and Efficient Approach for LLM Inference Serving](https://arxiv.org/abs/2409.14961): serving optimization in MaaS clouds
- [ ] [One Queue Is All You Need: Resolving Head-of-Line Blocking in Large Language Model Serving](https://arxiv.org/abs/2407.00047): shcduling the requests
- [ ] [ConServe: Harvesting GPUs for Low-Latency and High-Throughput Large Language Model Serving](https://arxiv.org/html/2410.01228v1): harvest stranded GPU resources for offline LLM inference tasks
- [ ] [LLM-Pilot: Characterize and Optimize Performance of your LLM Inference Services](https://arxiv.org/abs/2410.02425): accepted by SC'24
- [ ] [Revisiting SLO and Goodput Metrics in LLM Serving](https://arxiv.org/abs/2410.14257): check metrics SLO and Goodput in LLM serving
- [ ] [Hops: Fine-grained heterogeneous sensing, efficient and fair Deep Learning cluster scheduling system](https://dl.acm.org/doi/10.1145/3698038.3698515): schedule tasks in multi-tenant deep learning (DL) cluster, accepted by SoCC'24

- [ ] ⭐ [Ensuring Fair LLM Serving Amid Diverse Applications](https://arxiv.org/abs/2411.15997): ensures fair LLM access across diverse applications, with a copilot trace analysis
- [ ] [BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching](https://arxiv.org/abs/2411.16102):  exploits the relaxed latency requirements in offline batch inference to reorder and overlap requests with varied resource demands while ensuring high prefix sharing
- [ ] [BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching](https://arxiv.org/abs/2412.03594#): similar to blendserve
- [ ] [iServe: An Intent-based Serving System for LLMs](https://arxiv.org/abs/2501.13111): use cost model to dynamically set deployment configuration
- [ ] [TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms](https://arxiv.org/abs/2501.02600): seems a Practical work in engineering? Take into account temperature and power consumption
- [ ] [ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments](https://mlsys.org/virtual/2025/poster/3005): a novel scheduling algorithm, which optimizes the deployment plan of LLM serving to accommodate the heterogeneous resource and network bandwidth conditions in cloud environments, and fluctuating online conditions

- [ ] ⭐ [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263): we can learn for the expert-attention disaggregation
- [ ] [SkyServe: Serving AI Models across Regions and Clouds with Spot Instances](https://dl.acm.org/doi/abs/10.1145/3689031.3717459): seems subsequent work on spotserve, serve AI models over a mixture of spot and on-demand replicas, EuroSys'25
- [ ] [Past-Future Scheduler for LLM Serving under SLA Guarantees](https://dl.acm.org/doi/abs/10.1145/3676641.3716011): efficient requests scheduler via considering the historical distribution of request output lengths and calculating memory occupancy at each future time point, and the framework LightLLM
- [ ] [Deferred prefill for throughput maximization in LLM inference](https://dl.acm.org/doi/abs/10.1145/3721146.3721962): looks a bit counter-intuitive
- [ ] [Performance Aware LLM Load Balancer for Mixed Workloads](https://dl.acm.org/doi/abs/10.1145/3721146.3721947):  a heuristic-guided, reinforcement learning-based router with a trainable response-length predictor and a novel formulation for estimating the impact of mixing different workloads
- [ ] [Niyama : Breaking the Silos of LLM Inference Serving](https://arxiv.org/abs/2503.22562): request schedule paper

#### LLM as microservice

- [ ] ⭐ [A System for Microserving of LLMs](https://arxiv.org/abs/2412.12488): seems a idea and industrial practice that makes sense

#### Serverless LLM serving

- [ ] [DeepFlow: Serverless Large Language Model Serving at Scale](https://arxiv.org/abs/2501.14417): provide fine-grained LLM service
- [ ] ⭐ [Towards Swift Serverless LLM Cold Starts with ParaServe](https://arxiv.org/abs/2502.15524): pipeline parallelism and dynamic adjust parallelism strategy, and accelerate cold-start
- [ ] [λScale: Enabling Fast Scaling for Serverless Large Language Model Inference](https://arxiv.org/abs/2502.09922): serverless inference system to achieve fast model scaling, by fast model multicast, inference execution during model transmission and dynamically constructs execution pipelines
- [ ] [Medusa: Accelerating Serverless LLM Inference with Materialization](https://dl.acm.org/doi/10.1145/3669940.3707285): target at cold-start of LLM serverlesss, to solve the available KV cache blocks profiling and cuda graph capture problems, accepted by ASPLOS'25
- [ ] [SMore: Enhancing GPU Utilization in Deep Learning Clusters by Serverless-based Co-location Scheduling](https://ieeexplore.ieee.org/abstract/document/10912752): serverless computing reveals an opportunity to optimize gpu utilization with fine-grained resource allocation
- [ ] [PipeBoost: Resilient Pipelined Architecture for Fast Serverless LLM Scaling](https://arxiv.org/abs/2503.17707): rapidly launch inference services in response to bursty requests without preemptively over-provisioning GPUs

#### Aligning Systems

- [ ] [PUZZLE: Efficiently Aligning Large Language Models through Light-Weight Context Switch](https://www.usenix.org/conference/atc24/presentation/lei)

#### Comm kernels

- [ ] [Enabling Elastic Model Serving with MultiWorld](https://arxiv.org/abs/2407.08980): optimizing collective communication lib for LLM inference
- [ ] [Flexible Scheduling of Network and Computing Resources for Distributed AI Tasks](https://dl.acm.org/doi/10.1145/3672202.3673744)
- [ ] [AdapCC: Making Collective Communication in Distributed Machine Learning Adaptive](https://ieeexplore.ieee.org/document/10631011): communicating strategy based on runtime, ICDCS'24
- [ ] [Crux: GPU-Efficient Communication Scheduling for Deep Learning Training](https://dl.acm.org/doi/10.1145/3651890.3672239): a communication scheduler that aims to maximize GPU computation utilization by mitigating the communication contention among DLT jobs, SIGCOMM'24

#### Dynamic resource

- [ ] [TENPLEX: Changing Resources of Deep Learning Jobs using Parallelizable Tensor Collections](https://arxiv.org/abs/2312.05181): by Luo MAI, similar to SpotServe?
- [x] [SpotServe: Serving Generative Large Language Models on Preemptible Instances](https://arxiv.org/abs/2311.15566): by Xupeng MIAO and under guidence of Zhihao JIA
- [ ] [Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances](https://arxiv.org/abs/2403.14097): by team of SpotServe
- [ ] [FaPES: Enabling Efficient Elastic Scaling for Serverless Machine Learning Platforms](https://dl.acm.org/doi/10.1145/3698038.3698548): a FaaS-oriented Performance-aware Elastic Scaling system to enable efficient resource allocation in serverless platforms for ML jobs, accepted by SoCC'24
- [ ] [Serving Models, Fast and Slow:Optimizing Heterogeneous LLM Inferencing Workloads at Scale](https://arxiv.org/abs/2502.14617): resource allocation at cluster and data center scale

#### Request Scheduling

- [ ] [Compass: A Decentralized Scheduler for Latency-Sensitive ML Workflows](https://arxiv.org/abs/2402.17652): scheduler for latency-sensitive request
- [ ] [Llumnix: Dynamic Scheduling for Large Language Model Serving](https://arxiv.org/abs/2406.03243): scheduling in multi instances may by helpful for me now
- [ ] [Arlo: Serving Transformer-based Language Models with Dynamic Input Lengths](https://henryhxu.github.io/share/xin-icpp24.pdf): solve Dynamic Input Lengths by multi-instance and request scheduling
- [ ] [Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Scheduling](https://arxiv.org/abs/2408.13510): scheduling based on a output length predictor
- [ ] [Is the GPU Half-Empty or Half-Full? Practical Scheduling Techniques for LLMs](https://arxiv.org/abs/2410.17840): request scheduling in cluster and on instance
- [ ] [Fast Inference for Augmented Large Language Models](https://arxiv.org/abs/2410.18248): schedule for Augmented LLM
- [ ] [ALISE: Accelerating Large Language Model Serving with Speculative Scheduling](https://arxiv.org/abs/2410.23537): prediction-based scheduling + memory management + quantization's hodgepodge
- [ ] [The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving](https://arxiv.org/abs/2411.07447): cost model in request scheduling
- [ ] [Queue Management for SLO-Oriented Large Language Model Serving](https://dl.acm.org/doi/10.1145/3698038.3698523): schedule for request with differnt models and differnet SLO requirements
- [ ] [FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving](https://arxiv.org/abs/2411.18424): fairness and request switch
- [ ] [HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](https://arxiv.org/abs/2501.14808): request co-location to maximize serving throughput and prevent starvation, without compromising online serving latency
- [ ] [Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312)
- [ ] [Queueing, Predictions, and LLMs: Challenges and Open Problems](https://arxiv.org/abs/2503.07545): prediction-based queueing and serving

#### Shared Prefix Serving

- [ ] [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition](https://arxiv.org/abs/2402.15220): share prefix and optimize KV Cache

#### Serving for LoRA

- [x] [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285): beginninf of Serving for LoRA, under the guidence of Ion Stoica: accepted by MLSys'24
- [ ] [Dynamic LoRA Serving System for Offline Context Learning](https://people.eecs.berkeley.edu/~kubitron/courses/cs262a-F23/projects/reports/project1011_paper_92116151989678177816.pdf): successor of S-LoRA
- [x] [CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference](https://arxiv.org/abs/2401.11240): serving LoRA is becoming more and more important
- [x] [PUNICA: MULTI-TENANT LORA SERVING](https://arxiv.org/pdf/2310.18547.pdf): accepted by MLSys'24
- [ ] [Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188)
- [ ] [LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design](https://arxiv.org/abs/2405.17741): maybe useful, kernel optimization
- [x] [dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang): accepted by OSDI'24
- [ ] [Enhancing LoRA Model Serving Capacity via Adaptive Operator Scheduling for Multi-Tenancy on GPU](https://ieeexplore.ieee.org/abstract/document/10721583): optimize SGMV kernels
- [ ] [V-LoRA: An Efficient and Flexible System Boosts Vision Applications with LoRA LMM](https://arxiv.org/abs/2411.00915): LoRA for vision models, and optimize LoRA kernels, accepted by EuroSys'25
- [ ] [Efficient Multi-task LLM Quantization and Serving for Multiple LoRA Adapters](https://openreview.net/forum?id=HfpV6u0kbX): facilitates the sharing of a single quantized model for multiple LoRA adapters, accepted by NIPS'24
- [ ] [Comparative Analysis and Optimization of LoRA Adapter Co-serving for Large Language Models](https://dl.acm.org/doi/abs/10.1145/3704440.3704777): more like a survey for LoRA serving
- [ ] [DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs](https://arxiv.org/abs/2312.05215): compress model deltas to serves multiple full-parameter fine-tuned models(maybe not LoRA fine-tune?)

-------------------------------------  
For LoRA but not serving  

- [ ] [ASPEN: High-Throughput LoRA Fine-Tuning of Large Language Models with a Single GPU](https://arxiv.org/abs/2312.02515)
- [ ] [LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin](https://arxiv.org/abs/2312.09979): potential new style of LoRA
- [ ] [Higher Layers Need More LoRA Experts](https://arxiv.org/abs/2402.08562)
- [ ] [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [ ] [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789): how to find novel questions?
- [ ] [LoRA Meets Dropout under a Unified Framework](https://arxiv.org/abs/2403.00812): Analyze LoRA algorithmically
- [ ] [HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning](https://arxiv.org/abs/2404.19245): algorithm optimization for LoRA
- [ ] [SBoRA: Low-Rank Adaptation with Regional Weight Updates](https://arxiv.org/abs/2407.05413): an algorithm optimization for LoRA
- [ ] [A Survey on LoRA of Large Language Models](https://arxiv.org/abs/2407.11046): survey of LoRAs, incluing parallel LoRA computing and Multi-LoRA, [github](https://github.com/ZJU-LLMs/Awesome-LoRAs)
- [ ] [mLoRA: Fine-Tuning LoRA Adapters via Highly-Efficient Pipeline Parallelism in Multiple GPUs](https://arxiv.org/abs/2312.02515): can study the LoRA-aware pipeline parallelism scheme, [github](https://github.com/TUDB-Labs/mLoRA)
- [ ] [MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts](https://arxiv.org/abs/2404.15159): LoRA based MoE, [github](https://github.com/TUDB-Labs/MixLoRA)
- [ ] [GongBu: Easily Fine-tuning LLMs for Domain-specific Adaptation](https://dl.acm.org/doi/abs/10.1145/3627673.3679233): LLM fine-tuning tools
- [ ] [Adapters Selector: Cross-domains and Multi-tasks LoRA Modules Integration Usage Method](https://aclanthology.org/2025.coling-main.40/): select several LoRAs for a content
- [ ] [SplitLLM: Hierarchical Split Learning for Large Language Model over Wireless Network](https://arxiv.org/abs/2501.13318): split learning(?) train lora weights in wireless network environment, store lora in edge servers?
- [ ] [Revolutionizing Large Model Fine-Tuning: The Role of LoRA in Parameter-Efficient Adaptation](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174015835.57150536): a survey, can provide some reference
- [ ] [HyC-LoRA: Memory Efficient LoRA Fine-tuning with \textbf{Hy}brid Activation \textbf{C}ompression](https://mlsys.org/virtual/2025/poster/2975): optimize fine-tune memory overhead by quantization, accepted by MLSYS'25
- [ ] [ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory](https://arxiv.org/abs/2503.12668): fine-tune

#### Combining fine-tuning/training with inference

- [ ] [Deferred Continuous Batching in Resource-Efficient Large Language Model Serving](https://dl.acm.org/doi/abs/10.1145/3642970.3655835)
- [ ] [Latency-Guaranteed Co-Location of Inference and Training for Reducing Data Center Expenses](https://ieeexplore.ieee.org/document/10630927): place training and inference together, control the inference latency to the desired SLO, while maximizing the throughput of the training jobs co-located on the same GPUs, accepted by ICDCS'24

#### Serving Long-Context

Long-Context is a hot point recently.  

- [ ] [Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis](https://arxiv.org/abs/2405.08944)
- [ ] [Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](https://arxiv.org/abs/2407.11550): like a update for H2O or Dejevu, et.al, each attention head have different memory budget
- [ ] [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)
- [ ] [TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection](https://arxiv.org/abs/2411.02886): select some important KV cache to take part in attention computation

#### Complex ML loads

Process differnet ML loads in a cluster.  

- [ ] [PAL: A Variability-Aware Policy for Scheduling ML Workloads in GPU Clusters](https://www.computer.org/csdl/proceedings-article/sc/2024/529100a373/21HUVogFs2s): serve multiple different loads in GPU cluster, accepted by SC'24
- [ ] [PipeLLM: Fast and Confidential Large Language Model Services with Speculative Pipelined Encryption](https://arxiv.org/abs/2411.03357): why Encryption in LLM inference? by IPADS, accepted by ASPLOS'25
- [ ] [Topology-aware Preemptive Scheduling for Co-located LLM Workloads](https://arxiv.org/abs/2411.11560): schedule different workloads

### RAG with LLM

- [ ] ⭐ [Chameleon: a heterogeneous and disaggregated accelerator system for retrieval-augmented language models](https://arxiv.org/abs/2310.09949): retrieval will be helpful, but how to use it?
- [ ] [Generative Dense Retrieval: Memory Can Be a Burden](https://arxiv.org/abs/2401.10487): accepted by EACL'24
- [ ] ⭐ [Accelerating Retrieval-Augmented Language Model Serving with Speculation](https://arxiv.org/abs/2401.14021): also a paper for RaLM
- [ ] [RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation](https://arxiv.org/abs/2404.12457): improve RAG inference with cache, under guidence of Xin JIN
- [ ] [FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research](https://arxiv.org/abs/2405.13576)
- [ ] [Accelerating Retrieval-Augmented Language Model Serving with Speculation](https://arxiv.org/abs/2401.14021): help understand RaLM
- [ ] [NinjaLLM: Fast, Scalable and Cost-effective RAG using Amazon SageMaker and AWS Trainium and Inferentia2](https://arxiv.org/abs/2407.12057)
- [ ] [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/abs/2407.08223): RAG with spec decoding, different draft models with different RAG
- [ ] [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444): optimize KV cache reuse(prefix cache)
- [ ] [RAGServe: Fast Quality-Aware RAG Systems with Configuration Adaptation](https://arxiv.org/abs/2412.10543): trade-off between latency and quantity
- [ ] [Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation](https://arxiv.org/abs/2502.15734): combine RAG with prefix cache
- [ ] [RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving](https://arxiv.org/abs/2503.14649): analyse RAG algorithm then optimize system
- [ ] [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://dl.acm.org/doi/10.1145/3689031.3696098): reuses the precomputed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache, accepted by EuroSys'25

### Combine MoE with LLM inference

Here are two repositories have some papers for MoE: [Papers: MoE/Ensemble](https://huggingface.co/collections/mdouglas/papers-moe-ensemble-653fc75fe8eeea516bf739e1), and [MOE papers to read](https://huggingface.co/collections/davanstrien/moe-papers-to-read-657832cedea7e2122d052a83)  

- [x] ⭐ [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a): accepted by ICML'22
- [x] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin): both training and inference, accepted by ATC'23
- [ ] [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://proceedings.mlsys.org/paper_files/paper/2023/hash/f9f4f0db4894f77240a95bde9df818e0-Abstract-mlsys2023.html): accepted by MLSys'23
- [ ] [Tutel: Adaptive Mixture-of-Experts at Scale](https://proceedings.mlsys.org/paper_files/paper/2023/hash/9412531719be7ccf755c4ff98d0969dc-Abstract-mlsys2023.html): accepted by MLSys'23
- [ ] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066): accepted by ISCA'24
- [ ] [Optimizing Mixture of Experts using Dynamic Recompilations](https://arxiv.org/abs/2205.01848): under guidence of Zhihao JIA
- [ ] [Serving MoE Models on Resource-constrained Edge Devices via Dynamic Expert Swapping](https://arxiv.org/abs/2308.15030): expert swapping is interesting
- [x] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182): some hot optimizations for inference, accepted by NIPS'24
- [ ] [Exploiting Transformer Activation Sparsity with Dynamic Inference](https://arxiv.org/abs/2310.04361)
- [ ] [SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System](https://arxiv.org/abs/2205.10034)
- [ ] [Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production](https://aclanthology.org/2022.sustainlp-1.6/): accepted by ACL'22
- [ ] [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238): combine moe with offloading
- [x] ⭐ [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2401.14361): under guidence of Luo MAI, provided some features and design in moe inference
- [ ] [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033)
- [x] [FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement](https://dl.acm.org/doi/abs/10.1145/3588964): train MoE with new schedule plan, maybe work for inference
- [x] [Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference](https://arxiv.org/abs/2401.08383)
- [ ] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/abs/2308.14352): quantized experts and expers management
- [ ] [Toward Inference-optimal Mixture-of-Expert Large Language Models](https://arxiv.org/abs/2404.02852): some analysis for training moe based on inference cost
- [ ] [Parm: Efficient Training of Large Sparsely-Activated Models with Dedicated Schedules]: comm optimization in MoE, accepted by InfoCom'24
- [ ] [SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html): based on offload, accepted by MLSys'24
- [ ] [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334): introduce some features of MoE, accepted by ICLR'24
- [ ] [Demystifying the Compression of Mixture-of-Experts Through a Unified Framework](https://arxiv.org/abs/2406.02500): introduce some features of MoE too
- [ ] [Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models](https://arxiv.org/abs/2405.14297): introduction paper
- [ ] [Efficient All-to-All Collective Communication Schedules for Direct-Connect Topologies](https://arxiv.org/abs/2309.13541): all_to_all comm, HPDC'24
- [ ] [Scattered Mixture-of-Experts Implementation](https://arxiv.org/abs/2403.08245): ScatterMoE, an implementation of Sparse MoE
- [ ] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/abs/2404.05019): the Shortcut-connection looks more like a algorithm optimization, and provide oppotunity for overlapping
- [ ] [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434): a opsen-source work and it inferences based expert-parallel
- [ ] [SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget](https://arxiv.org/abs/2308.15030): MoE experts offloading, at the cost of reduced accuracy
- [ ] [ProMoE: Fast MoE-based LLM Serving using Proactive Caching](https://arxiv.org/abs/2410.22134): optimization on Pre-gated MoE, by IPADS
- [ ] [Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design](https://arxiv.org/abs/2410.19123): pre-gating router decoupled from the MoE backbone that facilitates system-friendly pre-computing and lookahead scheduling, NIPS'24
- [ ] [MoEsaic: Shared Mixture of Experts](https://dl.acm.org/doi/abs/10.1145/3698038.3698521): share Expert among different MoE instance, "MoE's modular architecture lets users compose their model from popular off-the-shelf experts" is a new scenario
- [ ] [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433): use quantization to decrease uncached MoE load overhead, on edge devices
- [ ] [ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2410.17954): prediction and offload based optimization

- [ ] [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217): use offload-pipeline to accelerate inference moe on single GPU
- [ ] ⭐ [MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems](https://arxiv.org/abs/2412.07067): benchmarking for MoE systems

- [ ] ⭐ [Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection](https://arxiv.org/abs/2411.08982): damn! I had considered this before:( . key insight is that expert importance varies significantly across tokens and inference phases, utilize this to solve the all-activate problem
- [ ] ⭐ [EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference](https://arxiv.org/abs/2410.12247): Gemm implemention optimization and alltoall communication overlap
- [ ] ⭐ [Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling](https://arxiv.org/abs/2410.17043): optimize all2all order, co-locate experts from different models
- [ ] [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643): utilize the expert dependency to opmizate GPU load balance and alltoall latency
- [ ] [fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving](https://arxiv.org/abs/2502.05370): fine-grained expert offload, prefetch and cache
- [ ] ⭐ [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811): fine-grained task schduling and computation-alltoall overlap
- [ ] [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://dl.acm.org/doi/10.1145/3669940.3707267): offload MoE weights to CPU by layers, accepted by ASPLOS'25
- [ ] [eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference](https://arxiv.org/abs/2503.06823): predict to preload experts from cpu, use same expert for subsequent prompts and skip routing for some tasks

#### MoE training

- [ ] [ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling](https://dl.acm.org/doi/10.1145/3627703.3650083): scheduling comp and comm in MoE training, perhaps useful for MoE inference. accepted by EuroSys'24
- [ ] [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906): a start work in MoE
- [ ] [MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services](https://ieeexplore.ieee.org/abstract/document/10528887)
- [ ] [Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models](https://arxiv.org/abs/2405.14297): algorithm change in MoE
- [ ] [Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping](https://proceedings.mlsys.org/paper_files/paper/2024/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html): Computation-Communication Overlapping, accepted by MLSys'24
- [ ] [Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training](https://openreview.net/forum?id=uLpyWQPyF9): training with offload, ICML'24
- [ ] [MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism](https://ieeexplore.ieee.org/document/10494556)
- [ ] [Parm: Efficient Training of Large Sparsely-Activated Models with Dedicated Schedules](https://arxiv.org/abs/2407.00599): Dedicated Schedules for MP+EP+ESP MoE training, maybe work for infernece
- [x] [Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing](https://arxiv.org/pdf/2404.16914): load is stabilized in the middle and late stages of training, but may not wrok greatly for insference

- [ ] [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization](https://www.usenix.org/conference/atc23/presentation/zhai): parallel strategy of MoE, accepted by ATC'23

- [ ] [APTMoE: Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes](https://www.computer.org/csdl/proceedings-article/sc/2024/529100b436/21HUWvO6IIo): fine-tune MoE models with CPU and some algorithm insights, accepted by SC'24
- [ ] [Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing](https://arxiv.org/abs/2404.16914): prediction the expert workload to optimize training
- [ ] [FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/10.1145/3669940.3707272): There isn't much of a novel technology(?), accepted by ASPLOS'25

### Inference with multimodal

- [ ] [MOSEL: Inference Serving Using Dynamic Modality Selection](https://arxiv.org/abs/2310.18481): improving system throughput by 3.6x with an accuracy guarantee and shortening job completion times by 11x
- [ ] [Generative AI Beyond LLMs: System Implications of Multi-Modal Generation](https://arxiv.org/abs/2312.14385): by META
- [ ] [Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations](https://arxiv.org/abs/2304.11267): by Google
- [ ] [Accelerating Text-to-image Editing via Cache-enabled Sparse Diffusion Inference](https://arxiv.org/abs/2305.17423): optimization for diffusion models by cache
- [ ] [DISTMM: Accelerating distributed multimodal model training](https://www.amazon.science/publications/distmm-accelerating-distributed-multimodal-model-training): helpful although it is made for training, accepted by NSDI'24
- [ ] [Addressing Model and Data Heterogeneity in Multimodal Large Language Model Training](https://arxiv.org/abs/2408.04275): distributed MM trainging
- [ ] [Efficiently serving large multimedia models using EPD Disaggregation](https://arxiv.org/abs/2501.05460)
- [ ] [MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving](https://arxiv.org/abs/2502.01960): position-independent caching, with both reuse and recompute, may lead to performance loss
- [ ] [Characterizing and Efficiently Accelerating Multimodal Generation Model Inference](https://arxiv.org/abs/2410.00215): some insights
- [ ] [ModServe: Scalable and Resource-Efficient Large Multimodal Model Serving](https://arxiv.org/abs/2502.00937): provide comprehensive systems analysis of two prominent LMM architectures, decoder-only and cross-attention

#### Training in Multimodal

- [ ] [DistTrain: Addressing Model and Data Heterogeneity with Disaggregated Training for Multimodal Large Language Models](https://arxiv.org/abs/2408.04275): disaggregation in MM training, under guidence of Xin JIN
- [ ] [Efficient Multi-Task Large Model Training via Data Heterogeneity-aware Model Management](https://arxiv.org/abs/2409.03365): efficient MM model training
- [ ] [Spindle: Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling](https://dl.acm.org/doi/abs/10.1145/3676641.3715992): ASPLOS'25

#### Diffusion Models

- [ ] [Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models](https://www.usenix.org/conference/nsdi24/presentation/agarwal-shubham): serving Diffusion models, accepted by NSDI'24
- [ ] [DiffusionPipe: Training Large Diffusion Models with Efficient Pipelines](https://arxiv.org/abs/2405.01248): accepted by MLSys'24
- [ ] [SwiftDiffusion: Efficient Diffusion Model Serving with Add-on Modules](https://arxiv.org/abs/2407.02031): more papers in diffusion models
- [ ] [PATCHEDSERVE: A Patch Management Framework for SLO-Optimized Hybrid Resolution Diffusion Serving](https://arxiv.org/abs/2501.09253): algorithm-based framework

### Compound Inference Systems

What is this? maybe multiple LLM?

- [ ] [Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems](https://arxiv.org/abs/2403.02419): a new scenario, by Stanford
- [ ] [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/abs/2403.04311): also new to me, by Stanford
- [ ] [Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling](https://dl.acm.org/doi/10.1145/3617232.3624849): accuracy scaling is interesting, accepted by ASPLOS'24
- [ ] [MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving](https://arxiv.org/abs/2404.02015): multiple LLMs
- [ ] [ROUTERBENCH: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031): but what is multi-LLM?  
- [ ] [Expert Router: Orchestrating Efficient Language Model Inference through Prompt Classification](https://arxiv.org/abs/2404.15153)
- [ ] [BlockLLM: Multi-tenant Finer-grained Serving for Large Language Models](https://arxiv.org/abs/2404.18322)
- [ ] [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934): prompt KV cache reuse, accepted by MLSys'24
- [ ] [Preble: Efficient Distributed Prompt Scheduling for LLM Serving](https://escholarship.org/uc/item/1bm0k1w0): similar to BlockLLM?
- [ ] [Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution](https://arxiv.org/abs/2406.00059): for LLM-based Applications
- [ ] [Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems](https://arxiv.org/abs/2403.02419)
- [ ] [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665): use multiple LLMs for efficient serving
- [ ] [USHER: Holistic Interference Avoidance for Resource Optimized ML Inference](https://www.usenix.org/conference/osdi24/presentation/shubha): inference several models simultaneously
- [ ] [CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory](https://arxiv.org/abs/2503.02354): a new scenario Collaboration-of-Experts instead of mixture-of-experts, provide some new oppotunities, acceped by ASPLOS'25

### LLM Application

- [ ] [Teola: Towards End-to-End Optimization of LLM-based Applications](https://arxiv.org/abs/2407.00326): endd-to-end optimization
- [ ] [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://arxiv.org/abs/2405.19888): accepted by OSDI'24
- [ ] [Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications](https://dl.acm.org/doi/10.1145/3627703.3629578): many LLM apps share GPU, accepted by EuroSys'24
- [ ] [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657): learn algorithm from it
- [ ] ⭐ [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965): multi-agent has something similar to LLM application, scheduling and preemption
- [ ] [Fast Inference for Augmented Large Language Models](https://arxiv.org/abs/2410.18248): seems a subclass of multi-agent
- [ ] [Towards End-to-End Optimization of LLM-based Applications with Ayo](https://dl.acm.org/doi/abs/10.1145/3676641.3716278):  utilizes task primitives as the basic units and represents each query's workflow as a primitive-level dataflow graph, enables optimizations in parallelization, pipelining across primitives of different modules, and enhances scheduling to improve application-level performance
- [ ] [Improving the End-to-End Efficiency of Offline Inference for Multi-LLM Applications Based on Sampling and Simulation](https://arxiv.org/abs/2503.16893): multi-LLM's end-to-end running

### Fault Tolerance

- [ ] [Characterization of Large Language Model Development in the Datacenter](https://www.usenix.org/conference/nsdi24/presentation/hu): fault-tolerant serving in the future?
- [ ] [Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement](https://arxiv.org/html/2407.04656v1): Fault Tolerance in MoE training
- [ ] [Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training](https://arxiv.org/abs/2408.04307): checkpointing in MoE

### Energy Optimization

It is usually related to CPU-GPU heterogeneity and GPU power consumption.

- [ ] [DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency](https://arxiv.org/abs/2408.00741)
- [ ] [Offline Energy-Optimal LLM Serving: Workload-Based Energy Models for LLM Inference on Heterogeneous Systems](https://arxiv.org/abs/2407.04014)

### Early Exits

- [ ] [Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving](https://arxiv.org/abs/2312.05385): early exits, accepted by SOSP'24
- [ ] [Improving DNN Inference Throughput Using Practical, Per-Input Compute Adaptation](https://gts3.org/assets/papers/2024/iyer:e3.pdf): early exits and some system optimization, accepted by SOSP'24

### RLHF

- [ ] [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143): framework for RLHF
- [ ] [HybridFlow: A Flexible and Efficient RLHF Framework](https://dl.acm.org/doi/10.1145/3689031.3696075): framework for RLHF, accepted by EuroSys'25
- [ ] [RLHFuse: Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion](https://arxiv.org/abs/2409.13221)
- [ ] [Systems Opportunities for LLM Fine-Tuning using Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3721146.3721944): optimization for LLM Fine-Tuning using Reinforcement Learning

### Some Interesting Idea

**Wise men learn by others.**  

- [ ] [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)  
- [ ] [FiDO: Fusion-in-Decoder optimized for stronger performance and faster inference](https://arxiv.org/abs/2212.08153): optimization for retrieval-augmented language model  
- [ ] [Optimizing Dynamic Neural Networks with Brainstorm](https://www.usenix.org/conference/osdi23/presentation/cui): this idea has the potential to go further, accepted by OSDI'23  
- [ ] [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889): Ring Attention?  
- [ ] [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198): by NVIDIA  
- [x] [Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models](https://openreview.net/forum?id=RJpAz15D0S): an interesting performance metric, accepted by NIPS'23
- [ ] [FEC: Efficient Deep Recommendation Model Training with Flexible Embedding Communication](https://dl.acm.org/doi/abs/10.1145/3589310): accepted by SIGMOD'23
- [ ] [Efficient Multi-GPU Graph Processing with Remote Work Stealing](https://ieeexplore.ieee.org/document/10184847): accepted by ICDE'23
- [ ] [ARK: GPU-driven Code Execution for Distributed Deep Learning](https://www.usenix.org/conference/nsdi23/presentation/hwang): accepted by NSDI'23
- [ ] [Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1d781258d409a6efc66cd1aa14a1681c-Abstract.html): accepted by MLSys'22  
- [ ] [Golgi: Performance-Aware, Resource-Efficient Function Scheduling for Serverless Computing](https://dl.acm.org/doi/abs/10.1145/3620678.3624645): Scheduling for Serverless Computing
- [ ] [FastFold: Optimizing AlphaFold Training and Inference on GPU Clusters](https://dl.acm.org/doi/10.1145/3627535.3638465): expand to other ML models instead of LLM
- [ ] [Arrow Matrix Decomposition: A Novel Approach for Communication-Efficient Sparse Matrix Multiplication](https://dl.acm.org/doi/10.1145/3627535.3638496)
- [ ] [FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications with High-Performance Computing](https://arxiv.org/abs/2402.13533)
- [ ] [Two-Face: Combining Collective and One-Sided Communication for Efficient Distributed SpMM](https://dl.acm.org/doi/10.1145/3620665.3640427): efficient SpMM, accepted by ASPLOS'24
- [ ] [GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching](https://dl.acm.org/doi/abs/10.1145/3620665.3640423): GPU memory pool, accepted by ASPLOS'24
- [ ] [QuickLLaMA: Query-aware Inference Acceleration for Large Language Models](https://arxiv.org/abs/2406.07528): an inference-friendly LLaMA architecture
- [ ] [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379): prefix cache for new model arch like combine attention with SSM
- [ ] [Comprehensive Deadlock Prevention for GPU Collective Communication](https://dl.acm.org/doi/10.1145/3689031.3717466): communication library

#### Dataflow

I'd like to create a separate area for data flows. It's just my preference.  

- [ ] ⭐ [FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://dl.acm.org/doi/10.1145/3575693.3575747): dataflow in inference  
- [ ] [Pathways: Asynchronous Distributed Dataflow for ML](https://proceedings.mlsys.org/paper_files/paper/2022/hash/37385144cac01dff38247ab11c119e3c-Abstract.html): accepted by MLSys'22  
- [ ] [VirtualFlow: Decoupling Deep Learning Models from the Underlying Hardware](https://proceedings.mlsys.org/paper_files/paper/2022/hash/7c47b303273905755d3e513ab43ef94f-Abstract.html): accepted by MLSys'22  
- [ ] [NeuStream: Bridging Deep Learning Serving and Stream Processing](https://dl.acm.org/doi/10.1145/3689031.3717489): dataflow in DNN serving, accepted by EuroSys'25

How about data pre-processing overhead in training?

- [ ] [Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation Ordering and Hybrid Placement](https://www.usenix.org/conference/atc24/presentation/graur)

#### GNN

Just my preference.

- [ ] [Boosting Distributed Full-graph GNN Training with Asynchronous One-bit Communication](https://arxiv.org/abs/2303.01277)
- [ ] [GNNPipe: Scaling Deep GNN Training with Pipelined Model Parallelism](https://arxiv.org/abs/2308.10087)
- [ ] [PckGNN: Optimizing Aggregation Operators with Packing Strategies in Graph Neural Networks](https://www.computer.org/csdl/proceedings-article/ipdps/2024/871100a002/1YpzV3puh32): accepted by IPDPS'24
- [ ] [NPA: Improving Large-scale Graph Neural Networks with Non-parametric Attention](https://dl.acm.org/doi/10.1145/3626246.3653399): SIGMOD'24
- [ ] [Eliminating Data Processing Bottlenecks in GNN Training over Large Graphs via Two-level Feature Compression](https://dl.acm.org/doi/10.14778/3681954.3681968): compress node features in graph, accepted by VLDB'24
- [ ] [Mega: More Efficient Graph Attention for GNNs](https://ieeexplore.ieee.org/document/10631005): optimize graph attention efficiency, ICDCS'24
- [ ] [TORCHGT: A Holistic System for Large-Scale Graph Transformer Training](https://www.computer.org/csdl/proceedings-article/sc/2024/529100b210/21HUWlk5iQE): graph transformer model

#### Blockchain

Just my preference, too.

- [ ] [Weaving the Cosmos: WASM-Powered Interchain Communication for AI Enabled Smart Contracts](https://arxiv.org/abs/2502.17604)
