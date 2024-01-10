# LLM-inference-optimization-paper

# Summary of some awesome works for optimizing  LLM inference    

This summary will including three parts: 
1. some **repositories** that you can follow
2. some representative **person** or **labs** that you can follow
3. some important **works** in the different research interests

## Repositories
For example, [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) contains many excellent articles, and is keeping updating (which I believe is the most important for a paperlist).   
[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) is also worth reading.    

Besides, [awesome-AI-system](https://github.com/lambda7xx/awesome-AI-system) works also very well. And you can find other repositories in its content.  

The log ["Large Transformer Model Inference Optimization"](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) helps me a lot at the beginning.  

## Person/Lab

**Follow others' research, and find yourself's idea.**  

It is not my intention to judge the work of these pioneers, and I understand that the shortness of my knowledge will lead me to leave out many important people.   
If you have a different opinion, please feel free to communicate with me through the issue.  
**In no particular order!!**  
**Damn, I can't remember the names of foreigners.**  

[Zhihao JIA](https://www.cs.cmu.edu/~zhihaoj2/): FlexFlow and other imporessive work, important role in MLSys, affiliated with CMU  
[Tianqi CHEN](https://tqchen.com/): TVM, XGBoost, and other imporessive work, important role in Machine Learning System and ML compilers, affiliated with CMU  
[Song HAN](https://hanlab.mit.edu/songhan): many impoertant work in efficient ML including sparsity and quantization. btw, the class [*TinyML and Efficient Deep Learning Computing*](https://efficientml.ai) is highly recommanded, affiliated with MIT     
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

Make useful benchmark or evaluation is helfpul.  

- [ ] [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549): [inference github](https://github.com/mlcommons/inference), a well-known benchmark
- [ ] [llmperf](https://github.com/ray-project/llmperf): evaluate both performance and correctness, but based on ray

### Interesting *NEW* Frameworks in Parallel Decoding 

[Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads](https://sites.google.com/view/medusa-llm)  
prior paper: [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)

[Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/): by lookahead decoding  

Both frameworks use parallel decoding, and deserve a more detailed research.  

#### Papers for Parallel Decoding

There are some interesting papers about parallel decoding.  

- [ ] [Fast Chain-of-Thought: A Glance of Future from Parallel Decoding Leads to Answers Faster](https://arxiv.org/abs/2311.08263)  
- [ ] [Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding](https://arxiv.org/abs/2307.15337)  

### Speculative Decoding

Also named as Speculative Sampling, model collaboration.  

- [x] ⭐ [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318): opening of *Speculative Decoding*, by DeepMind
- [x] ⭐ [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192): work of similar period with the upper one, by Google, accepted by ICML'23
- [x] [SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781): paper under guidance of Zhihao JIA, use Tree decoding and a set of draft models  
- [x] [LLMCad: Fast and Scalable On-device Large Language Model Inference](https://arxiv.org/pdf/2309.04255.pdf): paper under guidance of Xin JIN, speculative decoding for on-device LLM inference based on tree decoding and other optimizations  
- [ ] [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863): similar to speculative decoding, accepted in NIPS'23  
- [ ] [Online Speculative Decoding](https://arxiv.org/abs/2310.07177): update draft model online  
- [ ] [Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding](https://arxiv.org/pdf/2307.05908.pdf): the trade-off analyse deserves a reading
- [ ] [The Synergy of Speculative Decoding and Batching in Serving Large Language Models](https://arxiv.org/abs/2310.18813): analyse for combining the spec decoding with batching  
- [ ] [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252): use retrieval for spec decoding, some familiar names in the authors list  
- [ ] [Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462): by UIUC

### 3D Parallelism 💡

Some knowledege about data parallel, model tensor parallel, and model pipeline parallel will help in this track.  

- [x] ⭐ [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102): use model parallel to accelerating inference, by Google, in MLSys'23    
- [ ] [HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment](https://arxiv.org/abs/2311.11514):  a distributed inference engine that supports asymmetric partitioning of the inference computation

### Prune & Sparsity 💡

An enduring topic in efficient machine learning.  
We mainly focus on Semi-structured and Structured pruning becasue they can accelerate computing.  

- [ ] ⭐ [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378): use N:M sparsity to fully utilize the hardware for accelerating, by Nvidia
- [ ] ⭐ [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://proceedings.mlr.press/v202/liu23am.html): interesting paper in using sparsity, under guidence of Tri DAO and Ce ZHANG, accepted in ICML'23
- [ ] [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers](https://arxiv.org/abs/2305.15805)
- [ ] [Dynamic N:M Fine-Grained Structured Sparse Attention Mechanism](https://dl.acm.org/doi/abs/10.1145/3572848.3577500): accpted by PPoPP'23
- [x] ⭐ [PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation](https://dl.acm.org/doi/10.1145/3600006.3613139): A novel way to deal with dynamic sparsity may be used for GNN and MoE, accepted by SOSP'23

### Quantization 💡

Low-precision for memory and computing efficiency.  

- [ ] [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948)
- [ ] ⭐ [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339): by UW  
- [ ] ⭐ [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438): paper under guidance of Song HAN  
- [ ] ⭐ [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978): paper under guidance of Song HAN  
- [x] [Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](https://arxiv.org/abs/2310.19102): paper under guidance of Tianqi CHEN, quantization is not important, designing how to quantify is important, in review of MLSys'24 
- [ ] [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865): target on inference on a single GPU, equipped quantization  
- [ ] [FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs](https://arxiv.org/abs/2308.09723)
- [ ] [QUIK: Towards End-to-End 4-Bit Inference on Generative Large Language Models](https://arxiv.org/abs/2310.09259)  
- [ ] [Understanding the Impact of Post-Training Quantization on Large Language Models](https://arxiv.org/abs/2309.05210): tech report will help  
- [ ] ⭐ [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836): by HKUST, accepted in EMNLP'23
- [ ] ⭐ [Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization](https://arxiv.org/pdf/2311.16442.pdf): by SJTU, accepted in DAC'24
- [ ] [INT4 Wight + FP8 KV-Cache: optimization for LLM inference](https://zhuanlan.zhihu.com/p/653735572): INT4 Wight + FP8 KV-Cache + Continues batching
- [ ] [KIVI : Plug-and-play 2bit KV Cache Quantization with Streaming Asymmetric Quantization](https://www.researchgate.net/profile/Zirui-Liu-29/publication/376831635_KIVI_Plug-and-play_2bit_KV_Cache_Quantization_with_Streaming_Asymmetric_Quantization/links/658b5d282468df72d3db3280/KIVI-Plug-and-play-2bit-KV-Cache-Quantization-with-Streaming-Asymmetric-Quantization.pdf)

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
- [ ] [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588): under guidence of Ion Stoica

### Computing Optimization

This part include some impressive work optimizing LLM computing by observing the underlying computing properties. Such as FlashAttention, et.al.

#### FlashAttention Family

- [ ] ⭐ [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): one of the most important work these years, both simple and easy to use, by Tri DAO
- [ ] ⭐ [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691): you'd better not ignore it  
- [ ] ⭐ [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html): you'd better not ignore it, too  
- [ ] ⭐ [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285): successor to FlashAttention in inference, accepted by VLDB'24
- [ ] ⭐ [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282): worth reading, FLashDecoding follow-up  

#### Optimization focus on Auto-regressive Decoding

- [x] [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677): splitting prefill and decode in a map-reduce style, by UW and Microsoft

### Memory Manage

This part is inspired by PagedAttention of vLLM. And there are many Top-Conference paper discussing the memory management in DL computing on GPUs.  

- [x] ⭐ [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): memory page management for the KV-Cache in Attention-type model, accepted by SOSP'23 (many papers will cite the vLLM project instead of their paper, which makes it harder for us to find its *citated by*)
- [ ] ⭐ [AutoScratch: ML-Optimized Cache Management for Inference-Oriented GPUs](https://proceedings.mlsys.org/paper_files/paper/2023/hash/627b5f83ffa130fb33cb03dafb47a630-Abstract-mlsys2023.html): cache management for inference, accepted by MLSys'23
- [ ] [Improving Computation and Memory Efficiency for Real-world Transformer Inference on GPUs](https://dl.acm.org/doi/full/10.1145/3617689): block-based data layout, accepted by TACO'October-2023
- [ ] [AttMEMO : Accelerating Transformers with Memoization on Big Memory Systems](https://arxiv.org/abs/2301.09262): a unique observation that there is rich similarity in attention computation across inference sequences
- [ ] [BPIPE: memory-balanced pipeline parallelism for training large language models](https://dl.acm.org/doi/10.5555/3618408.3619090): memory balance perhaps can work well in inferencce, by SNU, accepted by ICML'23
- [ ] [Improving Large Language Model Throughput with Efficient LongTerm Memory Management](https://people.eecs.berkeley.edu/~kubitron/courses/cs262a-F23/projects/reports/project1010_paper_64287652274076362722.pdf): perhaps a new view
- [ ] [CacheGen: Fast Context Loading for Language Model Applications](https://arxiv.org/abs/2310.07240)
- [ ] [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669)

### Inference on hardware: GPUs, CPUs or based on SSD

#### Underlying optimization for GPU

- [ ] [Reducing shared memory footprint to leverage high throughput on Tensor Cores and its flexible API extension library](https://dl.acm.org/doi/abs/10.1145/3578178.3578238): implement some APIs to reduce the shared memory footprint, accepted in HPC Asia'23

#### CPUs or based on SSD

Heterogeneous scenarios or single PC are becoming increasingly important.  

Making optimization for the calculating on CPU or SSD will have different methods.  

- [ ] [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502): LLMs with quantization on CPUs, by Intel, accepted by NIPS'23  
- [ ] [Improving Throughput-oriented Generative Inference with CPUs](https://dl.acm.org/doi/abs/10.1145/3609510.3609815): cooperate of CPUs and GPU, accepted by APSys'23  
- [ ] [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865): inference a 30B model with a 16GB GPU, accepted by ICML'23
- [ ] [Chrion: Optimizing Recurrent Neural Network Inference by Collaboratively Utilizing CPUs and GPUs](https://arxiv.org/abs/2307.11339): execute the operators on the CPU and GPU in parallel, by SJTU
- [ ] [EdgeNN: Efficient Neural Network Inference for CPU-GPU Integrated Edge Devices](https://ieeexplore.ieee.org/document/10184528): inference on edge devices, accepted by ICDE'23
- [ ] [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456): by SJTU IPADS
- [ ] [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514): by Apple

#### Heterogeneous or decentralized environments

- [ ] [FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs](https://arxiv.org/abs/2309.01172): decentrailized system on consumer-level GPUs, through there will be some problems
- [ ] [Distributed Inference and Fine-tuning of Large Language Models Over The Internet](https://arxiv.org/abs/2312.08361): some techs in this paper will be instructive

### Algorithm Optimization 💡

In this part, researchers provide some algorithm-based method to optimizing LLM inference.  

- [x] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048): accepted by NIPS'23
- [ ] [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time](https://arxiv.org/abs/2305.17118): consider the different importance of tokens in KV Cache, similar to H2O
- [ ] ⭐ [SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference](https://arxiv.org/abs/2307.02628): skipping maybe an useful method like spec decoding
- [ ] [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487): also a potential optimization
- [ ] [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453): streaming LLM for infinite sequence lengths, by MIT and under guidence of Song HAN

### Industrial Inference Frameworks 💡

- [ ] ⭐ [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032): you must know DeepSpeed  
- [ ] [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
- [ ] [DeepSpeed Model Implementations for Inference (MII)](https://github.com/microsoft/DeepSpeed-MII)
- [x] [ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs](https://arxiv.org/abs/2210.03052): developed by ByteDance, accepted by IPDPS'23
- [ ] [TurboTransformers: an efficient GPU serving system for transformer models](https://dl.acm.org/doi/10.1145/3437801.3441578): by Tencent Inc, accepted by PPoPP'21  
- [ ] [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/?utm_content=273712248&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024): a blog in PyTorch, use only PyTorch code, [gpt-fast](https://github.com/pytorch-labs/gpt-fast)
- [ ] [FlexFlow Serve: Low-Latency, High-Performance LLM Serving](https://github.com/flexflow/FlexFlow): based on FlexFlow
- [ ] [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)

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
- [ ] [SpotServe: Serving Generative Large Language Models on Preemptible Instances](https://arxiv.org/abs/2311.15566): by Xupeng MIAO and under guidence of Zhihao JIA
- [ ] [BATCH: Machine Learning Inference Serving on Serverless Platforms with Adaptive Batching](https://ieeexplore.ieee.org/document/9355312): accepted by SC'20
- [ ] [MArk: exploiting cloud services for cost-effective, SLO-aware machine learning inference serving](https://dl.acm.org/doi/abs/10.5555/3358807.3358897): accepted by ATC'19
- [ ] ⭐ [MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters](https://www.usenix.org/conference/nsdi22/presentation/weng): challenges and solutions in real-world scenarios, accepted by NSDI'22
- [ ] [SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads](https://arxiv.org/pdf/2312.16733.pdf): under the guidence of Ion Stoica

- [ ] [Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences](https://www.usenix.org/conference/osdi22/presentation/han): accepted by OSDI'22, enables microsecond-scale kernel preemption and controlled concurrent execution in GPU scheduling
- [ ] [PipeSwitch: fast pipelined context switching for deep learning applications](https://dl.acm.org/doi/10.5555/3488766.3488794): PipeSwitch, a system that enables unused cycles of an inference application to be filled by training or other inference applications, accepted by OSDI'20

- [ ] ⭐ [Paella: Low-latency Model Serving with Software-defined GPU Scheduling](https://dl.acm.org/doi/abs/10.1145/3600006.3613163): how the tasks are scheduled to GPUs, accepted by SOSP'23

### RAG with LLM
 
- [ ] [Chameleon: a heterogeneous and disaggregated accelerator system for retrieval-augmented language models](https://arxiv.org/abs/2310.09949): retrieval will be helpful, but how to use it?

### Combine MoE with LLM inference
Here are two repositories have some papers for MoE: [Papers: MoE/Ensemble](https://huggingface.co/collections/mdouglas/papers-moe-ensemble-653fc75fe8eeea516bf739e1), and [MOE papers to read](https://huggingface.co/collections/davanstrien/moe-papers-to-read-657832cedea7e2122d052a83)  

- [x] ⭐ [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a): accepted by ICML'22
- [ ] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin): both training and inference, accepted by ATC'23
- [ ] [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://proceedings.mlsys.org/paper_files/paper/2023/hash/f9f4f0db4894f77240a95bde9df818e0-Abstract-mlsys2023.html): accepted by MLSys'23
- [ ] [Tutel: Adaptive Mixture-of-Experts at Scale](https://proceedings.mlsys.org/paper_files/paper/2023/hash/9412531719be7ccf755c4ff98d0969dc-Abstract-mlsys2023.html): accepted by MLSys'23
- [ ] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066)
- [ ] [Optimizing Mixture of Experts using Dynamic Recompilations](https://arxiv.org/abs/2205.01848): under guidence of Zhihao JIA
- [ ] [Serving MoE Models on Resource-constrained Edge Devices via Dynamic Expert Swapping](https://arxiv.org/abs/2308.15030): expert swapping is interesting
- [ ] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182): some hot optimizations for inference
- [ ] [Exploiting Transformer Activation Sparsity with Dynamic Inference](https://arxiv.org/abs/2310.04361)
- [ ] [SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System](https://arxiv.org/abs/2205.10034)
- [ ] [Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production](https://aclanthology.org/2022.sustainlp-1.6/): accepted by ACL'22
- [ ] [Fast Inference of Mixture-of-Experts LanguageModels with Offloading](https://arxiv.org/abs/2312.17238): combine moe with offloading

### Inference with multimodal

- [ ] [MOSEL: Inference Serving Using Dynamic Modality Selection](https://arxiv.org/abs/2310.18481): improving system throughput by 3.6× with an accuracy guarantee and shortening job completion times by 11×
- [ ] [Generative AI Beyond LLMs: System Implications of Multi-Modal Generation](https://arxiv.org/abs/2312.14385): by META
- [ ] [Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations](https://arxiv.org/abs/2304.11267): by Google
- [ ] [Accelerating Text-to-image Editing via Cache-enabled Sparse Diffusion Inference](https://arxiv.org/abs/2305.17423): optimization for diffusion models by cache

### Some Interesting Idea

**Wise men learn by others.**  

- [ ] [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)  
- [ ] [FiDO: Fusion-in-Decoder optimized for stronger performance and faster inference](https://arxiv.org/abs/2212.08153): optimization for retrieval-augmented language model  
- [ ] [Optimizing Dynamic Neural Networks with Brainstorm](https://www.usenix.org/conference/osdi23/presentation/cui): this idea has the potential to go further, accepted by OSDI'23  
- [ ] [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889): Ring Attention?  
- [ ] [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198): by NVIDIA  
- [ ] [Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models](https://openreview.net/forum?id=RJpAz15D0S): an interesting performance metric, accepted by NIPS'23
- [ ] [FEC: Efficient Deep Recommendation Model Training with Flexible Embedding Communication](https://dl.acm.org/doi/abs/10.1145/3589310): accpted by SIGMOD'23
- [ ] [Efficient Multi-GPU Graph Processing with Remote Work Stealing](https://ieeexplore.ieee.org/document/10184847): accpted by ICDE'23
- [ ] [ARK: GPU-driven Code Execution for Distributed Deep Learning](https://www.usenix.org/conference/nsdi23/presentation/hwang): accpted by NSDI'23
- [ ] [Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1d781258d409a6efc66cd1aa14a1681c-Abstract.html): accepted by MLSys'22  

#### Dataflow

I'd like to create a separate area for data flows. It's just my preference.  

- [ ] ⭐ [FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://dl.acm.org/doi/10.1145/3575693.3575747): dataflow in inference  
- [ ] [Pathways: Asynchronous Distributed Dataflow for ML](https://proceedings.mlsys.org/paper_files/paper/2022/hash/37385144cac01dff38247ab11c119e3c-Abstract.html): accepted by MLSys'22  
- [ ] [VirtualFlow: Decoupling Deep Learning Models from the Underlying Hardware](https://proceedings.mlsys.org/paper_files/paper/2022/hash/7c47b303273905755d3e513ab43ef94f-Abstract.html): accepted by MLSys'22  