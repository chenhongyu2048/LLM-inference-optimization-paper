# LLM-inference-optimization-paper

# Myself's reading list

2023/11/15--2023/11/21:  
- [x] [Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](https://arxiv.org/abs/2310.19102): paper under guidance of Tianqi CHEN, in review of MLSys'24  
Quantization is not important, what make it sense is how to quantify  
- [x] [LLMCad: Fast and Scalable On-device Large Language Model Inference](https://arxiv.org/pdf/2309.04255.pdf): paper under guidance of Xin JIN  
Better Speculative Decoding, with novel experimental scenes  
- [x] [Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning](https://www.usenix.org/conference/osdi23/presentation/zhang-chen): paper under guidance of Jidong ZHAI, accepted in OSDI'23  
Fine-grained uTask abstraction, control flow on GPU side, with more expansive compile optimization

2023/11/22--2023/11/28:
- [x] [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2311.09476.pdf): arxiv, by Stanford    
Automated Evaluation Framework trains a LLM-based judge with self-generated datapoints to check RAG systems, bolstering with rediction-powered inference (PPI)   

2023/11/30--2023/12/06:
- [ ] [EINNET: Optimizing Tensor Programs with Derivation-Based Transformations](https://www.usenix.org/conference/osdi23/presentation/zheng): under guidence of Zhihao JIA & Jidong ZHAI, accepted in OSDI'23  
- [ ] [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [ ] [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/abs/10.1145/3567955.3567959): Google, accepted in ASPLOS'23

# Summary of some awesome works for optimizing  LLM inference    

This summary will including three parts: 
1. some **repositories** that you can follow
2. some representative **person** or **labs** that you can follow
3. some important **works** in the different research interests

## Repositories
For example, [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) contains many excellent articles, and is keeping updating (which I believe is the most important for a paperlist).  

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

[SPCL](https://spcl.inf.ethz.ch/Publications/): Scalable Parallel Computing Lab, affiliated with ETHz  
[Luo MAI](https://luomai.github.io/): affiliated with University of Edinburgh

[IPADS](https://ipads.se.sjtu.edu.cn/zh/publications/): focus more on PURE systems, buut also make great progress in MLSys, affiliated with SJTU  
[EPCC](http://epcc.sjtu.edu.cn/): Emerging Parallel Computing Center, parallel computing and MLSys are Naturally combined, affiliated with SJTU

[Xin JIN](https://xinjin.github.io/): FastServe and LLMCad are impressive work, affiliated with PKU  
[Bin CUI](https://cuibinpku.github.io/): important role in MLSys including DL, GNN, and MoE, affiliated with PKU  
[Jidong ZHAI](https://pacman.cs.tsinghua.edu.cn/~zjd/): leading many important work in MLSys, affiliated with THU  
[Lingxiao MA](https://xysmlx.github.io/): with many important work in MLSys on Top-Conference, affiliated with MSRA  
[Cheng LI](http://staff.ustc.edu.cn/~chengli7/): high performce system and MLSys, affiliated with USTC  

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

### Survey/Evaluations 💡

- [ ] ⭐ [Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models](https://arxiv.org/abs/2311.03687): evaluations helps you find the bottleneck  
- [ ] ⭐ [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017): a survey by UCB  

### Interesting *NEW* Frameworks in Parallel Decoding 

[Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads](https://sites.google.com/view/medusa-llm)  
prior paper: [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)

[Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/): by lookahead decoding  

Both frameworks use parallel decoding, and deserve a more detailed research.  

#### Papers for Parallel Decoding

There are some interesting papers about parallel decoding.  

- [ ] [Fast Chain-of-Thought: A Glance of Future from Parallel Decoding Leads to Answers Faster](https://arxiv.org/abs/2311.08263)  
- [ ] [Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding](https://arxiv.org/abs/2307.15337)  

### 3D Parallelism 💡

Some knowledege about data parallel, model tensor parallel, and model pipeline parallel will help in this track.  

- [x] ⭐ [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102): use model parallel to accelerating inference, by Google, in MLSys'23    
- [ ] [HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment](https://arxiv.org/abs/2311.11514):  a distributed inference engine that supports asymmetric partitioning of the inference computation

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

### Prune & Sparsity 💡

An enduring topic in efficient machine learning.  
We mainly focus on Semi-structured and Structured pruning becasue they can accelerate computing.  

- [ ] ⭐ [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378): use N:M sparsity to fully utilize the hardware for accelerating, by Nvidia
- [ ] ⭐ [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://proceedings.mlr.press/v202/liu23am.html): interesting paper in using sparsity, under guidence of Tri DAO and Ce ZHANG, accepted in ICML'23
- [ ] [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers](https://arxiv.org/abs/2305.15805)
- [ ] [Dynamic N:M Fine-Grained Structured Sparse Attention Mechanism](https://dl.acm.org/doi/abs/10.1145/3572848.3577500): accpted by PPoPP'23

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

### Batch Processing

Perhaps the most important way for improving the throughput in LLM inference.  
This blog [Dissecting Batching Effects in GPT Inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) helps me a lot at the beginning.  

- [x] ⭐ [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu): dynamil batch processing without redundant computing, accepted in OSDI'23  
- [x] [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920): considering Job Completion Time(JCT) in LLM serving, paper under guidance of Xin JIN  
- [ ] [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144): schedule based on response length prediction by LLM, paper under guidance of Yang YOU  
- [ ] [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput](https://arxiv.org/abs/2306.06000): idea similar to above, by Harvard University  
- [ ] [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369): blocking the prefill phase and reduce pipeline bubbles, by MSRIndia  
- [ ] [Flover: A Temporal Fusion Framework for Efficient Autoregressive Model Parallel Inference](https://arxiv.org/abs/2305.13484): accepted by HiPC'23  
- [ ] [Handling heavy-tailed input of transformer inference on GPUs](https://dl.acm.org/doi/10.1145/3524059.3532372): accepted by ICS'22  
- [ ] [CoFB: latency-constrained co-scheduling of flows and batches for deep learning inference service on the CPU–GPU system](https://link.springer.com/article/10.1007/s11227-023-05183-6): Some form of inference service  
- [ ] [TCB: Accelerating Transformer Inference Services with Request Concatenation](https://dl.acm.org/doi/10.1145/3545008.3545052): perhaps similar to ByteTransformer, accepted by ICPP'22  

### Computing Optimization

This part include some impressive work optimizing LLM computing by observing the underlying computing properties. Such as FlashAttention, et.al.

#### FlashAttention Family

- [ ] ⭐ [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): one of the most important work these years, both simple and easy to use, by Tri DAO
- [ ] ⭐ [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691): you'd better not ignore it  
- [ ] ⭐ [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html): you'd better not ignore it, too  
- [ ] ⭐ [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285): successor to FlashAttention in inference, accepted by VLDB'24
- [ ] ⭐ [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282): worth reading, FLashDecoding follow-up  

### Memory Manage

This part is inspired by PagedAttention of vLLM. And there are many Top-Conference paper discussing the memory management in DL computing on GPUs.  

- [ ] ⭐ [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): memory page management for the KV-Cache in Attention-type model, accepted by SOSP'23 (many papers will cite the vLLM project instead of their paper, which makes it harder for us to find its *citated by*)
- [ ] ⭐ [AutoScratch: ML-Optimized Cache Management for Inference-Oriented GPUs](https://proceedings.mlsys.org/paper_files/paper/2023/hash/627b5f83ffa130fb33cb03dafb47a630-Abstract-mlsys2023.html): cache management for Inference, accepted by MLSys'23
- [ ] [Improving Computation and Memory Efficiency for Real-world Transformer Inference on GPUs](https://dl.acm.org/doi/full/10.1145/3617689): block-based data layout, accepted by TACO'October
- [ ] [AttMEMO : Accelerating Transformers with Memoization on Big Memory Systems](https://arxiv.org/abs/2301.09262): a unique observation that there is rich similarity in attention computation across inference sequences
- [ ] [BPIPE: memory-balanced pipeline parallelism for training large language models](https://dl.acm.org/doi/10.5555/3618408.3619090): memory balance perhaps can work well in inferencce, by SNU, accepted by ICML'23

### Inference on CPUs or based on SSD

Making optimization for the calculating on CPU or SSD will have different methods.  

- [ ] [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502): LLMs with quantization on CPUs, by Intel, accepted by NIPS'23  
- [ ] [Improving Throughput-oriented Generative Inference with CPUs](https://dl.acm.org/doi/abs/10.1145/3609510.3609815): cooperate of CPUs and GPU, accepted by APSys'23  
- [ ] [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865): inference a 30B model with a 16GB GPU, accepted by ICML'23
- [ ] [Chrion: Optimizing Recurrent Neural Network Inference by Collaboratively Utilizing CPUs and GPUs](https://arxiv.org/abs/2307.11339): execute the operators on the CPU and GPU in parallel, by SJTU

### Algorithm Optimization 💡

In this part, researchers provide some algorithm-based method to optimizing LLM inference.  

- [x] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048): accepted by NIPS'23
- [ ] ⭐ [SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference](https://arxiv.org/abs/2307.02628): skipping maybe an useful method like spec decoding
- [ ] [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487): also a potential optimization

### Industrial Inference Frameworks 💡

- [ ] ⭐ [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032): you must know DeepSpeed  
- [ ] [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
- [ ] [DeepSpeed Model Implementations for Inference (MII)](https://github.com/microsoft/DeepSpeed-MII)
- [x] [ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs](https://arxiv.org/abs/2210.03052): developed by ByteDance, accepted by IPDPS'23
- [ ] [TurboTransformers: an efficient GPU serving system for transformer models](https://dl.acm.org/doi/10.1145/3437801.3441578): by Tencent Inc, accepted by PPoPP'21  

### LLM Serving 💡

LLM server providers will focus on this part.  

- [ ] ⭐ [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665): accepted by OSDI'23
- [ ] ⭐ [STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining](https://arxiv.org/abs/2207.05022): Elastic will be important in the future, accepted by ASPLOS'23  
- [ ] [INFaaS: Automated Model-less Inference Serving](https://www.usenix.org/conference/atc21/presentation/romero): accepted by ATC'21  
- [ ] [Tabi: An Efficient Multi-Level Inference System for Large Language Models](https://dl.acm.org/doi/abs/10.1145/3552326.3587438): under guidence of Kai CHEN, accepted by EuroSys'23  
- [ ] [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176): cost is the service provider cares most  
- [ ] [FaaSwap: SLO-Aware, GPU-Efficient Serverless Inference via Model Swapping](https://arxiv.org/abs/2306.03622)

### Some Interesting Idea

**Wise men learn by others.**  

- [ ] [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)  
- [ ] [FiDO: Fusion-in-Decoder optimized for stronger performance and faster inference](https://arxiv.org/abs/2212.08153): optimization for retrieval-augmented language model  
- [ ] [Optimizing Dynamic Neural Networks with Brainstorm](https://www.usenix.org/conference/osdi23/presentation/cui): this idea has the potential to go further, accepted by OSDI'23  
- [ ] [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889): Ring Attention?  
- [ ] [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198): by NVIDIA  
- [ ] ⭐ [FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://dl.acm.org/doi/10.1145/3575693.3575747): dataflow in inference  
- [ ] [Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models](https://openreview.net/forum?id=RJpAz15D0S): an interesting performance metric, accepted by NIPS'23