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
- [ ] [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2311.09476.pdf)  
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

[Zhihao JIA](https://www.cs.cmu.edu/~zhihaoj2/): FlexFlow and other imporessive work, important role in MLSys, affiliated with CMU  
[Tianqi CHEN](https://tqchen.com/): TVM, XGBoost, and other imporessive work, important role in Machine Learning System and ML compilers, affiliated with CMU  
[Song HAN](https://hanlab.mit.edu/songhan): many impoertant work in efficient ML including sparsity and quantization. btw, the class [*TinyML and Efficient Deep Learning Computing*](https://efficientml.ai) is highly recommanded, affiliated with MIT     
[Zhen DONG](https://dong-zhen.com/): many important work in quantization and high-performance ML, affiliated with UCB  
[Tri DAO](https://tridao.me/): author of FlashAttention, affiliated with Princeton  

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

## Work 

I hope to conlude these impressive works based on their research direction.  
But my summary must not be informative enough, and I am looking forward to your addition.  

**Perhaps someone should write a detailed survey.**  

**Periodically check the "cited by" of the papers with ⭐ will be helpful.**  

### Parallelism

Some knowledege about data parallel, model tensor parallel, and model pipeline parallel will help in this track.  

- [x] ⭐ [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102): use model parallel to accelerating inference, by Google, in MLSys'23    

### Speculative Decoding

Also named as Speculative Sampling, model collaboration.  

- [x] ⭐ [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318): opening of *Speculative Decoding*, by DeepMind
- [x] ⭐ [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192): work of similar period with the upper one, by Google, accepted by ICML'23

### Prune & Sparsity

An enduring topic in efficient machine learning.  
We mainly focus on Semi-structured and Structured pruning becasue they can accelerate computing.  

- [ ] ⭐ [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378): use N:M sparsity to fully utilize the hardware for accelerating, by Nvidia

### Quantization

Low-precision for memory and computing efficiency.  
- [ ] ⭐ [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339): by UW  
- [ ] ⭐ [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438): paper under guidance of Song HAN  
- [ ] ⭐ [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978): paper under guidance of Song HAN  

### Batch Processing

Perhaps the most important way for improving the throughput in LLM inference.  
This blog [Dissecting Batching Effects in GPT Inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) helps me a lot at the beginning.  

- [x] ⭐ [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu): dynamil batch processing without redundant computing, accepted in OSDI'23
- [x] [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920): considering Job Completion Time(JCT) in LLM serving, paper under guidance of Xin JIN  

### Computing Optimization

This part include some impressive work optimizing LLM computing by observing the underlying computing properties. Such as FlashAttention, et.al.

- [ ] ⭐ [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): one of the most important work these years, both simple and easy to use, by Tri DAO
FlashAttentionV2, FLashDecoding, et.al
- [ ] [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282): worth reading, FLashDecoding follow-up  

### Memory Manage

This part is inspired by PagedAttention of vLLM. And there are many Top-Conference paper discussing the memory management in DL computing on GPUs.  

- [ ] ⭐ [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): memory page management for the KV-Cache in Attention-type model, accepted by SOSP'23

### Inference on CPUs or based on SSD

Making optimization for the calculating on CPU or SSD will have different methods.  

- [ ] [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865): inference a 30B model with a 16GB GPU, accepted by ICML'23

### Algorithm Optimization

In this part, researchers provide some algorithm-based method to optimizing LLM inference.  

- [x] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048): accepted by NIPS'23

### LLM Serving

LLM server providers will focus on this part.  

- [ ] ⭐ [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665): accepted by OSDI'23