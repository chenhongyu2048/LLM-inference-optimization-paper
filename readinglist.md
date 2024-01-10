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
- [x] [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369): bu MSRIndia  
Interesting idea of piggybacking, accurately found the issue of too many requests in decode phase in ORCA and the related performance bottleneck  
- [x] [Improving GPU Throughput through ParallelExecution Using Tensor Cores and CUDA Cores](https://ieeexplore.ieee.org/document/9912002): accepted by ISVLSI'22
Parallel execution of GPU CUDA cores and Tensor cores, but this paper only use simulation and the achieved performance is limited  
- [x] [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu): accepted in OSDI'23  
Continues batch processing without redundant computing
- [ ] [EINNET: Optimizing Tensor Programs with Derivation-Based Transformations](https://www.usenix.org/conference/osdi23/presentation/zheng): under guidence of Zhihao JIA & Jidong ZHAI, accepted in OSDI'23  
*Temporarily suspended*, I just read the intro. Maybe I need some prior knowledge such as TVM, Pet, et.al.  

2023/12/07--2023/12/13:  
- [x] [gSampler: General and Efficient GPU-based Graph Sampling for Graph Learning](https://dl.acm.org/doi/10.1145/3600006.3613168): accepted by SOSP'23  
With very detailed description of sampling algorithm, optimization for graph sampling using GPUs, and proposes a paradigm
- [x] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): memory page management for the KV-Cache in Attention-type model, accepted by SOSP'23
vLLM is famous for its PagedAttention memory management
- [x] [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677): splitting prefill and decode in a map-reduce style, by UW and Microsoft
Map-Reduce arch in LLM inference, these evaluations will bring us some throughts. perhaps only useful in great clusters  

2023/12/14--2023/12/20
- [x] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a): provide hybrid MoE parallel framework, communication and kernel optimization

2023/12/28--2024/01/03
- [x] [HongTu: Scalable Full-Graph GNN Training on Multiple GPUs (via communication-optimized CPU data offloading)](https://arxiv.org/abs/2311.14898): full-graph GNN training on single-machine-multi-gpu, by group of Bingsheng HE  
combines the partition-based GNN training and *recomputation*-cache-hybrid intermediate data management, and deduplicated communication to utilize the inter-GPU and intra-GPU(local) data access  
- [x] [Generative AI Beyond LLMs: System Implications of Multi-Modal Generation](https://arxiv.org/abs/2312.14385): A quantitative analysis of the inference system performance of multimodal models and the observation of some interesting bottlenecks. The idea of first building an evaluation framework is good  
- [ ] [Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models](https://arxiv.org/abs/2311.03687): evaluations helps you find the bottleneck  
*Temporarily suspended*, I just read the evaluation of inference and the microbenchmark parts. The Module-wise Analysis part in inference is also incomplete

2024/01/04--2024/01/10 
- [x] ⭐ [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234)  
provide a detailed survey of decoder-only LLM inference, and with a similar categorization to our repo :\). I also update some papers based on this survey  
the authors list a number of optimization methods, and based on some methods they have made their own excellent work. This reminds us of the importance of setting a novelty goal and thinking creatively  
- [x] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182): some hot optimizations for inference  
For a scenario of MoE (encoder-decoder model and a large number of experts), some original and effective optimization strategies are provided: Dynamic Gating, Expert Buffering, and Expert Load Balancing  
- [x] ⭐ [PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation](https://dl.acm.org/doi/10.1145/3600006.3613139): A novel way to deal with dynamic sparsity may be used for GNN and MoE, accepted by SOSP'23
Novel observations and solid engineering implementations and experiments, the disadvantage is that the application requirements are relatively high (more than 90% sparsity)

2024/01/11--2024/01/17
- [ ] [Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models](https://openreview.net/forum?id=RJpAz15D0S): an interesting performance metric, accepted by NIPS'23
- [ ] ⭐ [MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters](https://www.usenix.org/conference/nsdi22/presentation/weng): challenges and solutions in real-world scenarios
- [ ] [SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads](https://arxiv.org/pdf/2312.16733.pdf): under the guidence of Ion Stoica
- [ ] [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669)