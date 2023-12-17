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
Map-Reduce arch in LLM inference, these evaluations will bring us some throughts

2023/12/14--2023/12/20
- [x] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a): provide hybrid MoE parallel framework, communication and kernel optimization