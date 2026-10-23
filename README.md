# Data-Primitives: Data Science & Data Engineering Architectures

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Build-Passing-success.svg?style=for-the-badge" alt="Build Status">
</p>

---

![Abstract Network and Data Connections](https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## 📌 Executive Overview

**Data-Primitives** is a unified, enterprise-grade infrastructure and algorithmic suite engineered by **Rayaan Tasnim**. It bridges high-level probabilistic AI research with high-performance, bare-metal stream and tabular data engineering tools. 

This repository operates as a **Dual-Project Infrastructure Suite**:
1. **Data Science Suite**: Deep probabilistic inference engines, self-supervised representation transformers, agentic decision frameworks, and physics-informed neural networks.
2. **Data Engineering Suite**: Zero-copy columnar processing frameworks, low-latency binary log emulation, out-of-core memory-mapped relational operations, and adaptive backpressure-driven ingestion engines.

---

## 🏗 High-Level System Architecture

```
                                  [ RAW DATA SOURCE AGGREGATION ]
                                                 │
            ┌────────────────────────────────────┴────────────────────────────────────┐
            ▼                                                                         ▼
┌─────────────────────────────────────────┐               ┌─────────────────────────────────────────┐
│     DATA-ENGINEERING STREAM CORE        │               │       DATA-SCIENCE INFERENCE CORE       │
├─────────────────────────────────────────┤               ├─────────────────────────────────────────┤
│ • Zero-Copy Parquet Reader/Writer       │               │ • Variational Inference BNNs            │
│ • Memory-Mapped Out-of-Core Joins       │ ───────┬─────>│ • Transformer Decoders w/ KV-Cache      │
│ • Vectorized Byte-Array Log Parsers     │        │      │ • Contrastive SimCLR Self-Supervised    │
│ • Async Backpressure Ring Buffers       │        │      │ • Physics-Informed Loss Functions (PINN)│
│ • CDC via Merkle-Tree State Replication │        │      │ • Self-Rerouting Agentic RAG            │
└─────────────────────────────────────────┘        │      └─────────────────────────────────────────┘
                                                   │
                                                   ▼
                                 ┌───────────────────────────────────┐
                                 │     FEATURE STORE & MODEL HUB     │
                                 └───────────────────────────────────┘
```

---

## 📁 Repository Structure & Deep Technical Profiles

```
Data-Primitives/
├── data_science/
│   ├── 01_Bayesian_Neural_Networks_with_Variational_Inference.py
│   ├── 02_Contrastive_Self_Supervised_Learning_SimCLR.py
│   ├── 03_Custom_Transformer_Architecture_with_KV_Caching.py
│   ├── 04_Deep_Generative_Infilling_via_VAEs.py
│   ├── 05_Differentiable_Physics_Informed_Neural_Networks.py
│   ├── 06_LLM_Guided_Synthetic_Data_Generation_Pipeline.py
│   ├── 07_Graph_Neural_Networks_for_Fraud_Ring_Detection.py
│   ├── 08_Multi_Agent_RL_Market_Simulation.py
│   ├── 09_Self_Rerouting_Agentic_RAG_Drift_Correction.py
│   └── 10_Streaming_Anomaly_Detection_Isolation_Forests.py
└── data_engineering/
    ├── 01_Adversarial_Web_Scraping_Fingerprint_Mutation.py
    ├── 02_Async_API_Web_Harvester_Rate_Limiting.py
    ├── 03_Automated_Schema_Evolution_Auditor.py
    ├── 04_CDC_Pipeline_Binary_Log_Emulation.py
    ├── 05_High_Performance_ETL_Columnar_Parquet.py
    ├── 06_Memory_Mapped_Tabular_Join_Deduplication.py
    ├── 07_Micro_Batch_Streaming_Ingestion_Backpressure.py
    ├── 08_Multi_Threaded_Distributed_Web_Crawler.py
    ├── 09_Vectorized_Signal_Processing_ETL.py
    └── 10_Vectorized_Event_Stream_Log_Parser.py
```

---

![Data Science Machine Learning](https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## 🔬 1. Data-Science Repository

Focuses on exact mathematical formulations, custom PyTorch layers, loss engineering, and real-time inference strategies.

<details>
<summary><b>01. Bayesian Neural Networks (BNNs) with Variational Inference (14 KB)</b></summary>

* **Architecture Core:** Implements BayesByBackprop over dense weights to quantify epistemic and aleatoric uncertainty.
* **Mathematical Foundation:** Minimizes the Evidence Lower Bound (ELBO):
  $$\mathcal{L}(\theta) = \mathbb{E}_{q(\mathbf{w}|\theta)}[\log p(\mathcal{D}|\mathbf{w})] - D_{\text{KL}}(q(\mathbf{w}|\theta) \parallel p(\mathbf{w}))$$
* **Key Mechanisms:** Monte Carlo sampling across weight posterior distributions $q(\mathbf{w}|\theta)$, dynamic reparameterization trick for Gaussian distributions, and variance scaling priors.
</details>

<details>
<summary><b>02. Contrastive Self-Supervised Learning (SimCLR) for Unlabeled Data (15 KB)</b></summary>

* **Architecture Core:** Self-supervised visual representation extractor using dual data augmentations and dynamic projection heads.
* **Loss Function:** Normalized Temperature-scaled Cross Entropy Loss (NT-Xent):
  $$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$
* **Key Mechanisms:** Multi-crop random transformations, projection head dimensionality reduction, and cosine distance batch optimization.
</details>

<details>
<summary><b>03. Custom Transformer Architecture from Scratch with KV Caching (12 KB)</b></summary>

* **Architecture Core:** Decoder-only autoregressive language model built natively with multi-head self-attention.
* **Optimization:** Low-latency Key-Value (KV) caching mechanisms reducing inference decoding complexity per token from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.
* **Key Mechanisms:** Rotary Positional Embeddings (RoPE), pre-layer normalization, and scaled dot-product attention state retention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
</details>

<details>
<summary><b>04. Deep Generative Infilling via Variational Autoencoders (11 KB)</b></summary>

* **Architecture Core:** Generative image inpainting utilizing continuous latent space sampling.
* **Mathematical Engine:** Employs probabilistic encoder-decoder pipelines subject to reconstruction loss combined with KL-Divergence regularization.
* **Key Mechanisms:** Reparameterization trick ($\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$), mask-guided loss gating, and latent manifold continuity optimization.
</details>

<details>
<summary><b>05. Differentiable Physics-Informed Neural Networks (PINNs) (11 KB)</b></summary>

* **Architecture Core:** Scientific machine learning pipeline integrating partial differential equations (PDEs) directly into model gradients.
* **Loss Formulation:** Total loss regularized by physics residuals:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \cdot \frac{1}{N} \sum \left| \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} \right|^2$$
* **Key Mechanisms:** Dynamic Hessian/Laplacian higher-order autograd tensors and boundary-condition enforcement layers.
</details>

<details>
<summary><b>06. End-to-End LLM-Guided Synthetic Data Generation Pipeline (18 KB)</b></summary>

* **Architecture Core:** Automated structured dataset synthesis driven by Large Language Models and strict schema rules.
* **Validation Layer:** Integrated Pydantic dynamic models for real-time statistical constraint validation.
* **Key Mechanisms:** Dynamic prompt engineering loops, formal schema inference, and variance-driven synthetic sampling.
</details>

<details>
<summary><b>07. Graph Neural Networks (GNN) for Fraud Ring Detection (16 KB)</b></summary>

* **Architecture Core:** Graph Convolutional Networks (GCN) and GraphSAGE layers for fraud cluster detection.
* **Graph Aggregation:**
  $$h_v^{(k)} = \sigma \left( W \cdot \text{AGGREGATE}\left(\{h_u^{(k-1)}, \forall u \in \mathcal{N}(v)\}\right) \right)$$
* **Key Mechanisms:** Sparse matrix message passing, edge feature propagation, and topological clustering algorithms.
</details>

<details>
<summary><b>08. Multi-Agent Reinforcement Learning (MARL) for Market Simulation (16 KB)</b></summary>

* **Architecture Core:** Decentralized Multi-Agent Actor-Critic framework for order book and market execution modeling.
* **Key Mechanisms:** Independent Proximal Policy Optimization (IPPO), dynamic reward shaping, and Nash equilibrium convergence metrics.
</details>

<details>
<summary><b>09. Self-Rerouting Agentic RAG with Real-Time Data Drift Correction (13 KB)</b></summary>

* **Architecture Core:** Adaptive Retrieval-Augmented Generation agent with automated vector distribution monitoring.
* **Key Mechanisms:** Vector embedding drift evaluation via cosine similarity thresholds, automated prompt fallback paths, and dynamic semantic rerouting.
</details>

<details>
<summary><b>10. Streaming Anomaly Detection with Online Isolation Forests (14 KB)</b></summary>

* **Architecture Core:** Out-of-core streaming anomaly detection engine designed for real-time feature streams.
* **Key Mechanisms:** Dynamic subsampling trees, rolling window path-length evaluation, and minimal-memory partition updates.
</details>

---

![Data Engineering Infrastructure](https://images.pexels.com/photos/1181675/pexels-photo-1181675.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## ⚡ 2. Data-Engineering Repository

High-throughput, memory-efficient data processing systems optimized for concurrency and minimal overhead.

```
                      [ BINARY DATA / EVENT STREAM ]
                                     │
      ┌──────────────────────────────┴──────────────────────────────┐
      ▼                                                             ▼
┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐
│     ZERO-COPY COLUMNAR ETL ENGINE       │       │    OS MMAP POINTER JOIN ENGINE          │
├─────────────────────────────────────────┤       ├─────────────────────────────────────────┤
│ • Run-Length Encoding (RLE)             │       │ • OS Page Slicing (mmap)                │
│ • Dictionary Encoding                   │       │ • Out-of-Core External Sort-Merge       │
│ • Direct PyArrow Pointer Pass-through   │       │ • Memory Boundary Guard Rails           │
└─────────────────────────────────────────┘       └─────────────────────────────────────────┘
      │                                                             │
      └──────────────────────────────┬──────────────────────────────┘
                                     ▼
                      [ HIGH-THROUGHPUT ANALYTICS ]
```

<details>
<summary><b>01. Adversarial Web Scraping Engine with Dynamic Fingerprint Mutation (14 KB)</b></summary>

* **Engine Core:** Evasion harvester engineered to bypass anti-bot systems.
* **Key Mechanisms:** Dynamic TLS cipher suite rotation, canvas/WebGL fingerprint scrambling, and headless browser header mutation.
</details>

<details>
<summary><b>02. Asynchronous API Web Harvester with Rate-Limiting and Proxy Rotation (14 KB)</b></summary>

* **Engine Core:** Concurrent HTTP harvester built on `aiohttp` and `asyncio`.
* **Key Mechanisms:** Async token-bucket rate limiters, proxy pool health-checking, and backoff retry logic.
</details>

<details>
<summary><b>03. Automated Schema Evolution and Type Inference Auditor (15 KB)</b></summary>

* **Engine Core:** Data lake monitoring engine for tracking schema drift across incoming JSON/Parquet batches.
* **Key Mechanisms:** Structural diff computation, automatic type promotion inference, and Null-ratio anomaly alerting.
</details>

<details>
<summary><b>04. Event-Driven Change Data Capture (CDC) Pipeline with Binary Log Emulation (13 KB)</b></summary>

* **Engine Core:** Database state replication engine mimicking MySQL/PostgreSQL binlog streaming.
* **Key Mechanisms:** Merkle tree state comparison, transaction delta tracking, and event stream serialization.
</details>

<details>
<summary><b>05. High-Performance ETL Engine for Columnar Parquet Files from Scratch (12 KB)</b></summary>

* **Engine Core:** Custom columnar storage reader and writer built without external heavy database engines.
* **Key Mechanisms:** Run-Length Encoding (RLE), dictionary compression, and zero-copy memory buffers.
</details>

<details>
<summary><b>06. Memory-Mapped Large-Scale Tabular Join and De-duplication Engine (11 KB)</b></summary>

* **Engine Core:** Out-of-core join processor designed to exceed physical RAM limits.
* **Key Mechanisms:** OS-level `mmap` pointer slicing, multi-way external merge sorting, and zero-allocation hash tables.
</details>

<details>
<summary><b>07. Micro-Batch Streaming Ingestion Pipeline with Backpressure Management (14 KB)</b></summary>

* **Engine Core:** Ingestion framework handling high-burst streaming sources.
* **Key Mechanisms:** Dynamic lock-free token-bucket queues, NumPy ring-buffers, and backpressure signal propagation.
</details>

<details>
<summary><b>08. Multi-Threaded Distributed Web Crawler with Fault-Tolerant State Recovery (18 KB)</b></summary>

* **Engine Core:** High-concurrency Web crawler featuring transactional state persistence.
* **Key Mechanisms:** Transactional SQLite URL frontier tracking, atomic checkpointing, and thread-pool execution controls.
</details>

<details>
<summary><b>09. Vectorized Multi-Dimensional Signal Processing ETL Engine (14 KB)</b></summary>

* **Engine Core:** Fast Fourier Transform (FFT) signal processing pipeline.
* **Key Mechanisms:** Vectorized C-level NumPy matrix operations, Butterworth frequency domain filters, and dynamic Z-score outlier masking.
</details>

<details>
<summary><b>10. Vectorized Real-Time Event Stream Log Parser and Feature Store (11 KB)</b></summary>

* **Engine Core:** High-speed string and log processing framework.
* **Key Mechanisms:** Zero-regex byte-array slice matching, vectorized string array transforms, and continuous feature store updates.
</details>

---

## 🛠 Setup & Installation

### Prerequisites
* **Python**: 3.10 or higher
* **C++ Compiler**: GCC/Clang (for native vector extensions)
* **CUDA Toolkit**: 11.8+ (Optional, for GPU acceleration)

### Installation
```bash
# Clone the infrastructure repository
git clone https://github.com/rayaantasnim/Data-Primitives.git
cd Data-Primitives

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade core build components
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Quickstart Usage

### Running a Data Science Module (Transformer with KV-Cache)
```python
from data_science.custom_transformer import CustomTransformerDecoder

# Initialize custom transformer architecture
model = CustomTransformerDecoder(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    max_seq_len=2048
)

# Generate logits using KV caching for accelerated sampling
tokens = [1, 104, 305]  # Example input prompt
logits, kv_cache = model.forward_with_cache(tokens)
print("Logits Shape:", logits.shape)
```

### Running a Data Engineering Engine (Memory-Mapped Out-of-Core Join)
```python
from data_engineering.mmap_join_engine import MemoryMappedJoinEngine

# Instantiate memory-mapped join system
engine = MemoryMappedJoinEngine(
    left_filepath="data/large_dataset_A.bin",
    right_filepath="data/large_dataset_B.bin",
    output_filepath="data/joined_output.bin"
)

# Execute out-of-core sort-merge join without exceeding memory limits
engine.execute_join(join_key_index=0, max_ram_mb=512)
```

---

<p align="center">
  Engineered with Precision by Rayaan Tasnim<br>
  © Data-Primitives: Data Science & Data Engineering Architectures; All rights reserved<br>
  Powered by Olympiad Edge
</p>