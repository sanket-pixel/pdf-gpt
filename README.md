## I. Profiling & Performance Analysis

### End-to-end latency vs throughput, P99
1. What is end-to-end latency? </br>
Time taken for a single inference request to complete — from input arrival to output response, including preprocessing, model execution, and postprocessing.

2. What is throughput?</br>
Number of inference requests processed per unit time, usually measured in requests per second (RPS) or samples per second.

3. How are latency and throughput related?</br>
They are inversely related for a fixed compute budget. Increasing batch size improves throughput but increases per-request latency.

4. What is P99 latency?</br>
The 99th percentile latency — 99% of all requests complete faster than this value. It represents tail latency rather than average latency.

5. Why is P99 latency important?</br>
It captures worst-case behavior critical for real-time systems where occasional slow requests can violate timing requirements.

6. What typically affects latency vs throughput tradeoff?</br>
Batch size, parallelism (GPU streams), I/O scheduling, synchronization points, and kernel launch overhead.

7. What is pinned (page-locked) memory?</br>
Pinned memory is a region of host (CPU) memory that the operating system guarantees will not be moved, paged out, or swapped. It stays resident in physical RAM.

8. Why does the GPU need pinned memory?</br>
Because the GPU’s DMA engine performs direct reads/writes to system memory using physical addresses. If those pages could move, the GPU might read invalid or stale data.

9. What happens if you use normal (pageable) memory?</br>
CUDA must first allocate an internal pinned buffer, copy your data into it, then start DMA to the GPU.
This extra copy adds latency and prevents asynchronous overlap.

10. What is DMA (Direct Memory Access)?</br>
A hardware mechanism that lets the GPU (or any PCIe device) transfer data between GPU VRAM and host DRAM without CPU intervention.
The CPU only sets up descriptors (source, destination, size); the actual transfer happens autonomously.

11. Why does page-locking enable asynchronous transfers?</br>
With pinned memory, the GPU knows the physical addresses and can perform non-blocking DMA in the background.
Without pinning, the driver must synchronize and perform the transfer immediately, blocking the CPU.

12. Are there downsides?</br>
Yes. Pinned memory reduces available pageable RAM for the OS. Allocating too much can hurt system performance.
Use it only for active data batches.

13. How can you confirm async transfer works?</br>
Use nsys or the PyTorch profiler — you’ll see cudaMemcpyAsync events overlapped with GPU kernels on the timeline instead of serialized before them.

14. In one sentence — why is this important?</br>
Pinned memory enables the GPU to pull data directly from CPU RAM via DMA, allowing asynchronous, high-bandwidth data transfers that keep the GPU fully utilized during inference or training.

15. What are the benefits?</br>
Faster data transfer (~2× vs pageable memory).
CPU and GPU can overlap work — I/O and compute run in parallel.
Reduces GPU idle time → better throughput and lower latency.


### Compute- vs Memory-bound

1. What does it mean for a workload to be memory-bound vs compute-bound?</br>
A compute-bound workload is limited by how many arithmetic operations (FLOPs) the GPU can perform per second. The GPU’s ALUs or Tensor Cores are saturated — adding more memory bandwidth wouldn’t help.
A memory-bound workload is limited by how fast data can be moved between memory and compute units. The ALUs spend time idle waiting for operands to arrive from DRAM or caches.
In other words, the ratio of operations performed per byte fetched (arithmetic intensity) determines which side dominates. Low arithmetic intensity → memory-bound; high arithmetic intensity → compute-bound.

2. How can you identify whether a model or kernel is memory-bound or compute-bound?</br>
You can determine this by profiling with Nsight Compute or PyTorch Profiler and looking at metrics such as:
SM (Streaming Multiprocessor) Active %: High = compute-bound; low = memory stalls.
Memory Throughput vs Peak Bandwidth: If near 100%, you’re memory-bound.
FLOPs Utilization (Tensor Core or FP32): If low utilization but high memory traffic, memory-bound.
Roofline Analysis: Compare operational intensity (FLOPs/byte) to the hardware roofline; points below the memory roof → memory-bound, near the compute roof → compute-bound.

4. In deep learning inference, what kinds of layers are typically memory-bound and which are compute-bound?</br>
Compute-bound: Large dense matrix multiplies and convolutions (e.g., GEMM, Conv2D, self-attention matmuls). These perform many FLOPs per byte of data.
Memory-bound: Elementwise ops like activation functions, layer normalization, bias adds, residual adds, and small tensor reshapes. They involve few operations per memory access, so bandwidth dominates.
In transformers, the matmul (QKᵀ and AV) are compute-heavy, but the surrounding normalization and softmax stages are memory-bound.

5. How can you optimize memory-bound kernels?</br>
Fuse operations (e.g., LayerNorm + bias + activation) to reuse data while it’s in registers or shared memory, reducing memory traffic.
Use mixed precision (FP16/BF16) to cut data size in half — fewer bytes moved per FLOP.
Improve data locality — tile data to fit in shared memory or L2 cache.
Use tensor layouts optimized for memory coalescing (e.g., NHWC for Tensor Cores).
Overlap compute and memory using CUDA streams or prefetching.
Goal: reduce the number of global memory transactions and maximize data reuse.

6. How can you optimize compute-bound kernels?</br>
Use Tensor Cores or lower-precision math (FP16/INT8) to increase throughput per FLOP.
Maximize occupancy — ensure enough warps are scheduled per SM to hide instruction latency.
Use fused kernels to minimize kernel launch overhead between compute stages.
Unroll loops / use vectorized loads to keep ALUs fed continuously.
Leverage compiler or library optimizations — cuBLAS, cuDNN, or TensorRT kernels already approach hardware limits.
For compute-bound layers, your main job is to increase arithmetic intensity or make better use of specialized units (Tensor Cores, MMA ops).

7. How would you explain arithmetic intensity in this context?</br>
Arithmetic intensity (AI) = FLOPs performed / bytes accessed from memory.
It measures how much computation is done per memory transfer.
A high AI (e.g., dense matmul) means most data stays in registers or cache and is reused → compute-bound.
A low AI (e.g., elementwise ops) means you’re reading/writing lots of data per simple operation → memory-bound.
NVIDIA’s roofline model uses AI to visually separate these regimes — helping decide if you should optimize memory movement or computation.

8. Background: how a GPU executes work</br>
Each GPU has multiple Streaming Multiprocessors (SMs).
Each SM can execute hundreds or thousands of lightweight threads in groups of 32 threads, called warps.
At any moment Only a few warps are actively executing instructions on the SM’s functional units.
Others may be waiting — for memory loads, synchronization, or dependencies.

8. The latency-hiding problem</br>
Global memory access on a GPU can take hundreds of cycles (≈400–800 cycles).
If a warp issues a ld.global instruction (load from global memory), that warp can’t proceed until data arrives.
The key idea:
Instead of stalling, the SM switches to another ready warp whose data is already available.
This context switch costs zero cycles — warps are hardware-scheduled.
This ability to switch warps instantly is how GPUs hide latency instead of reducing it.

9. What “occupancy” means</br>
Occupancy = number of active warps per SM ÷ maximum warps an SM can hold.
Example:
If an SM can hold 64 warps (architecture-dependent),
and your kernel launches 32 warps that fit within resource limits (registers, shared memory),
then occupancy = 50%.
The more warps you have ready to run, the better chance the scheduler has to find one that’s not waiting on memory → higher ALU utilization.

10. What limits occupancy</br>
Each thread block consumes finite SM resources:
Registers (per-thread),
Shared memory (per-block),
Threads per block.
If your kernel uses too many registers or too much shared memory per block, the SM can host fewer blocks → fewer warps → lower occupancy.

11. Why “maximizing occupancy” helps </br>
With high occupancy:
When one warp stalls on memory, others are ready to execute,
The SM’s compute units (ALUs/Tensor Cores) stay busy,
Latency from memory or dependencies is effectively hidden.
However, occupancy only helps until you have enough warps to cover latency. Beyond that, returns diminish — the kernel may become bandwidth- or compute-limited instead.
Example
Suppose each global load takes 400 cycles, and a warp’s instruction throughput is 1 instruction every 4 cycles.
If you have only one warp per SM:
After the load, it sits idle for 400 cycles.
If you have 8 warps:
While warp 0 waits on data, warp 1 executes,
While warp 1 waits, warp 2 executes,
… and so on, effectively hiding memory latency.
The GPU’s goal is to keep the ALUs busy all the time.

12. In Deep Learning kernels </br>
In GEMMs or convolution kernels:
High occupancy ensures matrix tiles are continuously processed.
Low occupancy can lead to SMs sitting idle between global memory fetches of tiles.
Frameworks like cuBLAS, cuDNN, and TensorRT already balance register/shared memory usage to achieve near-max occupancy

### Precision & Quantization

1. What does precision mean in deep learning inference? </br>
Precision defines how many bits are used to represent numbers (weights, activations, gradients).
Common formats:
FP32 (32-bit float)
FP16 / BF16 (16-bit float)
INT8 (8-bit integer)
INT4 / FP8 (emerging for LLMs)
Lower precision reduces the range and resolution of representable values, affecting numerical stability and model accuracy — but drastically improves memory bandwidth, cache fit, and computational throughput.

2. Why does precision affect inference latency and throughput? </br>
Because hardware throughput scales inversely with data width:
A GPU can perform twice as many FP16 or BF16 ops per cycle as FP32.
INT8 or FP8 can provide 4× to 8× higher throughput, as Tensor Cores and DPUs can process multiple narrow values in parallel.
Lower precision also:
Cuts memory footprint (less data to load/store).
Reduces PCIe and DRAM bandwidth needs.
Improves cache reuse and fits more activations into on-chip SRAM.
However, latency improves only if the workload was compute-bound.
If the workload is already memory-bound, the gain may be smaller.

3. What is quantization, and how does it relate to precision? <br>
Quantization is the process of converting continuous (float) values to discrete integer representations.
Example (8-bit affine quantization):
q=round(x/s+z)
q=round(x/s+z)
where s = scale, z = zero-point.
It reduces precision (fewer representable values) to compress the model and enable integer arithmetic on specialized hardware (INT8 Tensor Cores, DSPs).
Precision refers to the datatype used; quantization is the method to achieve lower precision while preserving accuracy.

4. How does lowering precision or quantizing affect accuracy?<br>
FP16 / BF16: negligible accuracy loss for most models; rounding errors are small.
INT8: can degrade accuracy if quantization parameters are poorly calibrated, especially in layers with wide dynamic range.
INT4 / FP8: significant accuracy drop unless fine-tuned or using quantization-aware training.
Accuracy loss mainly arises from clipping and rounding, which distort weight/activation distributions.
Calibration (collecting activation statistics on representative data) minimizes this.

5. What are the trade-offs between precision, latency, and accuracy?</br>
Format	Accuracy Impact	Latency / Throughput	Memory	Typical Use
FP32	Baseline	Slowest	Largest	Training, baseline inference
FP16 / BF16	~0–1% loss	~2× faster	½ memory	Most inference workloads
INT8	1–3% loss (well-calibrated)	3–4× faster	¼ memory	Edge / datacenter inference
INT4 / FP8	5–10% loss unless retrained	5–8× faster	⅛ memory	LLM, ultra-low-latency


### INT8 PTQ vs QAT; per-tensor vs per-channel

1. What is Post-Training Quantization (PTQ)?</br>
PTQ quantizes a pretrained FP32 model after training without updating weights.
It runs a calibration step on representative data to collect activation statistics (min/max or distribution histograms) and computes quantization parameters (scale and zero-point).
It’s fast and doesn’t require retraining, but accuracy can degrade, especially for layers with wide or skewed activation ranges (e.g., softmax, layernorm).
Pros: quick deployment, no retraining cost.
Cons: accuracy may drop, especially for small dynamic ranges or sensitive layers.

2. What is Quantization-Aware Training (QAT)? </br>
QAT simulates quantization effects during training (using fake quantization nodes).
The forward pass quantizes weights and activations; the backward pass updates the underlying FP32 weights to learn robustness against quantization noise.
This trains the network to adapt its weight distributions to the discrete representation, restoring accuracy close to FP32 even with INT8 or INT4 deployment.
Pros: best accuracy retention; suitable for low-precision (INT8, INT4).
Cons: requires retraining; more complex pipeline.

3. How do PTQ and QAT compare in practice?</br>
Aspect	PTQ	QAT
Training needed	No	Yes (fine-tuning)
Speed of setup	Fast	Slower
Accuracy loss	Moderate (1–3%)	Minimal (<1%)
Use cases	Datacenter, edge where retraining not possible	Mission-critical, INT4, FP8, mobile deployment
Typical frameworks	TensorRT PTQ Calibrator, ONNXRuntime INT8	PyTorch QAT, TensorRT-QAT, TF Lite QAT


### CPU↔GPU bottlenecks, Amdahl’s law
1. What are CPU↔GPU bottlenecks in deep learning inference?</br>
They occur when data transfer or synchronization between the CPU (host) and GPU (device) limits overall performance. Even if GPU kernels are fast, time is lost moving data (H2D/D2H) or waiting for CPU coordination. Common bottlenecks include slow PCIe transfers, frequent kernel launches (host overhead), and synchronous API calls that stall GPU execution.

2. What causes CPU↔GPU transfer bottlenecks?</br>
Data must traverse the PCIe or NVLink bus. PCIe bandwidth (~12–16 GB/s) is far lower than GPU VRAM bandwidth (~1 TB/s).
If every batch or layer requires CPU↔GPU data movement, the bus becomes a choke point. 
Non-pinned memory and synchronous transfers exacerbate this by preventing overlap of copy and compute.

3. How can you mitigate CPU↔GPU bottlenecks?</br>
Use pinned memory for fast DMA transfers, asynchronous (non-blocking) data copies to overlap 
compute and I/O, keep model weights resident on GPU, batch small transfers, and use CUDA Graphs or 
torch.compile to reduce kernel launch overhead. For high-throughput inference, move preprocessing to 
GPU and minimize CPU-side control logic.

4. What is Amdahl’s Law and why is it relevant to GPU performance?</br>
Amdahl’s Law states that the maximum achievable speedup of a system is limited by the fraction of work that cannot be parallelized.

5. How does Amdahl’s Law explain CPU↔GPU inefficiency?</br>
Even if GPU kernels run 100× faster than CPU code, end-to-end latency won’t scale proportionally if the CPU handles 
significant sequential work (data loading, preprocessing, synchronization). The CPU portion becomes the non-parallelizable
fraction (1 − P) that dominates total time. Therefore, optimizing CPU–GPU overlap, offloading more logic to GPU, and 
minimizing host overhead are essential to realize true GPU acceleration.
PyTorch Profiler / NVTX / Nsight Systems / Nsight Compute

6. How do pinned memory and asynchronous transfers help reduce CPU↔GPU bottlenecks?</br>
Pinned (page-locked) memory ensures that host memory pages stay fixed in physical RAM, 
allowing the GPU’s DMA engine to directly read or write data without an intermediate copy. 
This enables true asynchronous data transfers using non-blocking operations (non_blocking=True), 
allowing data movement and computation to overlap. While the GPU executes one batch, the next batch 
can already be transferred over PCIe in the background. This hides transfer latency, keeps the GPU 
fed with data, and significantly improves overall throughput by preventing idle GPU cycles caused by 
waiting on slow host-to-device copies.

7. What is the fundamental difference between PCIe bus bandwidth and GPU VRAM bandwidth?</br>
The PCIe bus connects the CPU and GPU and determines how fast data can move between host memory and device memory. 
Its bandwidth is limited by physical link speed and protocol overhead, typically in the range of tens of gigabytes 
per second. GPU VRAM bandwidth, in contrast, measures how fast the GPU’s own cores can read and write data from 
its onboard high-speed memory (GDDR or HBM), which operates in the terabytes-per-second range. The huge gap between
these two bandwidths means that once data is inside VRAM, computations can proceed at full speed, but frequent PCIe 
transfers to or from host memory immediately become the performance bottleneck.


### Kernel launch overhead, CUDA Graphs
1. What is kernel launch overhead?</br>
Each time the CPU instructs the GPU to execute a CUDA kernel, it must perform a launch through 
the CUDA driver stack. This involves preparing kernel arguments, managing synchronization, 
and queuing the launch on a stream. Even though a single launch costs only a few microseconds, 
deep learning models can invoke thousands of small kernels per inference step (e.g., elementwise ops,
 activations, reshapes). The accumulated overhead from these frequent CPU→GPU dispatches can dominate
runtime, especially when the GPU executes very short kernels faster than the CPU can issue new ones.

2. Why does kernel launch overhead limit inference speed?</br>
Modern GPUs can complete many lightweight operations in nanoseconds, but the CPU is comparatively 
slow at scheduling and issuing new kernels. When every small layer or tensor operation requires a 
separate kernel launch, the GPU spends time idle between launches, waiting for the CPU to enqueue 
the next task. This creates a host-side bottleneck where CPU dispatch latency, not raw GPU compute 
power, caps throughput and increases end-to-end latency.

3. What are CUDA Graphs, and how do they address this problem?</br>
CUDA Graphs allow developers to capture an entire sequence of GPU operations (kernels, memory 
copies, dependencies) as a reusable, pre-optimized execution graph. Once captured, the graph 
can be launched with a single API call, eliminating per-kernel CPU launch overhead. This means 
 the CPU sets up the sequence once, and subsequent inferences replay it directly on the GPU,
achieving lower latency and higher determinism. Frameworks like PyTorch and TensorRT integrate
CUDA Graphs to make repeated inference workloads almost entirely GPU-driven.

4. How does torch.compile() help reduce launch overhead?</br>
torch.compile() in PyTorch 2.x captures and fuses multiple small operations into fewer, 
larger composite kernels. It does so by tracing the computation graph, generating optimized 
code via backends like TorchInductor, Triton, or NVFuser. By reducing the number of kernel 
 launches and combining operations that previously required CPU orchestration, torch.compile() 
minimizes dispatch overhead and improves GPU utilization. The result is fewer launches, 
longer-running kernels, and smoother execution that better matches the GPU’s parallel 
 hardware capabilities.

5. How do CUDA Graphs and torch.compile() differ in their approach?</br>
CUDA Graphs focus on replaying a pre-defined execution pattern efficiently, 
removing runtime launch overhead but not changing the kernel structure. It’s
ideal for steady-state, repeatable inference loops. torch.compile(), on the 
other hand, rewrites and fuses kernels at the framework level to reduce the number
of launches in the first place. Combined use — compiling a model and then capturing 
it with CUDA Graphs — provides both graph-level optimization and near-zero CPU
 dispatch overhead during inference.

### FP32/FP16/BF16 trade-offs, Tensor Cores
1. What are the key differences between FP32, FP16, and BF16 formats?</br>
FP32 (single precision) uses 32 bits with 23 mantissa bits, 8 exponent bits, and
1 sign bit — offering high precision and dynamic range. FP16 (half precision) uses 16 
bits (10 mantissa, 5 exponent, 1 sign), reducing precision and range but halving memory and bandwidth use.
BF16 (bfloat16) also uses 16 bits but keeps the same 8-bit exponent as FP32 and a shorter mantissa (7 bits). 
This retains dynamic range close to FP32 while reducing precision. BF16 avoids most underflow/overflow issues common 
in FP16 and is easier to train or infer with on modern hardware.

2. What are the trade-offs between these precisions in inference workloads?</br>
FP32: High numerical stability but high memory use and latency; often overkill for inference.
FP16: Doubles throughput and halves memory footprint but can suffer from overflow/underflow without scaling.
BF16: Nearly the same range as FP32 with FP16-level efficiency; ideal for mixed-precision inference and training.
Lower precision improves speed and reduces bandwidth, but if numerical stability or small value sensitivity matters,
accuracy can drop without proper scaling or quantization-aware calibration.

3. How do Tensor Cores use these formats for acceleration?</br>
Tensor Cores are specialized matrix-multiply-and-accumulate (MMA) units inside NVIDIA GPUs designed
to process FP16, BF16, TF32, and INT8 operations at very high throughput. They perform matrix 
multiplications (A × B + C) in mixed precision — multiplying low-precision operands (FP16/BF16) while
accumulating results in higher precision (FP32). This allows massive compute throughput (hundreds of TFLOPs) 
compared to traditional FP32 CUDA cores. Tensor Cores are used heavily in deep learning layers like GEMM,
convolution, and attention.

4. How do Tensor Cores differ from standard CUDA cores?</br>
CUDA cores execute general-purpose scalar and vector operations (add, multiply, logic) 
for individual threads, optimized for flexibility. Tensor Cores, by contrast, are fixed-function
matrix engines optimized for dense linear algebra on small tiles (e.g., 16×16 matrices). They deliver 
much higher FLOP/s per watt for workloads expressible as matrix multiplies. Most deep learning frameworks 
automatically map GEMM and Conv2D ops to Tensor Cores when using FP16/BF16/TF32 datatypes.

5. What are the main precision-related performance and accuracy trade-offs in practice?</br>
Using FP16 or BF16 allows 2–4× faster matrix math and halves memory transfers, reducing latency and energy 
consumption. However, naive use can lead to numerical instability, especially in normalization or 
accumulation-heavy layers. BF16 typically offers the best balance between performance and reliability. 
FP32 remains necessary for operations requiring very high precision (e.g., loss computation, model calibration), 
while Tensor Core–optimized formats (FP16/BF16/TF32/INT8) are preferred for production inference where speed and
efficiency dominate.

6. How does reducing precision from FP32 to FP16 actually improve inference speed for a large
matrix multiplication, such as 1024×1024?</br>
Switching from FP32 (4 bytes per value) to FP16 (2 bytes per value) halves the data size, 
so the GPU moves half as much information through memory and caches. This directly cuts memory transfer
time and improves cache reuse. On the compute side, Tensor Cores can execute several FP16 operations per
cycle instead of one FP32, effectively doubling or quadrupling math throughput. Together, reduced memory 
traffic and higher arithmetic density allow a large matmul to run about 2–4× faster, depending on whether
it’s memory- or compute-limited.

7. What exactly happens inside the GPU during mixed-precision multiply–accumulate operations?</br>
Inputs A and B are stored as FP16 (or BF16) values. When a Tensor Core processes them, it temporarily 
widens each operand to FP32 for the multiply, computes the product in FP32 precision, and accumulates the 
result into an FP32 register. After all partial sums are complete, the final output can be converted back 
to FP16 for storage. This approach keeps computation stable while still benefiting from lower-precision data movement.

8. Why do modern GPUs perform FP16×FP16 multiplications in FP32 precision internally instead 
of staying in pure FP16 arithmetic?</br>
Pure FP16 math uses a 10-bit mantissa and easily loses precision when summing many terms, leading to rounding 
and overflow errors. By performing the multiply and accumulation in FP32, the GPU retains much higher 
numerical accuracy—enough to match FP32-level results for most deep learning layers—while still using 
FP16 for input and output to save bandwidth and memory. This mixed-precision path balances speed and
stability, which is why frameworks and TensorRT default to FP16-input / FP32-accumulate kernels for inference.



### Weight-only quant (INT8/INT4), activation quant
1. What is weight-only quantization, and how does it differ from full quantization?</br>
Weight-only quantization converts only the model’s weights (parameters) from floating point to lower-bit integers such as INT8 or INT4, while keeping activations (intermediate tensors) in higher precision (FP16 or FP32). This reduces model size and memory bandwidth but avoids the accuracy loss that can occur when quantizing dynamic activations. It’s simpler to deploy since no activation calibration is needed and often yields large memory savings with minimal accuracy impact.

2. How does weight-only quantization improve inference efficiency?</br>
Lowering weight precision cuts the amount of data fetched from memory per multiply–accumulate. For example, INT8 weights use 1 byte instead of 2 or 4, reducing model bandwidth by up to 4×. The compute units (Tensor Cores or integer ALUs) can process more operations per cycle using low-precision integer math. Even when activations stay in FP16, the mixed-format computation (FP16 × INT8) significantly lowers memory traffic and improves throughput, especially in large matrix multiplications like transformer attention or feed-forward layers.

3. What is activation quantization, and why is it more challenging than weight-only quantization?</br>
Activation quantization converts intermediate outputs to lower precision (INT8/INT4) as well. Unlike weights, activations vary with input data, so their range must be estimated via calibration or quantization-aware training. Poor calibration can cause clipping or underflow, severely affecting accuracy. Activation quantization demands dynamic scaling or per-layer calibration, whereas weight-only quantization can use static scales derived from model parameters.

4. How do frameworks implement weight-only vs activation quantization in practice?</br>
Weight-only: Applied offline; model weights are quantized and stored as integers. During inference, they are dequantized or multiplied directly in integer form. Used in TensorRT-LLM, GPTQ, and bitsandbytes.
Full (weights + activations): Requires runtime quant/dequant steps for activations, adding overhead but maximizing end-to-end compression. Supported by TensorRT INT8 and QAT workflows, which use calibration datasets to determine activation scales.

5. When should you choose weight-only quantization over full INT8 quantization?</br>
Weight-only quantization is ideal when memory footprint or model loading bandwidth dominates and you want minimal engineering effort or accuracy loss. It suits large transformer models and LLMs where weight storage is the main bottleneck. Full quantization provides greater latency and energy savings when hardware supports true INT8 compute for both weights and activations but requires careful calibration or retraining. In short: weight-only is the low-risk optimization; full quantization offers maximum compression at higher complexity and accuracy cost.
KV-cache quantization (LLMs/VLMs)

### Graph & Runtime Optimizations
1. What are graph-level optimizations in deep learning frameworks?</br>
Graph-level optimizations analyze the entire computation graph to eliminate redundant operations and restructure execution for efficiency. Common examples include constant folding (precomputing static results), operator fusion (combining elementwise ops into one kernel), dead-code elimination (removing unused branches), and layout transformation (choosing tensor memory formats that best match hardware). These changes reduce kernel launches, improve cache locality, and minimize memory traffic before runtime execution begins.

2. How do runtime optimizations differ from graph-level optimizations?</br>
Graph optimizations are applied ahead of time — before the model runs — while runtime optimizations happen dynamically as the model executes. Runtime optimizations adapt to actual tensor shapes, batch sizes, and hardware state. Examples include dynamic kernel selection (choosing the fastest algorithm for a given shape), stream scheduling, memory reuse, and automatic mixed precision. Runtime systems like TensorRT or PyTorch’s Inductor re-optimize or cache kernels on the fly to match execution conditions.

3. Why is operator fusion a key optimization for inference?</br>
Each kernel launch from the CPU incurs overhead and forces separate memory reads/writes. By fusing sequential operations (e.g., Linear → BiasAdd → ReLU → Dropout) into a single kernel, the framework keeps data in registers or shared memory, eliminating intermediate memory accesses and CPU dispatches. This reduces latency, improves arithmetic intensity, and fully utilizes GPU compute units. TorchInductor, TensorRT, and XLA all aggressively fuse elementwise ops for this reason.

4. What role do compilers like TorchInductor, XLA, or TensorRT play in optimization?</br>
They transform high-level PyTorch or TensorFlow graphs into optimized GPU code. These compilers perform graph rewriting, kernel fusion, tiling, mixed precision conversion, and memory planning. They may emit code directly in CUDA, Triton, or cuBLAS/cuDNN calls. The result is a smaller number of large, well-optimized kernels instead of many small operations, leading to improved GPU occupancy and reduced CPU-GPU coordination overhead.

5. How do CUDA Graphs and torch.compile() relate to runtime optimization?</br>
CUDA Graphs remove per-kernel CPU launch overhead by capturing and replaying entire GPU execution graphs as single launch units. torch.compile() complements this by performing ahead-of-time fusion and optimization, producing fewer, larger kernels. Together, they create a pipeline where model computation is statically optimized at compile time and executed with minimal runtime CPU interaction, leading to deterministic, low-latency inference execution.
TorchDynamo/torch.compile, Inductor


### Operator fusion (Linear+Bias+Act, Norm+MatMul)</br>
1. What is operator fusion in the context of deep learning inference?
Operator fusion combines multiple consecutive operations into a single GPU kernel to reduce overhead and improve efficiency. Instead of launching separate kernels for each op (e.g., Linear → BiasAdd → ReLU), fusion executes them together, keeping intermediate results in registers or shared memory instead of writing them to global memory. This minimizes memory traffic, reduces kernel launch overhead, and increases GPU occupancy, resulting in faster inference and lower latency.

2. Why is Linear + Bias + Activation fusion effective?</br>
These three operations are sequential, memory-bound, and operate on the same tensor. In a naive implementation, the Linear layer writes its output to VRAM, then a second kernel reads that output to add bias, and a third applies the activation function — resulting in multiple reads/writes. Fusing them into one kernel allows the computation Y = Act(XW + b) to happen entirely in registers, avoiding intermediate global memory access. This yields significant speedups, especially for small or repeated matrix operations common in MLPs and attention blocks.

3. What about fusing Normalization + MatMul or other attention-related ops?</br>
Layers like LayerNorm or RMSNorm often precede large matmuls in transformer architectures. Fusing normalization and matmul lets the GPU compute the normalized tensor and immediately consume it for matrix multiplication, keeping it in on-chip memory. Similarly, fusing attention sub-ops (QK^T, scaling, softmax, and weighted sum) avoids multiple tensor materializations. This is the principle behind FlashAttention — it computes attention without explicitly forming the large attention matrix, fusing softmax, scaling, and dropout into a single kernel.

4. What are the main benefits of operator fusion?</br>
Reduced memory I/O: fewer global reads/writes, less bandwidth pressure.</br>
Lower launch overhead: fewer kernels mean less CPU→GPU scheduling latency.</br>
Better cache/reuse: intermediate results stay in registers/shared memory.</br>
Improved energy efficiency: memory access costs more power than arithmetic.</br>
Higher throughput: especially impactful for transformer-style workloads with many small, sequential ops.

5. How do frameworks and compilers perform fusion automatically?</br>
Modern compilers like TorchInductor, XLA, TensorRT, and TVM analyze computational graphs to detect fusable patterns (e.g., elementwise ops following a GEMM). They generate fused CUDA or Triton kernels that perform the entire subgraph in one pass. Frameworks like TensorRT further fuse across layers (e.g., MatMul + Bias + GELU) using precision-specific Tensor Core kernels. This automatic fusion is one of the biggest contributors to the performance gap between raw PyTorch execution and optimized deployment runtimes.

### CUDA Graph capture in PyTorch/TensorRT
1. What is CUDA Graph capture and why was it introduced?</br>
CUDA Graphs allow capturing a sequence of GPU operations — kernels, memory copies, and synchronization events — into a single executable graph object. Once captured, this graph can be replayed repeatedly with a single launch call. The goal is to eliminate per-kernel CPU launch overhead and make inference highly deterministic. This is especially beneficial for workloads with many small GPU kernels or static inference loops (e.g., transformer blocks, CNN layers), where CPU scheduling latency dominates runtime.

2. How does CUDA Graph capture work conceptually?</br>
During capture mode, the CUDA driver records all GPU operations into a directed acyclic graph (DAG) instead of executing them immediately. Dependencies between kernels, streams, and memory operations are preserved in the graph structure. Once capture ends, the graph is "instantiated" — compiled and optimized internally by the driver. Replaying it executes the entire sequence on the GPU without additional CPU intervention, effectively replacing thousands of kernel launches with one pre-recorded, pre-validated call.

3. How is CUDA Graph capture used in PyTorch?</br>
In PyTorch, torch.cuda.CUDAGraph() provides a simple API for graph capture. You first run a warm-up iteration to allocate all necessary tensors, then wrap your inference code inside a capture block:
   ```python
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       output = model(inputs)
   ```
   Subsequent inferences call g.replay(), executing the pre-captured GPU sequence instantly. All tensors must maintain the same shapes and memory addresses between captures, as the graph assumes static memory layout and execution order.

4. How does TensorRT leverage CUDA Graphs?</br>
TensorRT integrates CUDA Graph capture directly in its engine execution pipeline. When a TensorRT engine runs a fixed-shape inference multiple times, the CUDA driver can capture the underlying kernel launches and replay them as a graph for later executions. This reduces CPU scheduling overhead, improves determinism, and increases throughput — especially for small batch sizes where kernel launch latency is a significant fraction of total time. TensorRT exposes this as "CUDA Graph replay mode" for static-shape inference.

5. What are the benefits and limitations of CUDA Graphs in inference?</br>
Benefits:</br>
Drastically reduces CPU launch overhead.</br>
Improves determinism and jitter (especially for real-time systems).</br>
Enables consistent high GPU utilization across repeated inferences.</br>
Limitations:</br>
Works best for static input shapes and static execution graphs.</br>
Cannot capture dynamic control flow or varying tensor sizes.</br>
Memory addresses must remain fixed; dynamic allocation inside capture is disallowed.</br>

In practice, frameworks combine CUDA Graphs with static-shape inference (e.g., TensorRT or compiled PyTorch models) to achieve ultra-low latency and stable performance for repetitive workloads.


1. What is the difference between Nsight Systems and Nsight Compute?

Answer:
Nsight Systems gives a timeline view of the entire workload — showing kernel launches, data transfers, CPU↔GPU synchronization, and how different processes overlap.
Nsight Compute is a kernel-level profiler — it analyzes one CUDA kernel in depth (occupancy, memory transactions, warp efficiency, instruction mix).

2. How can you identify whether a model is CPU-bound or GPU-bound using PyTorch Profiler?

Answer:
Check the CUDA time percentage versus total CPU time in the profiler output.

High CPU time with small CUDA utilization → CPU-bound (e.g., dataloader bottleneck, syncs).

High CUDA kernel times and idle CPU → GPU-bound (compute or memory limit).

3. How can you tell if a model is memory-bound or compute-bound using Nsight Compute?

Answer:
Look at the Roofline chart — if achieved FLOPs are far below the compute roof but close to the memory roof, the kernel is memory-bound.
Alternatively, compare achieved occupancy, DRAM throughput, and SM utilization; high memory utilization with low FP throughput confirms memory-bound behavior.

4. What causes GPU kernels to appear as many small green bars in Nsight Systems?

Answer:
That indicates many tiny kernel launches — usually due to unfused ops, small tensor sizes, or Python overhead between launches.
Fix by using operator fusion (Torch Compile / TensorRT) or CUDA Graphs to reduce launch overhead.

5. How can you detect CPU↔GPU synchronization overhead in PyTorch Profiler or Nsight Systems?

Answer:
You’ll see gaps in the GPU timeline where kernels aren’t executing, and CPU activities waiting on cudaMemcpy or implicit syncs.
Excessive cudaDeviceSynchronize() or .item() calls often cause this.
Use async data transfers (non_blocking=True) and pinned memory to minimize such stalls.

6. What does “achieved occupancy” mean in Nsight Compute, and why might it be low?

Answer:
Achieved occupancy = ratio of active warps per SM to the hardware maximum.
Low occupancy often results from high register pressure, shared memory overuse, or thread/block size mismatch — reducing the GPU’s ability to hide latency.
Tuning block size or reducing register usage can improve occupancy.

7. How do you profile a PyTorch model to generate a Chrome trace and interpret it?

Answer:
Use torch.profiler.profile(..., with_stack=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("traces")).
Then open the .json trace in Chrome → chrome://tracing.
You can inspect per-op CUDA launches, overlaps, and gaps — showing kernel concurrency and data transfer timing.

8. What are some key Nsight Compute metrics to check when analyzing a slow CUDA kernel?

Answer:

sm__throughput – shows ALU utilization.

dram__throughput – shows memory bandwidth usage.

warp_execution_efficiency – divergence within warps.

achieved_occupancy – how many threads are active.

gld_efficiency/gst_efficiency – memory coalescing efficiency.
Together, they reveal whether a kernel is bottlenecked by compute, memory access, or warp divergence.

9. In PyTorch Profiler, how do you differentiate between data loader bottlenecks and compute bottlenecks?

Answer:
Use record_shapes=True, profile_memory=True, with_stack=True.
If CPU time (especially under DataLoader or copy_) dominates before GPU kernels launch, the model is input-pipeline-bound.
Adding num_workers, prefetch_factor, or async data loading can fix it.

10. When profiling inference, how do you detect and minimize kernel launch overhead?

Answer:
Kernel launch overhead appears as short, frequent GPU kernels separated by CPU idle gaps in Nsight Systems.
To minimize:

Use fused operators (Torch Inductor, TensorRT).

Use CUDA Graph capture to remove per-launch overhead.

Avoid running small batch sizes where kernel utilization is poor.

## Attention & Sequence Optimizations

### FlashAttention v1/v2, tiling & online softmax
1. What fundamental problem does FlashAttention solve? </br>
Traditional attention explicitly computes and stores the full attention matrix QKT
, which has O(N2) memory and computational complexity for sequence length 
N. This means each token interacts with every other token, producing an 
N×N matrix that is both memory-heavy and slow to move through GPU DRAM.
Even if the arithmetic is fast, the operation is memory-bound, since every matrix element is read and written multiple times across stages (QK^T, softmax, and P·V). FlashAttention reorders this computation so that intermediate results never leave on-chip memory, reducing both the memory complexity to O(N) and DRAM traffic to the theoretical minimum

2. How does FlashAttention reduce memory and improve efficiency? </br>
Instead of computing the entire attention matrix at once, FlashAttention divides Q, K, and V into small tiles that fit entirely into shared memory or registers on the GPU. For each query tile 
Qi , it streams over corresponding K/V tiles (Kj,Vj), computing partial attention scores and accumulating the output on the fly.
By doing this, the algorithm reads each element of Q, K, and V once, and writes each element of the output once.
There is no intermediate global memory write for attention scores or softmax numerators. All intermediate values are kept on-chip, leading to a bandwidth-optimal design — i.e., the fewest possible memory transactions for exact attention.

3. How does FlashAttention achieve its speedup on GPUs? </br>
The speedup comes from fusing multiple operations — QK^T, scaling, masking, softmax, and the final P·V — into a single GPU kernel.
Because the kernel uses tiling, data loaded from global memory stays in shared memory during the entire sequence of operations, avoiding redundant DRAM reads.
Each tile’s matmul uses Tensor Cores for FP16/BF16 inputs, and warp-level reductions for max/sum operations.
By minimizing global memory traffic and maximizing arithmetic intensity (FLOPs per byte moved), the kernel becomes compute-efficient rather than memory-bound.
Additionally, since the fused kernel executes as one GPU launch, it eliminates kernel launch overhead and synchronization points between stages.

4. How is numerical stability ensured if only small tiles are processed at a time? </br>
FlashAttention maintains numerical stability by using the log-sum-exp trick at every tile step. Each new block’s contributions are normalized by the running max 
mi , ensuring that all exponentials are computed in a safe numeric range.
This method guarantees mathematically exact results — the same as standard softmax — up to floating-point rounding.
Even though computations are performed incrementally, the normalization logic ensures the final result is identical to that of the full, unstreamed attention.

6. What kind of performance improvement does FlashAttention provide? </br>
FlashAttention typically gives a 2× to 4× speedup compared to naïve PyTorch attention for long sequence lengths (1k tokens and above).
The memory footprint drops from O(N²) to O(N), which allows models to process much longer contexts — 16k to 32k tokens on GPUs like A100 or H100 that would otherwise run out of memory.
It’s not only faster but also enables previously infeasible sequence lengths, unlocking long-context transformers.

7. What are the limitations or trade-offs of FlashAttention?</br>
The algorithm assumes contiguous, regular attention patterns (full attention or simple masks). It’s not optimal for highly sparse or irregular patterns like local or block-sparse attention.
The backward pass is more complex because it must recompute or checkpoint intermediate statistics.
FlashAttention also introduces slight numerical differences when using fast exponentiation functions (e.g., CUDA’s __expf) or lower precision arithmetic (FP16/BF16).

8. How does FlashAttention relate to CUDA Graphs, Torch.compile, or Triton kernels?</br>
FlashAttention is implemented as a single custom fused kernel, often written in Triton or native CUDA.
Frameworks like PyTorch 2.0 (Inductor), TensorRT-LLM, and vLLM integrate it directly into their runtime.
When combined with CUDA Graph capture or Torch.compile, the fused FlashAttention kernel eliminates CPU–GPU synchronization and kernel launch overhead, further improving latency and throughput.
The idea aligns perfectly with NVIDIA’s push toward kernel fusion and graph-based execution for inference optimization.
9. How does FlashAttention interact with other inference optimizations like KV cache and paged attention? </br>
FlashAttention optimizes the compute pattern of attention itself (within one forward step). KV cache and paged attention, on the other hand, optimize temporal reuse across steps in autoregressive decoding.
Together they work synergistically:
FlashAttention reduces the compute and memory cost per step.
KV cache eliminates recomputation for previous tokens.
Paged attention handles variable-length and batched decoding efficiently.
When combined, they maximize both per-step throughput and long-sequence efficiency.

10. Why is FlashAttention considered “bandwidth-optimal”?</br>
A kernel is bandwidth-optimal if it performs the fewest possible DRAM reads and writes given its mathematical task.
In standard attention, each element may be read/written three or four times — far exceeding the theoretical minimum.
In FlashAttention, each element of Q, K, V, and the output is loaded or stored exactly once, with all intermediate computation done on-chip.
This makes it achieve near-maximum possible speed for the GPU’s available memory bandwidth, which is why it’s referred to as flash — it’s as fast as the hardware memory pipeline allows.

11. Derive O(N2d2M-1)

### Paged Attention 
1. How does inference proceed in an autoregressive LLM, and why is it computationally heavy?</br>
Large language models (LLMs) generate text token by token in an autoregressive fashion. During the prefill phase, the model processes the entire input prompt once to produce internal key–value (KV) representations for each layer. These KV tensors encode the context needed for attention.
During the decoding phase, each new token is generated sequentially: the model computes a new query vector for that token, retrieves all previous keys and values from the cache, performs attention between the new query and the stored keys, and then produces the next token.
This process is inherently sequential because each token depends on the one before it. Even though GPUs are massively parallel, only the computations within a single token’s forward pass are parallelizable. The repeated attention across the growing context makes inference increasingly expensive as sequence length increases.

2. Why is KV caching used in LLM inference, and how does it reduce compute and memory cost?</br>
Without caching, the model would recompute the key and value tensors for all past tokens every time it generates a new token — resulting in quadratic time complexity with respect to sequence length.
KV caching solves this by storing, for every transformer layer, the computed keys and values for all tokens that have already been processed. When generating a new token, the model only needs to compute the new token’s key and value once and append them to the cache. The attention operation then uses these cached tensors directly, avoiding redundant recomputation.
This converts the per-token compute complexity from O(n2) to O(n), since each decoding step now only attends to the cached history rather than re-encoding it. The trade-off is higher memory consumption (because all past K and V must stay in GPU memory), but in practice, caching is essential for high-throughput LLM serving.

3. How does batching improve compute utilization in LLM serving, and what challenges arise?</br>
GPU utilization during inference is often low because many requests arrive asynchronously and have variable prompt and output lengths. Batching improves utilization by processing multiple requests together, allowing the GPU to amortize the cost of loading model weights across all requests in the batch.
However, naive batching leads to inefficiencies: early requests may wait for later ones, and uneven sequence lengths force padding, wasting compute and memory. To address this, modern systems use fine-grained batching techniques such as cellular batching or iteration-level scheduling. These operate at the iteration level rather than the request level — after each decoding step, completed sequences are removed from the batch and new ones are added immediately. Specialized GPU kernels handle the resulting irregular shapes without padding.
This dynamic batching approach minimizes queuing delays, avoids wasted computation from padding, and substantially increases overall GPU throughput during real-time LLM serving.

4. What are the key memory challenges that make LLM serving memory-bound?</br>
The main bottleneck arises from the KV cache, which grows linearly with both the sequence length and the number of concurrent requests. Each token’s KV pair stores activations for all layers, and for large models (e.g., OPT-13B), this can reach ~1.6 GB per request.
Three factors amplify the memory problem:
Unknown input/output lengths → requests vary widely, so it’s impossible to pre-size memory precisely; output growth during decoding can exhaust GPU memory.
Large KV cache footprint → each token requires two tensors (K, V) × hidden size × #layers × datatype size. Even high-end GPUs can only host a few dozen concurrent sequences.
Complex decoding algorithms → sampling and beam search create divergent execution paths; some parts of KV can be shared (e.g., prompt phase) while others cannot, complicating allocation and reuse.
Together these make throughput memory-bound, not compute-bound, since the GPU runs out of memory before it saturates compute.

5. .How do current memory-management schemes waste GPU memory (internal, external fragmentation, reserved slots)?</br>
Existing frameworks allocate the KV cache for each request as a contiguous chunk sized to its maximum possible sequence length. This static pre-allocation leads to three sources of waste:
Reserved slots → space for future tokens is held for the entire request lifetime even if unused yet; it blocks new requests from using that memory.
Internal fragmentation → over-provisioning causes unused space inside each allocation, revealed only after decoding ends.
External fragmentation → dynamic allocators (e.g., buddy allocator) create small gaps between chunks that can’t be merged or reused efficiently.
In combination, these issues can reduce effective GPU memory utilization to as low as ≈ 20 %, severely limiting batch size and throughput.

6. What is the key idea behind PagedAttention and how does it differ from traditional attention?</br>
PagedAttention rethinks how the key–value (KV) cache is stored and accessed during inference. Traditional attention assumes that all KV tensors for a sequence are stored contiguously in GPU memory, which leads to severe memory fragmentation and over-allocation.
PagedAttention breaks the KV cache into fixed-size KV blocks (analogous to pages in an operating system). Each block stores the key and value vectors for a fixed number of tokens. During attention, the kernel dynamically fetches the relevant KV blocks — which can be non-contiguous in physical memory — and performs blockwise attention computation.
This design decouples logical sequence layout from physical memory layout, allowing much more flexible allocation and eliminating the need to pre-reserve large contiguous chunks.

7. How does vLLM manage KV memory and coordinate computation using PagedAttention?</br>
vLLM introduces a centralized scheduler that coordinates multiple distributed GPU workers. Each worker has a KV cache manager responsible for allocating and mapping the physical memory used by the PagedAttention kernel.
The KV cache manager organizes GPU memory into fixed-size physical KV blocks, which act like pages in a virtual memory system. For each request, it maintains a block table mapping the request’s logical KV blocks (token order) to physical KV blocks (actual memory locations).
As generation proceeds, vLLM allocates new physical blocks only when the previous ones are full, enabling dynamic growth of the cache without reserving memory for the maximum sequence length upfront. This eliminates the massive internal and external fragmentation seen in static allocation schemes.

8. How does PagedAttention operate during attention computation and decoding?</br>
During decoding, the PagedAttention kernel retrieves the necessary KV blocks for the current query token. For example, if the query corresponds to token i, it identifies which KV blocks contain the keys and values of the relevant context tokens and loads them in parallel.
Each block’s key matrix Kj is multiplied with the query qi to compute attention scores Aij , which are then multiplied by the corresponding value matrix 
Vj to form the partial output. The outputs from all blocks are summed to produce the final attention result.
By operating in block units, PagedAttention enables non-contiguous access to context tokens, parallelizes computation across multiple positions within a block, and allows hardware-efficient streaming of KV data.

9. How does vLLM handle dynamic decoding and multiple concurrent sequences efficiently?</br>
In vLLM, decoding proceeds incrementally, adding one new token per iteration across many sequences. When a new token’s KV cache is generated:
If the current logical block still has space, it is stored there, and the block table’s “filled positions” counter is updated.
If the block is full, a new physical block is allocated and linked to the next logical block entry.
Across multiple requests, their logical KV blocks are mapped to distinct physical blocks — these blocks do not need to be contiguous. This ensures high GPU memory utilization and enables many concurrent sequences to coexist in memory.
By limiting waste to at most one partially filled block per request and recycling blocks immediately after requests finish, vLLM achieves near-optimal memory efficiency and supports larger dynamic batches, directly improving throughput during LLM serving.

10. How does vLLM handle parallel sampling and memory sharing across multiple generated outputs for the same prompt?</br>
In parallel sampling, a single request produces multiple output sequences (samples) from the same input prompt — a common setup in program-assistant LLMs where users view several candidate completions. All these samples share the same prompt tokens, so their prompt-phase KV cache is identical and can be reused.
vLLM achieves this by mapping the logical KV blocks of each sample’s prompt to the same physical KV blocks in GPU memory. Each physical block has a reference count indicating how many sequences share it. When a sample begins generating its own unique continuation, vLLM applies a copy-on-write (CoW) mechanism at the block level:
If a shared block needs modification and its reference count > 1, vLLM allocates a new physical block, copies the shared data into it, and decrements the original block’s reference count.
If the block is already unique (reference = 1), it is modified in place.
This scheme allows multiple samples to share all prompt KV data while keeping separate copies only for the diverging generation phase. As a result, vLLM drastically reduces memory use for long prompts while maintaining independent sampling behavior for each output.

11. What scheduling policy does vLLM use when request traffic exceeds system capacity?</br>
vLLM employs a First-Come, First-Serve (FCFS) scheduling policy to ensure fairness and prevent starvation. When GPU memory or compute resources are saturated, the earliest arrived requests continue execution, while later ones are preempted. This policy guarantees that requests already in progress are prioritized and completed, avoiding partial starvation or unpredictable ordering. It also simplifies memory management since active sequences are treated as cohesive groups whose state must remain valid in GPU memory.

12. How does vLLM handle preemption when GPU memory runs out during decoding?</br>
When all available GPU KV blocks are occupied and new tokens require more memory, vLLM triggers preemption to reclaim space. Instead of evicting individual blocks randomly, it uses an all-or-nothing policy — either evicting all KV blocks belonging to a sequence or none at all. This avoids fragmenting logical sequences across GPU and CPU memory, since all blocks of a given sequence are accessed together during attention.
Moreover, in multi-sequence decoding (e.g., beam search or parallel sampling), vLLM gang-schedules the group: all sequences from the same request are preempted or resumed together, preserving internal memory sharing and synchronization.
What eviction strategies does vLLM use to reclaim GPU memory, and how do they work?
vLLM supports two complementary eviction and recovery strategies:
Swapping: Evicted KV blocks are copied from GPU memory to CPU RAM, managed by a CPU block allocator. When the sequence is resumed, its KV blocks are transferred back. The total swap space needed is bounded by the GPU’s KV cache allocation, ensuring controlled memory usage. During swapping, vLLM temporarily halts new request admissions until all preempted sequences complete.
Recomputation: Instead of copying data out, vLLM recomputes the KV cache for preempted sequences when they are resumed. This is often faster than full recomputation because the already-generated tokens can be treated as an extended prompt, allowing all required KV states to be rebuilt in one forward pass.

13. What determines whether swapping or recomputation is more efficient in practice?</br>
The relative performance of the two methods depends on system bandwidth and GPU compute capability. If the PCIe or NVLink bandwidth between CPU and GPU is high, swapping is preferred because it avoids recomputation overhead. However, when the GPU is compute-rich but memory-bandwidth-limited, recomputation can be faster — especially since it reuses the previous tokens as a single prompt and regenerates all necessary KV states in one batch.
Thus, vLLM can flexibly choose between the two approaches depending on deployment hardware and workload patterns.

14. How does vLLM handle distributed execution when model weights exceed single-GPU memory?</br>
For large LLMs whose weights cannot fit on one GPU, vLLM distributes computation across multiple GPU workers. Each worker stores a partition (shard) of the model weights and a portion of the KV cache. The centralized scheduler orchestrates execution so that all GPUs stay synchronized across layers and time steps.
PagedAttention still operates per worker — each manages its own physical KV blocks, and the logical-to-physical mapping is tracked globally. During forward passes, only the necessary blocks for each token are communicated across GPUs. This hybrid of tensor parallelism (for weights) and paged KV parallelism (for activations) allows vLLM to scale efficiently while maintaining the same paged memory abstraction as in the single-GPU case.
### Speculative Decoding
### MoE

### TensorRT
Q1. What is TensorRT and where does it fit in the inference pipeline?</br>
TensorRT is NVIDIA’s high-performance deep learning inference optimizer and runtime.
It takes a trained model (ONNX, TF, PyTorch export, etc.), applies graph-level optimizations (layer fusion, constant folding, kernel selection), optionally performs quantization (FP16/INT8), and builds a platform-specific engine plan that runs efficiently on NVIDIA GPUs.
It sits between model training and deployment — typically below ONNX-Runtime or Triton-Inference-Server — as the layer that performs hardware-specific optimization and execution.

Q2. What are the main optimization techniques TensorRT applies during engine building?</br>
Layer Fusion: Combines compatible layers (e.g., Conv + BN + ReLU) to reduce kernel launches and memory bandwidth.
Kernel Auto-Tuning: Chooses the fastest CUDA kernel implementation for each layer based on GPU architecture, tensor shape, and precision.
Precision Calibration / Quantization: Converts weights and activations to FP16 or INT8 while maintaining acceptable accuracy.
Memory Reuse and Tensor Layout Optimization: Reuses intermediate buffers to minimize memory footprint and arranges tensors in the most efficient layout (e.g., NHWC vs NCHW).
Dynamic Tensor Shapes: Allows optimized execution paths for variable input sizes.

Q3. What happens inside TensorRT when you run trtexec to build an engine?</br>
trtexec runs three major stages:
Network Parsing — reads the ONNX/other model into TensorRT’s internal graph.
Optimization & Calibration — applies graph transformations, layer fusion, precision calibration if INT8 mode is enabled, and kernel tuning for the target GPU (SM architecture).
Serialization — generates a binary engine (“plan”) containing optimized CUDA kernels, memory bindings, and runtime metadata.
During inference, this engine is deserialized into memory and executed by TensorRT’s execution context, which binds input/output buffers and launches kernels asynchronously.

Q4. How does TensorRT achieve lower latency compared to PyTorch/ONNX Runtime?</br>
Graph Compilation eliminates the Python interpreter, dynamic control flow, and kernel dispatch overhead.
Fused Kernels and optimized CUDA streams reduce launch overhead.
Reduced precision (FP16/INT8) doubles or quadruples effective throughput per SM.
Static memory planning ensures deterministic execution without runtime allocations.
Asynchronous execution contexts allow overlapping compute and I/O across streams.
Result: lower latency and higher throughput per watt compared to high-level frameworks.

Q5. What are TensorRT Execution Contexts and Optimization Profiles?</br>
Execution Context: A lightweight runtime object derived from an engine that holds activation memory, bindings, and stream state.
You can create multiple contexts per engine to serve concurrent inferences on the same GPU.
Optimization Profile: Defines input dimension ranges for dynamic shape support (min, opt, max).
TensorRT builds optimized kernels for these shapes. At runtime, the context selects the closest matching profile based on the current input size.

### TensorRT-LLM
Q1. What is TensorRT-LLM and how does it differ from standard TensorRT?</br>
TensorRT-LLM extends TensorRT to large language model (LLM) workloads.
While TensorRT optimizes general DNNs (CNNs, transformers, etc.) as static computation graphs, TensorRT-LLM adds features required for autoregressive decoding — like KV cache management, paged attention, dynamic sequence batching, and multi-GPU execution.
It builds on TensorRT’s kernel fusion and quantization but adds specialized fused transformer kernels, support for multi-token and multi-sequence decoding, and integration with vLLM-like scheduling for high GPU utilization.

Q2. How does TensorRT-LLM improve throughput for autoregressive decoding?</br>
In autoregressive generation, every new token depends on all previous tokens, so naïve implementations recompute large parts of attention.
TensorRT-LLM avoids this by:
Caching KV tensors across decoding steps (key-value cache).
Using Paged Attention to store cache blocks non-contiguously, minimizing fragmentation and enabling fine-grained memory reuse.
Implementing fused multi-token attention kernels that can process several new tokens per batch in one kernel launch.
Supporting continuous batching so that new sequences can join mid-generation without stopping existing ones.
These reduce launch overhead and idle SM time, yielding higher throughput per GPU.

Q3. What quantization and precision modes does TensorRT-LLM support?</br>
TensorRT-LLm supports:
FP16, BF16 — high-accuracy reduced precision.
INT8 and FP8 — for aggressive optimization.
FP8 is particularly important in H100 GPUs; TensorRT-LLM provides fused FP8 kernels with per-channel scaling to preserve accuracy.
It also supports mixed-precision accumulation (e.g., FP16 compute, FP32 accumulation) and weight-only quantization to minimize memory bandwidth use — crucial for large models.

Q4. How does TensorRT-LLM manage dynamic batching and paged KV caches at runtime?</br>
At runtime, TensorRT-LLM maintains a scheduler that:
Groups requests with similar sequence lengths into shared batches.
Uses paged KV caching, where each sequence’s cache is stored in small blocks that can be dynamically allocated, reused, or evicted.
This design enables fine-grained preemption and multi-tenant inference — new sequences can be inserted or removed between iterations without fully rebuilding batches, allowing near-100% GPU utilization even with variable-length prompts.

Q5. How does TensorRT-LLM scale across multiple GPUs or nodes?</br>
TensorRT-LLM supports tensor parallelism, pipeline parallelism, and multi-context execution:
Tensor Parallelism splits model layers (e.g., attention heads, feed-forward matrices) across GPUs.
Pipeline Parallelism assigns consecutive layers to different GPUs
Each GPU runs an optimized TensorRT engine for its shard, with efficient inter-GPU communication via NCCL / NVLink.
The runtime hides data transfer latency by overlapping compute and communication, enabling efficient scaling for large models that exceed single-GPU memory.

### CUDA
1. Grid Block Threads
2. GPU computation heirarchy
3. Warp
4. Thread Divergence
5. CUDA Memory model
6. bandwidth vs latency, latency hiding
7. Locality - spatial temporal
8. Memory coalescing, Degree of Coalescing, AoS,SoA
9. Shared Memory, syncthreads(), dynamic shared memory
10. Bank Conflict
11. Synchronization, Data Race, Atomics, Barriers, Reductions
12. CPU GPU synchronization, Pinned Memory
13. Functions
14. CUDA Profiler nvprof
15. Streams



TensorRT & TensorRT-LLM

Engine building: precision, tactics, calibrators

Layer/tactic selection, tactic replay

Dynamic shapes, optimization profiles

Plugins (custom ops), fused attention kernels

TensorRT-LLM: paged KV cache, rope, FP8/INT8 paths

vLLM / SGLang (Serving LLM/VLM)

PagedAttention, memory planning

Continuous batching scheduler

Token throughput vs latency trade-offs

Triton Inference Server (Serving)

Model Repository, config.pbtxt

Instance Groups, Dynamic Batching, Ensembles

Concurrency, rate limits, timeouts

Health, metrics, NVTX/NVML/Prometheus

Multi-model, multi-GPU placement

Memory, I/O & Overlap

Pinned memory, async H2D/D2H, streams

Overlap copy/compute, multiple streams

PCIe vs NVLink bandwidth; GPUDirect RDMA

NUMA pinning, CPU affinity

Model Compression & Adaptation

Pruning (structural/unstructured)

Distillation (teacher→student for inference)

LoRA/QLoRA at inference (merging/adapters)

Low-rank KV cache

Tensor Layouts & Formats

NHWC vs NCHW; channels-last

FP8 (E4M3/E5M2) considerations

Packed INT formats (INT8/INT4)

Calibration datasets for PTQ

Numerical drift checks across precisions

Determinism & seeds in inference
