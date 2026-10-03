# AdaptGreen: Python Simulation Prototype

A Python simulation prototype strictly reproducing the algorithmic architecture, decision logic, and evaluation methodologies of the research paper:

> **"AdaptGreen: A Context-Adaptive Runtime Framework for Energy-Efficient Mobile AI Inference"**

---

## 1. What AdaptGreen Is

**AdaptGreen** is an energy-aware, context-adaptive runtime framework designed to optimize the execution of deep learning models on resource-constrained edge and mobile devices. Rather than executing fixed neural network architectures regardless of device state, AdaptGreen continuously senses environmental telemetry (battery level, SoC temperature, available RAM, network signal, and user interactivity) and dynamically adapts the inference execution tier.

The framework balances three competing objectives:
1. **Energy Conservation**: Minimizing battery drain during high-strain or background states.
2. **Thermal Stability**: Preventing SoC thermal throttling by avoiding compute-heavy models when temperatures approach hardware thresholds.
3. **Semantic Quality & Low Latency**: Providing full-precision neural representations and fast response times when resources are abundant and the user is actively engaged.

---

## 2. What Is Implemented in this Prototype

This repository implements the complete Python simulation prototype designed to validate the core framework ahead of physical mobile deployment:

- **AI Workload**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional sentence embedding model).
- **Complete 5-Tier Model Library**:
  - **Tier 1 (T1)**: Full Precision FP32 PyTorch neural inference (45.0 mJ, 120 MB RAM, 1.0 golden fidelity).
  - **Tier 2 (T2)**: L2 Structured Pruned PyTorch model targeting intermediate MLP layers (30.0 mJ, 85 MB RAM, >0.97 fidelity).
  - **Tier 3 (T3)**: Dynamic INT8 quantized PyTorch model (quantizing 37 linear layers in the transformer attention and feed-forward sub-modules, 15.0 mJ, 50 MB RAM, 0.995 fidelity).
  - **Tier 4 (T4)**: Cloud Offload REST Client modeling Wi-Fi transmission latency and cellular baseband power (8.0 mJ, 15 MB RAM, with C5 RSSI cutoff at -85 dBm).
  - **Tier 5 (T5)**: In-memory vector embedding cache with sub-millisecond retrieval (1.0 mJ, 5 MB RAM).
- **Multi-Device Class Configuration Profiles**:
  - **Flagship**: 12GB RAM, 400 MB headroom floor, 82°C thermal ceiling (`configs/devices/flagship.json`).
  - **Mid-Range**: 6GB RAM, 600 MB headroom floor, 80°C thermal ceiling (`configs/devices/mid_range.json`).
  - **Budget**: 4GB RAM, 800 MB headroom floor, 78°C thermal ceiling (`configs/devices/budget.json`).
- **Academic Paper Recommender Dataset & Metrics**:
  - 200 academic paper abstract corpus spanning Edge AI, Green Computing, Model Compression, and Transformers (`data/paper_abstracts.json`).
  - Pre-encoded 384-dimensional corpus reference embeddings (`data/corpus_embeddings.npy`).
  - Downstream Top-5 ranking overlap accuracy metric measuring end-to-end task fidelity relative to the FP32 reference.
- **Context Monitoring Engine (CME)**: Simulates 6D device state trajectories $s(t) = (\text{battery}, \text{temperature}, \text{RSSI}, \text{user\_activity}, \text{task\_type}, \text{available\_RAM})$.
- **Policy Decision Unit (PDU)**:
  - **Stage 1 (Feasibility Filter)**: Enforces hard constraints $C_1$ (latency), $C_2$ (accuracy), $C_3$ (thermal), $C_4$ (battery), $C_5$ (network), and $C_6$ (memory) across all 5 candidate tiers.
  - **Stage 2 (Offline Q-Learning)**: Tabular Q-learning agent mapping discretized 6D states across 5 actions, serialized to `pdu/policy_table.json`.
  - **Hysteresis Cooldown (4 epochs)**: Enforces switching stability to eliminate ping-pong overhead, with emergency override for catastrophic states.
- **Runtime Inference Executor**: Dispatches text queries to the selected tier, measures real CPU execution latency, and assigns reproducible simulated energy values.
- **Comparative Static Baselines**: Implements Static-T1 (always FP32) and Static-T3 (always INT8) over identical query traces.
- **Evaluation & Telemetry**: Generates epoch-by-epoch CSV logs (`foreground_active_results.csv`, `battery_critical_results.csv`, `thermal_stress_results.csv`, `background_batch_results.csv`, `all_results.csv`).
- **Publication-Quality Visualization**: Automatically renders 9 standalone figures (300 DPI) using standard Matplotlib (no seaborn).
- **Automated Reporting**: Outputs comprehensive summary reports in `results/final_report.json` and `results/final_report.txt`.
- **Rigorous Test Suite**: 28 automated unit tests covering all tiers, constraints, device presets, Q-learning updates, and recommendation metrics.

---

## 3. Difference Between the Paper's Proposed Android Architecture and this Python Simulation

| Aspect | Paper's Conceptual Android Prototype | This Python Simulation Prototype |
| :--- | :--- | :--- |
| **Operating System** | Android 12+ (AOSP / Linux Kernel) | Cross-platform Python 3.10+ (Windows/Linux/macOS) |
| **Telemetry Sensing** | Android `BatteryManager`, `ThermalManager`, `WifiManager`, `ActivityManager` | Deterministic mathematical simulation curves via `ContextMonitoringEngine` |
| **Polling Rate** | 500 ms periodic broadcast listeners | 1 decision step per simulation epoch |
| **Power Measurement** | External hardware power monitor (e.g. Monsoon High Voltage Power Monitor) | Configured reproducible tier energy profiles ($T_1=45\,\text{mJ}, T_3=15\,\text{mJ}, T_5=1\,\text{mJ}$) |
| **Model Runtime** | ONNX Runtime Mobile / PyTorch Mobile Android AAR | PyTorch CPU runtime + dynamic quantization engine |
| **Primary Purpose** | Real-world validation on physical mobile hardware | Algorithmic validation of policy convergence, hysteresis stability, and energy conservation |

---

## 4. How to Install

### Prerequisites
- Python 3.10 or Python 3.11
- Microsoft Visual C++ 2015-2022 Redistributable (on Windows)

### Installation Commands

```bash
# 1. Clone or navigate to the repository
cd adaptgreen

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
```

*Note: For CPU-only PyTorch on Windows/Linux:*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers numpy pandas matplotlib scikit-learn
```

---

## 5. How to Run

Execute the entire simulation and evaluation pipeline with a single command:

```bash
python main.py
```

This single command will:
1. Load or train the offline Q-learning policy (`pdu/policy_table.json`).
2. Download/cache the `sentence-transformers/all-MiniLM-L6-v2` model weights.
3. Quantize the model dynamically to INT8.
4. Execute Scenario 1 (`FOREGROUND_ACTIVE`) for 60 epochs.
5. Execute Scenario 2 (`BACKGROUND_BATCH`) for 60 epochs.
6. Execute Scenario 3 (`THERMAL_STRESS`) for 60 epochs.
7. Execute Scenario 4 (`BATTERY_CRITICAL`) for 60 epochs.
8. Run Static-T1 and Static-T3 baselines over identical query inputs.
9. Save CSV telemetry files under `results/`.
10. Generate 9 publication figures under `results/figures/`.
11. Generate `results/final_report.json` and `results/final_report.txt`.
12. Print the live epoch log and terminal summary comparison table.

---

## 6. How Tier 1 (T1) Works — Full FP32

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` loaded in native 32-bit floating point precision (`torch.float32`).
- **Architecture**: 6 transformer encoder layers, 384 hidden dimension, 12 attention heads, 22.7 million parameters.
- **Inference**: Tokenizes input query text, passes tensors through the transformer encoder, performs mean pooling over token embeddings weighted by attention mask, and normalizes the vector to unit L2 norm.
- **Characteristics**:
  - Highest representation fidelity (golden reference = 1.0).
  - Highest CPU computational load and highest thermal dissipation.
  - Simulated energy: **45.0 mJ** per inference.

---

## 7. How Tier 3 (T3) Works — Dynamic INT8

- **Quantization Engine**: Implemented via PyTorch dynamic quantization:
  ```python
  torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
  ```
- **Quantized Modules**:
  Quantizes all 37 `torch.nn.Linear` layers across:
  - `BertSelfAttention`: `query`, `key`, `value` projections ($384 \times 384$).
  - `BertSelfOutput`: `dense` layer ($384 \times 384$).
  - `BertIntermediate`: `dense` projection ($384 \times 1536$).
  - `BertOutput`: feed-forward `dense` layer ($1536 \times 384$).
  - `BertPooler`: `dense` layer ($384 \times 384$).
- **Characteristics**:
  - Converts 32-bit floating point matrix weights to 8-bit signed integers (`qint8`).
  - Activations remain float32 and are dynamically quantized during matrix multiplication.
  - Reduces memory bandwidth and accelerates CPU vector execution.
  - Semantic fidelity: **> 0.995** cosine similarity relative to FP32.
  - Simulated energy: **15.0 mJ** per inference (66.7% energy reduction vs. T1).

---

## 8. How Tier 5 (T5) Works — In-Memory Vector Cache

- **Mechanism**: In-memory dictionary cache mapping normalized query strings to previously computed 384-dimensional embedding vectors:
  ```python
  cache[normalized_text] = embedding_vector
  ```
- **Cache Hit**: Returns the exact pre-computed vector with sub-millisecond memory access latency ($< 0.1\,\text{ms}$) and near-zero CPU computation.
- **Cache Miss**: T5 is unavailable for that request. The runtime executes the valid T3 fallback, charges T3 energy, records `selected_tier=T5`, `executed_tier=T3`, and records the miss before storing the new result in the cache.
- **Characteristics**:
  - Ultra-low simulated energy: **1.0 mJ** per inference (97.8% reduction vs. T1).
  - Perfect fidelity for identical recurring queries.

---

## 9. How CME Works — Context Monitoring Engine

The CME simulates the 6-dimensional mobile context vector:
$$s(t) = (\text{battery}, \text{temperature}, \text{RSSI}, \text{user\_activity}, \text{task\_type}, \text{available\_RAM})$$

Where:
- `battery`: Battery level percentage $[0.0, 100.0]\,\%$.
- `temperature`: SoC temperature $[20.0, 100.0]\,^\circ\text{C}$.
- `rssi`: Signal strength $[-120.0, 0.0]\,\text{dBm}$.
- `user_activity`: $1$ if foreground user interaction is active, $0$ if background.
- `task_type`: Workload classification ($1 = \text{interactive query}, 2 = \text{background sync}, 3 = \text{batch analytics}$).
- `available_ram`: Free physical RAM in Megabytes (MB).

---

## 10. How PDU Works — Policy Decision Unit

The PDU operates in two sequential stages:

### Stage 1: Feasibility Filter
Evaluates constraints $C_1$ through $C_6$:
- **$C_3$ Thermal Constraint**: If $\text{temperature} \ge 80.0^\circ\text{C}$, Tier 1 is marked **infeasible** to prevent thermal runaway.
- **$C_4$ Battery Constraint**: If $\text{battery} \le 20.0\%$, Tier 1 is restricted. If $\text{battery} \le 10.0\%$, Tier 3 is restricted to preserve charge.
- **$C_6$ Memory Constraint**: If $\text{available\_RAM} < 600\,\text{MB}$, heavy memory allocations (T1) are restricted.
- **$C_1$ Latency Constraint**: Evaluates projected execution delays under throttling.
- **$C_2$ Accuracy Constraint**: Enforces that candidate tiers satisfy minimum cosine fidelity.
- **$C_5$ Network Viability**: Monitors network RSSI.

### Stage 2: Q-Table Policy Query
Queries the pre-trained Q-table for the highest Q-value action among the **feasible tiers** identified in Stage 1:
$$a^* = \arg\max_{a \in \mathcal{A}_{\text{feasible}}} Q(s, a)$$

---

## 11. How Q-Learning Works

- **Offline Training**: The policy table is trained offline over thousands of simulated state episodes and serialized to `pdu/policy_table.json`. **No online reinforcement learning exploration is executed during runtime inference.**
- **State Discretization**: The continuous 6D state is mapped to discrete bins:
  - Battery: 4 bins $[0-20, 20-50, 50-80, 80-100]\,\%$
  - Temperature: 4 bins $[0-45, 45-65, 65-80, 80-100]\,^\circ\text{C}$
  - RSSI: 3 bins $[-120 \text{ to } -85, -85 \text{ to } -65, -65 \text{ to } 0]\,\text{dBm}$
  - User Activity: 2 values $\{0, 1\}$
  - Task Type: 3 values $\{1, 2, 3\}$
  - RAM: 3 bins $[0-600, 600-1200, 1200+]\,\text{MB}$
- **Reward Function**:
  $$R(s, a) = - w_e \cdot E_{\text{norm}}(a) - w_v \cdot \text{Violation}(s, a) - w_l \cdot \text{LatencyPenalty}(s, a) + w_a \cdot \text{Accuracy}(a)$$
- **Bellman Equation**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a' \in \mathcal{A}_{\text{feasible}}} Q(s', a') - Q(s, a) \right]$$

---

## 12. How Hysteresis Works

To eliminate high-frequency tier oscillation ("ping-ponging"):
- **Cooldown Window**: After switching tiers, a cooldown timer of **4 epochs** is enforced.
- **During Cooldown**:
  $$\text{selected\_tier} = \text{previous\_tier}$$
- **Exceptions to Cooldown**:
  1. **Feasibility Violation**: If the currently active tier becomes infeasible (e.g. temperature hits $80.5^\circ\text{C}$, invalidating T1 under $C_3$), the system switches immediately to the highest feasible alternative.
  2. **Catastrophic Emergency**: If catastrophic states occur (e.g. temperature $\ge 85^\circ\text{C}$ or battery $\le 10\%$), the emergency fallback breaks cooldown and drops directly to T5.

---

## 13. How Scenarios Work

### Scenario 1: `FOREGROUND_ACTIVE` (60 Epochs)
- **Environment**: High battery (98% $\rightarrow$ 86%), moderate temperature (40°C $\rightarrow$ 54°C, well below 80°C), abundant RAM (2200 MB $\rightarrow$ 2000 MB), active foreground interaction (`user_activity = 1`, `task_type = 1`).
- **Expected Outcome**: Resources remain abundant throughout; Stage 1 keeps all tiers feasible; Q-table selects T1 for maximum interactive fidelity. **T1 remains active for all 60 epochs.**

### Scenario 2: `BACKGROUND_BATCH` (60 Epochs)
- **Environment**: Display off, periodic background sync (`user_activity = 0`, `task_type = 2`), cool temperature (36°C $\rightarrow$ 43°C), gradual battery drain (92% $\rightarrow$ 82%), healthy RAM (1850 MB $\rightarrow$ 1750 MB).
- **Expected Outcome**: Relaxes $C_1$ latency ceiling, allowing PDU to prioritize energy minimization. Policy shifts entirely to ultra-low energy caching (T5), achieving **97.8% energy reduction vs. Static T1**.

### Scenario 3: `THERMAL_STRESS` (60 Epochs)
- **Environment**: Foreground active inference under sustained continuous compute (`user_activity = 1`, `task_type = 1`), temperature starts warm (~60°C) and rises rapidly past the 80.0°C $C_3$ thermal boundary up to 83.2°C.
- **Expected Outcome**: Starts on T1 for initial epochs; as temperature mounts and violates $C_3$, the PDU downshifts to INT8 (T3) and maintains stability under hysteresis cooldown (13 epochs T1, 47 epochs T3), avoiding ping-pong oscillation.

### Scenario 4: `BATTERY_CRITICAL` (60 Epochs)
- **Environment**: Rapid battery drop (75% $\rightarrow$ 15%), elevated temperature approaching 80.0°C, and application moves to background (`user_activity = 0`) as battery enters critical range.
- **Expected Outcome**: Dynamic multi-tier adaptation: transitions to T3 under resource constraints, then engages T5 cache during critical battery reserve, delivering **74.4% energy savings vs. Static T1**.

---

## 14. How Results Are Calculated

1. **Energy Conservation Percentage**:
   $$\text{Saving}_{\text{baseline}} = \frac{E_{\text{baseline}} - E_{\text{AdaptGreen}}}{E_{\text{baseline}}} \times 100\%$$
2. **Embedding Semantic Fidelity (Cosine Similarity)**:
   $$\text{CosineSim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
   *Note: Cosine similarity measures directional alignment in latent vector space relative to FP32 reference inference. It does NOT represent categorical classification accuracy.*
3. **Cache Hit Rate**:
   $$\text{HitRate} = \frac{\sum \text{Cache Hits}}{\text{Total Epochs}} \times 100\%$$

---

## 15. What Is Simulated vs. What Requires Real Android Hardware

### SIMULATED IN THIS PROTOTYPE:
- Battery level depletion curves.
- System-on-Chip (SoC) temperature rises and thermal dissipation.
- Available physical RAM memory pressure.
- Network RSSI fluctuations.
- Energy consumption per tier (using paper's empirical values: 45 mJ for T1, 15 mJ for T3, 1 mJ for T5).
- User interaction transitions (foreground interactive $\rightarrow$ background sync).

### ACTUAL RUNNING SOFTWARE:
- Real `sentence-transformers/all-MiniLM-L6-v2` loading and execution.
- Real 32-bit floating point PyTorch CPU forward pass (T1).
- Real PyTorch Dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic`) across 37 Linear layers (T3).
- Real in-memory dictionary vector embedding cache (T5).
- Real wall-clock execution latency measurement via `time.perf_counter()`.
- Real Policy Decision Unit, Stage 1 Feasibility Filter, and Stage 2 Q-table policy.
- Real 4-epoch Hysteresis cooldown logic.
- Real cosine similarity math, telemetry dataframes, and 300-DPI Matplotlib figure generation.

### REAL HARDWARE NOT REQUIRED (AND NOT CLAIMED):
- Physical Android smartphone.
- Android system service daemons (`BatteryManager`, `ThermalManager`, `WifiManager`).
- External hardware power measurement tools (Monsoon Power Monitor).
- Physical battery electrochemical wear.

---

## 16. Reproducing Graphs & Changing Thresholds

All thresholds are centralized in `adaptgreen/config.py`:

```python
# To modify thermal threshold:
CONFIG.constraints.thermal_max = 75.0  # Default: 80.0 °C

# To modify battery threshold:
CONFIG.constraints.battery_min = 25.0  # Default: 20.0 %

# To modify RAM threshold:
CONFIG.constraints.ram_min = 800.0     # Default: 600.0 MB

# To modify hysteresis cooldown epochs:
CONFIG.hysteresis.cooldown_epochs = 6  # Default: 4 epochs
```

After modifying settings, re-run:
```bash
python main.py
```
All 9 figures in `results/figures/` will automatically update with the new configurations.
