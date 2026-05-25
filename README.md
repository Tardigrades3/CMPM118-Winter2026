## To Download All NinaPro Data
Run:
```bash
    python download_ninapro.py
```


---

## 🚀 Running the Training Pipeline

The training pipeline is controlled by a central mission control script: `build_ss.py`. This script handles the data routing, model initialization, and execution of various Continual Learning (CL) methodologies.

### Prerequisites

Before running the scripts, ensure you have the hardware-accelerated linear attention library installed:

```bash
pip install flash-linear-attention

```

### Basic Usage

The training script requires two mandatory arguments: the training `--mode` and the `--data_path`.

```bash
python build_ss.py --mode <TRAINING_MODE> --data_path /path/to/NinaPro

```

> **Important:** The `--data_path` must point to the parent directory containing the individual subject folders (e.g., `s1/`, `s2/`, etc.).

---

### Command-Line Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--mode` | String | **Required** | The training paradigm to execute (see modes below). |
| `--data_path` | String | **Required** | Absolute or relative path to the root NinaPro dataset. |
| `--exercise` | Int | `1` | The NinaPro exercise number to process. |
| `--batch_size` | Int | `32` | Number of sequences per batch. |
| `--epochs_per_task` | Int | `5` | Number of epochs to train on each subject before moving on. |
| `--lr` | Float | `1e-4` | Learning rate for the AdamW optimizer. |

---

### Training Modes & Examples

The pipeline supports five distinct training paradigms, ranging from standard deep learning baselines to advanced Continual Learning defenses.

#### 1. Standard Baseline (`stateless`)

Wipes the hidden state between sequences and shuffles the data. Treats the dataset like a standard, non-continuous classification task.

```bash
python build_ss.py --mode stateless --data_path ../data/NinaPro

```

#### 2. Naive Continual Learning Baseline (`stateful`)

Maintains a continuous, unbroken hidden state across batches. Processes data in strict chronological order per subject. Suffers from Catastrophic Forgetting (used as the lower-bound baseline).

```bash
python build_ss.py --mode stateful --data_path ../data/NinaPro

```

#### 3. Experience Replay Baseline (`replay_stateless`)

Standard shuffled training, but utilizes a reservoir sampling buffer to randomly inject historical subject data into the current batches.

```bash
python build_ss.py --mode replay_stateless --data_path ../data/NinaPro

```

#### 4. Stateful Experience Replay (`replay_stateful`)

**Advanced CL.** Uses a two-pass forward method. It maintains an unbroken hidden state for the current subject's time-series, while running a simultaneous stateless pass on historical memory buffers to prevent forgetting.

```bash
python build_ss.py --mode replay_stateful --data_path ../data/NinaPro

```

#### 5. Elastic Weight Consolidation (`ewc_stateful`)

**Advanced CL.** A regularization method requiring no historical data storage. Calculates the Fisher Information Matrix at the end of each subject to mathematically penalize the network for altering previously critical neural pathways.

```bash
python build_ss.py --mode ewc_stateful --data_path ../data/NinaPro

```

---

### Customizing Hyperparameters

You can easily override the default hyperparameters for specific experiments. For example, to run EWC on Exercise 2 with a larger batch size and a higher learning rate:

```bash
python build_ss.py \
  --mode ewc_stateful \
  --data_path ../data/NinaPro \
  --exercise 2 \
  --batch_size 64 \
  --epochs_per_task 10 \
  --lr 5e-4

```