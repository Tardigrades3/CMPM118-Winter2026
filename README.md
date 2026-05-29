# FastHGRN Continual Learning Framework

This repository contains the training and evaluation pipeline for testing the FastHGRN state-space model on continuous EMG bionic signals. It supports both Domain-Incremental Learning (DIL) across multiple subjects and Class-Incremental Learning (CIL) across multiple gesture sets.

---

## Project Structure

* `build_ss.py`: The primary entry point for all training experiments and the macro continual learning loop.
* `fastHGRN.py`: Contains the core state-space model architecture and feature extraction logic.
* `preprocessing.py` & `ss_preprocessing.py`: Handles raw `.mat` extraction, lowpass and notch filtering, sliding window generation, and PyTorch DataLoader packaging.
* `training_functions.py`: Contains the specific training loops for each continual learning strategy (stateless, stateful, replay, EWC).
* `evaluation_functions.py`: Calculates final metrics and formats the output JSON.
* `plot_bwt.py`: Generates publication-ready Backward Transfer matrices for visualizing knowledge retention.

---

## Execution Guide and Commands

The pipeline is controlled via the command line. You must specify the scenario, the training mode, and the path to the extracted NinaPro dataset.

| Experiment Target | Training Mode | Command |
| --- | --- | --- |
| **DIL Baseline (Forgetting)** | Stateful | `python build_ss.py --scenario dil --mode stateful --data_path ../NinaProData/ --exercise 1` |
| **DIL with Herding Replay** | Herding Stateful | `python build_ss.py --scenario dil --mode herding_stateful --data_path ../NinaProData/ --exercise 1` |
| **CIL Baseline (Forgetting)** | Stateful | `python build_ss.py --scenario cil --mode stateful --data_path ../NinaProData/ --subject 1` |
| **CIL with Herding Replay** | Herding Stateful | `python build_ss.py --scenario cil --mode herding_stateful --data_path ../NinaProData/ --subject 1` |
| **CIL with EWC** | EWC Stateful | `python build_ss.py --scenario cil --mode ewc_stateful --data_path ../NinaProData/ --subject 1` |

---

## Configurable Hyperparameters

### Command Line Arguments

These parameters can be adjusted rapidly when executing `build_ss.py` to test different broad training dynamics:

* `--batch_size`: Default is 32. Lower this to 16 or 8 if you encounter VRAM constraints during CIL training.
* `--epochs_per_task`: Default is 5.
* `--lr`: Default learning rate is 1e-4.

### Hardcoded Architectural Parameters

To fine-tune the neuromorphic aspects of the model and memory buffers, you will need to modify specific variables directly within the Python files:

* **Herding Buffer Capacity** (`build_ss.py`): Search for `capacity_per_class=20`. This dictates exactly how many prototypical sequences are saved per gesture.
* **Replay Loss Weight** (`training_functions.py`): Search for `replay_weight = 0.5` inside the `train_replay_stateful` function. Adjust this scalar to balance the retention of old tasks against the adaptation to new ones.
* **Replay Jitter** (`training_functions.py`): Search for `noise = torch.randn_like(replay_seqs) * 0.01`. This Gaussian noise injection prevents the model from overfitting to the static herded exemplars.
* **Signal Windowing** (`preprocessing.py`): Inside the extraction functions, the sliding window parameters are explicitly set to `seq_len=200` and `overlap=100`.
* **EWC Lambda** (`build_ss.py`): Search for `ewc_lambda=2000` in the training loop. This controls the rigidity of the Fisher Information Matrix penalty.
* **Model Dimensions** (`build_ss.py`): The FastHGRN initialization specifies `d_model=128` and `num_layers=4`.

---

## Viewing Results

### 1. Model Weights

Every run generates a timestamped directory structured as `weights_[scenario]_[mode]_[timestamp]`. After every task completes, the model and optimizer states are saved as `.pt` files. These can be loaded later for manual inference.

### 2. Raw Metrics (JSON)

Upon completing the final task evaluation, a comprehensive JSON file is saved in the working directory. This file contains the loss curves, immediate post-task accuracies, and final accuracies for the entire training sequence. The terminal will print the exact path to this file upon completion.

### 3. Visualizing Backward Transfer (BWT)

To generate a visual representation of catastrophic forgetting or knowledge retention, pass the generated JSON file to the plotting script:

`python plot_bwt.py path/to/your/evaluation_file.json`

This will generate and save a `.png` heatmap in the same directory as the JSON file. It is color-coded to show accuracy degradation (red bars) or positive transfer (green bars) for each individual task, providing clear proof of the memory buffer's efficacy.