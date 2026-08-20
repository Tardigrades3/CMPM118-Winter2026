"""
Device-independent and device-dependent compute-cost profiling for HGRN/LSTM.

Runs on synthetic, shape-matched tensors (no NinaPro dataset needed), so it can
run identically on a laptop or on the EC2 box. Hyperparameter defaults mirror
hgrn/train.py's CLI defaults so numbers are directly comparable to real runs.

Every quantity this script records is numbered 1-14 and described in
METRIC_CATALOG below. The same numbering is written into the output JSON
(`metric_legend`) and stamped onto every value (`metric_id`), so the JSON is
self-documenting without needing to read this file.

Metrics 1-7 are device-independent (analytic: parameter/tensor-shape counts,
same value on any machine). Metrics 8-12 are device-dependent (measured wall
clock / memory on whatever machine runs the script; always read them next to
metric 13, the device_info block, which records what hardware produced them).
Metric 14 is a derived heuristic estimate, not a direct measurement.

Usage:
    python analysis/profile_compute.py --arch both --device auto
    python analysis/profile_compute.py --arch hgrn --device cpu --n_repeats 200
    python analysis/profile_compute.py --arch lstm --device cuda --output results_compute_profile/lstm_ec2.json
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from hgrn.model import HGRNModel, _HAS_FLA
from models.lstm import LSTMModel
from hgrn.buffers import HerdingBuffer

_ARCH_REGISTRY = {
    'hgrn': HGRNModel,
    'lstm': LSTMModel,
}

# ---------------------------------------------------------------------------
# 1-14: the numbered catalog of every quantity this script records.
# (id, key, category, unit, description)
# ---------------------------------------------------------------------------
METRIC_CATALOG = [
    (1, "param_count_total", "device_independent", "count",
     "Total model parameters (sum of p.numel() over all model.parameters())."),
    (2, "param_count_by_component", "device_independent", "count",
     "Parameter count broken down by top-level module (input_proj, layers, head)."),
    (3, "model_size_bytes", "device_independent", "bytes",
     "Storage footprint of param_count_total at fp32 (4B), fp16 (2B), and int8 (1B) per param."),
    (4, "flops_per_timestep_marginal", "device_independent", "FLOPs",
     "Marginal MACs*2 added per extra timestep of recurrence, i.e. (window_macs - "
     "single_step_macs) / (seq_len - 1). Excludes the classification head, which "
     "fires once per window regardless of length. Counted via forward hooks on "
     "every nn.Linear/nn.LSTM sub-module from actual tensor shapes (not a "
     "hand-derived per-architecture formula), so it stays correct if d_model/"
     "num_layers change. Elementwise ops (sigmoid, SiLU, LayerNorm) are excluded "
     "as asymptotically negligible next to the matmuls."),
    (5, "flops_per_window", "device_independent", "FLOPs",
     "MACs*2 for one full forward pass over a seq_len-timestep window (batch=1), "
     "via the same hook-based counter as metric 4. This is the stateless-inference "
     "compute cost."),
    (6, "replay_buffer_bytes_simple", "device_independent", "bytes",
     "Memory footprint of SimpleMemoryBuffer at --replay_capacity: "
     "capacity * (sequence + label + mask) bytes, using the actual dtypes stored "
     "by hgrn/buffers.py (float32 sequence, int64 label, bool mask)."),
    (7, "replay_buffer_bytes_herding", "device_independent", "bytes",
     "Memory footprint of HerdingBuffer at --herding_capacity_per_class * "
     "num_classes, plus a growth curve up to --max_cil_classes. Unlike metric 6 "
     "(fixed capacity), this grows linearly with classes seen, which matters for "
     "on-device claims about a long CIL stream."),
    (8, "inference_latency_stateless_window_ms", "device_dependent", "milliseconds",
     "Wall-clock latency of one full-window forward pass (batch=1, eval mode, "
     "no_grad), mean/median/p95/p99/std over --n_repeats after --n_warmup warmup "
     "calls. torch.cuda.synchronize() brackets each call when on GPU."),
    (9, "inference_latency_streaming_single_timestep_ms", "device_dependent", "milliseconds",
     "Wall-clock latency of one incremental single-EMG-sample forward pass (T=1), "
     "with recurrent state carried from the previous call. This is a real-time "
     "streaming/on-device deployment pattern (classify as each new sample "
     "arrives) and is DISTINCT from this repo's 'stateful' training/eval mode, "
     "which carries state window-to-window (T=seq_len per call), never "
     "timestep-to-timestep -- see AGENT_HANDOFF.md section 3 on not conflating "
     "state-handling axes. Same statistics as metric 8."),
    (10, "peak_activation_memory_bytes", "device_dependent", "bytes",
     "Peak memory during one stateless-window forward pass. On CUDA this is exact "
     "and isolated to the call (torch.cuda.max_memory_allocated after "
     "reset_peak_memory_stats). On CPU this is resource.getrusage's whole-process "
     "peak RSS high-water mark *after* the call, not isolated to it (stdlib has no "
     "exact per-call CPU activation-memory counter) -- treat the CPU number as an "
     "upper bound, not a precise delta."),
    (11, "herding_selection_wall_time_s", "device_dependent", "seconds",
     "Wall-clock time for HerdingBuffer.select_exemplars over a synthetic dataset "
     "of num_classes * --herding_dataset_size_per_class samples: one feature-"
     "extraction forward pass per sample plus the O(capacity_per_class * "
     "N_class) per-class iterative argmin selection loop."),
    (12, "replay_step_overhead_ms", "device_dependent", "milliseconds",
     "Extra wall-clock time one training step costs when a replay batch of "
     "--replay_batch_size samples is concatenated onto a --batch_size task batch, "
     "vs a task-only baseline step (both forward+backward+optimizer.step on real "
     "gradients). Reports both step distributions plus the median delta/ratio."),
    (13, "device_info", "context", "n/a",
     "Hardware/software context (CPU model, core count, CUDA device if any, torch "
     "version, whether the FLA/Triton fused HGRN kernel or the pure-PyTorch CPU "
     "timestep-loop fallback is active) needed to interpret metrics 8-12, which "
     "are only meaningful next to the machine that produced them."),
    (14, "training_flops_per_sample_estimate", "derived_estimate", "FLOPs",
     "3 * flops_per_window (metric 5), using the standard forward=1x / backward"
     "~=2x heuristic for total training compute per sample. This is an "
     "approximation, not a profiler measurement -- backward-pass FLOPs are not "
     "counted directly by the hook counter."),
]
METRIC_BY_KEY = {m[1]: m for m in METRIC_CATALOG}


def _tag(key, value):
    """Attach the metric's numbered id/category/unit to its computed value."""
    mid, _, category, unit, _ = METRIC_BY_KEY[key]
    return {"metric_id": mid, "category": category, "unit": unit, "value": value}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(arch, args):
    if arch == 'hgrn':
        return HGRNModel(args.in_channels, args.d_model, args.num_classes,
                          args.num_layers, disable_gamma_floor=args.disable_gamma_floor)
    elif arch == 'lstm':
        return LSTMModel(args.in_channels, args.d_model, args.num_classes, args.num_layers)
    raise ValueError(f"Unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Metrics 1-3: params and storage size
# ---------------------------------------------------------------------------

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    by_component = {
        name: sum(p.numel() for p in child.parameters())
        for name, child in model.named_children()
    }
    return total, by_component


def model_size_bytes(param_count_total):
    return {"fp32": param_count_total * 4, "fp16": param_count_total * 2, "int8": param_count_total * 1}


# ---------------------------------------------------------------------------
# Metrics 4-5: FLOPs via forward-hook shape counting
# ---------------------------------------------------------------------------

class MacCounter:
    """Counts multiply-accumulate ops in nn.Linear / nn.LSTM sub-modules for one
    forward call, from the actual tensor shapes seen at trace time."""

    def __init__(self, model):
        self.model = model
        self.macs = 0
        self._handles = []

    def _linear_hook(self, module, inputs, output):
        n_calls = output.numel() // module.out_features
        self.macs += n_calls * module.in_features * module.out_features

    def _lstm_hook(self, module, inputs, output):
        x = inputs[0]
        batch_size, seq_len = x.shape[0], x.shape[1]
        macs_per_step = 4 * module.hidden_size * (module.input_size + module.hidden_size)
        self.macs += batch_size * seq_len * module.num_layers * macs_per_step

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, nn.LSTM):
                self._handles.append(m.register_forward_hook(self._lstm_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def measure_flops(model, in_channels, seq_len, device):
    model.eval()
    x1 = torch.randn(1, 1, in_channels, device=device)
    xw = torch.randn(1, seq_len, in_channels, device=device)
    with torch.no_grad():
        with MacCounter(model) as mc1:
            model(x1)
        macs_t1 = mc1.macs
        with MacCounter(model) as mcw:
            model(xw)
        macs_window = mcw.macs
    macs_marginal = (macs_window - macs_t1) / (seq_len - 1) if seq_len > 1 else macs_t1
    return macs_marginal * 2, macs_window * 2  # MACs -> FLOPs (x2, standard convention)


# ---------------------------------------------------------------------------
# Metrics 6-7: replay buffer memory footprint
# ---------------------------------------------------------------------------

def per_sample_bytes(seq_len, in_channels):
    seq = torch.empty(seq_len, in_channels, dtype=torch.float32)
    label = torch.empty((), dtype=torch.int64)
    mask = torch.empty(seq_len, dtype=torch.bool)
    return sum(t.element_size() * t.nelement() for t in (seq, label, mask))


def profile_buffer_memory(args):
    per_sample = per_sample_bytes(args.seq_len, args.in_channels)
    simple_bytes = per_sample * args.replay_capacity
    herding_curve = [
        {"num_classes": c, "bytes": per_sample * args.herding_capacity_per_class * c}
        for c in range(1, args.max_cil_classes + 1)
    ]
    herding_bytes_at_num_classes = per_sample * args.herding_capacity_per_class * args.num_classes
    return {
        "replay_buffer_bytes_simple": _tag("replay_buffer_bytes_simple", {
            "per_sample_bytes": per_sample,
            "capacity": args.replay_capacity,
            "total_bytes": simple_bytes,
        }),
        "replay_buffer_bytes_herding": _tag("replay_buffer_bytes_herding", {
            "per_sample_bytes": per_sample,
            "capacity_per_class": args.herding_capacity_per_class,
            "num_classes_configured": args.num_classes,
            "total_bytes_at_configured_num_classes": herding_bytes_at_num_classes,
            "growth_curve": herding_curve,
        }),
    }


# ---------------------------------------------------------------------------
# Metrics 8-9: inference latency
# ---------------------------------------------------------------------------

def measure_latency_ms(fn, n_warmup, n_repeats, device):
    for _ in range(n_warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "std_ms": float(times.std()),
        "n_repeats": n_repeats,
    }


def measure_window_latency(model, args, device):
    model.eval()
    x = torch.randn(1, args.seq_len, args.in_channels, device=device)

    def step():
        with torch.no_grad():
            model(x)

    return measure_latency_ms(step, args.n_warmup, args.n_repeats, device)


def measure_streaming_single_timestep_latency(model, args, device):
    """Single-EMG-sample (T=1) incremental forward, state carried call-to-call.

    Distinct from this repo's 'stateful' training mode, which carries state
    window-to-window (T=seq_len), not timestep-to-timestep. See metric 9.
    """
    model.eval()
    x = torch.randn(1, 1, args.in_channels, device=device)
    state_holder = [None]

    def step():
        with torch.no_grad():
            _, next_states = model(x, states=state_holder[0])
        state_holder[0] = next_states

    return measure_latency_ms(step, args.n_warmup, args.n_repeats, device)


# ---------------------------------------------------------------------------
# Metric 10: peak activation memory
# ---------------------------------------------------------------------------

def measure_peak_activation_memory(model, args, device):
    model.eval()
    x = torch.randn(1, args.seq_len, args.in_channels, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        return {"bytes": int(torch.cuda.max_memory_allocated(device)),
                "measurement": "exact, isolated to this call (cuda)"}
    try:
        import resource
        with torch.no_grad():
            model(x)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        unit = 1 if sys.platform == 'darwin' else 1024  # macOS reports bytes, Linux reports KB
        return {"bytes": int(rss * unit),
                "measurement": "whole-process peak RSS high-water mark after this call, "
                                "NOT isolated to this call (cpu)"}
    except ImportError:
        return {"bytes": None, "measurement": "unavailable: resource module not found"}


# ---------------------------------------------------------------------------
# Metric 11: herding exemplar-selection cost
# ---------------------------------------------------------------------------

def measure_herding_selection(model, args, device):
    loader = []
    for cls in range(args.num_classes):
        x = torch.randn(args.herding_dataset_size_per_class, args.seq_len, args.in_channels)
        y = torch.full((args.herding_dataset_size_per_class,), cls, dtype=torch.int64)
        m = torch.ones(args.herding_dataset_size_per_class, args.seq_len, dtype=torch.bool)
        loader.append((x, y, m))

    buf = HerdingBuffer(capacity_per_class=args.herding_capacity_per_class)
    t0 = time.perf_counter()
    buf.select_exemplars(model, loader, device, num_classes=args.num_classes)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - t0
    return {
        "wall_time_s": wall_time_s,
        "num_classes": args.num_classes,
        "samples_per_class": args.herding_dataset_size_per_class,
        "resulting_buffer_size": len(buf),
    }


# ---------------------------------------------------------------------------
# Metric 12: replay-step training overhead
# ---------------------------------------------------------------------------

def measure_replay_step_overhead(model, args, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    def make_batch(n):
        x = torch.randn(n, args.seq_len, args.in_channels, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        return x, y

    def base_step():
        x, y = make_batch(args.batch_size)
        optimizer.zero_grad()
        logits, _ = model(x)
        criterion(logits, y).backward()
        optimizer.step()

    def replay_step():
        x, y = make_batch(args.batch_size)
        rx, ry = make_batch(args.replay_batch_size)
        x_all, y_all = torch.cat([x, rx], dim=0), torch.cat([y, ry], dim=0)
        optimizer.zero_grad()
        logits, _ = model(x_all)
        criterion(logits, y_all).backward()
        optimizer.step()

    base = measure_latency_ms(base_step, args.n_warmup, args.n_repeats, device)
    replay = measure_latency_ms(replay_step, args.n_warmup, args.n_repeats, device)
    return {
        "baseline_step_ms": base,
        "replay_augmented_step_ms": replay,
        "overhead_ms_median": replay["median_ms"] - base["median_ms"],
        "overhead_ratio_median": (replay["median_ms"] / base["median_ms"]) if base["median_ms"] > 0 else None,
    }


# ---------------------------------------------------------------------------
# Metric 13: device/software context
# ---------------------------------------------------------------------------

def get_device_info(device):
    info = {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "fla_triton_available": _HAS_FLA,
        "fla_triton_kernel_active_for_hgrn": bool(_HAS_FLA and device.type == 'cuda'),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info


# ---------------------------------------------------------------------------
# Per-architecture orchestration
# ---------------------------------------------------------------------------

def profile_one_arch(arch, args, device):
    torch.manual_seed(args.seed)
    model = build_model(arch, args).to(device)

    param_total, param_by_component = count_params(model)
    flops_marginal, flops_window = measure_flops(model, args.in_channels, args.seq_len, device)

    result = {
        "param_count_total": _tag("param_count_total", param_total),
        "param_count_by_component": _tag("param_count_by_component", param_by_component),
        "model_size_bytes": _tag("model_size_bytes", model_size_bytes(param_total)),
        "flops_per_timestep_marginal": _tag("flops_per_timestep_marginal", flops_marginal),
        "flops_per_window": _tag("flops_per_window", flops_window),
        "inference_latency_stateless_window_ms": _tag(
            "inference_latency_stateless_window_ms", measure_window_latency(model, args, device)),
        "inference_latency_streaming_single_timestep_ms": _tag(
            "inference_latency_streaming_single_timestep_ms",
            measure_streaming_single_timestep_latency(model, args, device)),
        "peak_activation_memory_bytes": _tag(
            "peak_activation_memory_bytes", measure_peak_activation_memory(model, args, device)),
        "herding_selection_wall_time_s": _tag(
            "herding_selection_wall_time_s", measure_herding_selection(model, args, device)),
        "training_flops_per_sample_estimate": _tag(
            "training_flops_per_sample_estimate", flops_window * 3),
    }
    # Last: mutates model weights via real SGD steps, so it runs after everything else.
    result["replay_step_overhead_ms"] = _tag(
        "replay_step_overhead_ms", measure_replay_step_overhead(model, args, device))
    return result


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Device-independent + device-dependent compute-cost profiling for HGRN/LSTM.")
    p.add_argument('--arch', type=str, default='both', choices=list(_ARCH_REGISTRY) + ['both'])
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])

    # Architecture hyperparameters -- default to hgrn/train.py's CLI defaults
    # for direct comparability with real runs.
    p.add_argument('--in_channels', type=int, default=10, help="NinaPro DB1 raw EMG channels.")
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--num_classes', type=int, default=17)
    p.add_argument('--seq_len', type=int, default=200, help="win_len from preprocessing.")
    p.add_argument('--disable_gamma_floor', action='store_true')

    # Buffer config -- defaults match hgrn/train.py.
    p.add_argument('--replay_capacity', type=int, default=10000)
    p.add_argument('--replay_batch_size', type=int, default=16)
    p.add_argument('--herding_capacity_per_class', type=int, default=20)
    p.add_argument('--herding_dataset_size_per_class', type=int, default=30,
                    help="Synthetic samples/class fed to select_exemplars for timing metric 11.")
    p.add_argument('--max_cil_classes', type=int, default=20,
                    help="Upper bound of the herding-buffer growth curve (metric 7).")

    # Training-step overhead config (metric 12).
    p.add_argument('--batch_size', type=int, default=32, help="Matches hgrn/train.py default.")

    # Measurement config.
    p.add_argument('--n_warmup', type=int, default=10)
    p.add_argument('--n_repeats', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default=None)
    return p


def print_summary(report):
    print("\n" + "=" * 78)
    print("COMPUTE PROFILE SUMMARY  (see METRIC_CATALOG / metric_legend for full definitions)")
    print("=" * 78)
    di = report['device_info']['value']
    print(f"Device: {di['requested_device']}  "
          f"({di['processor']}, cuda_available={di['cuda_available']})")
    bm = report["replay_buffer_memory"]
    print(f"\n[6] replay buffer (Simple, cap={bm['replay_buffer_bytes_simple']['value']['capacity']}): "
          f"{bm['replay_buffer_bytes_simple']['value']['total_bytes'] / 1e6:.2f} MB")
    print(f"[7] replay buffer (Herding, {bm['replay_buffer_bytes_herding']['value']['num_classes_configured']} "
          f"classes): {bm['replay_buffer_bytes_herding']['value']['total_bytes_at_configured_num_classes'] / 1e6:.2f} MB")
    for arch, m in report["architectures"].items():
        print(f"\n--- {arch} ---")
        print(f"[1] params: {m['param_count_total']['value']:,}")
        print(f"[3] size (fp32): {m['model_size_bytes']['value']['fp32'] / 1e6:.2f} MB")
        print(f"[4] FLOPs/timestep (marginal): {m['flops_per_timestep_marginal']['value']:,.0f}")
        print(f"[5] FLOPs/window: {m['flops_per_window']['value']:,.0f}")
        print(f"[8] window latency: {m['inference_latency_stateless_window_ms']['value']['median_ms']:.3f} ms "
              f"(p95={m['inference_latency_stateless_window_ms']['value']['p95_ms']:.3f})")
        st = m['inference_latency_streaming_single_timestep_ms']['value']
        print(f"[9] streaming single-timestep latency: {st['median_ms']:.3f} ms (p95={st['p95_ms']:.3f})")
        pk = m['peak_activation_memory_bytes']['value']
        pk_mb = f"{pk['bytes'] / 1e6:.2f} MB" if pk['bytes'] is not None else "n/a"
        print(f"[10] peak activation memory: {pk_mb} ({pk['measurement']})")
        print(f"[11] herding selection: {m['herding_selection_wall_time_s']['value']['wall_time_s']:.3f} s")
        rs = m['replay_step_overhead_ms']['value']
        print(f"[12] replay step overhead: +{rs['overhead_ms_median']:.3f} ms "
              f"(x{rs['overhead_ratio_median']:.2f})")
        print(f"[14] training FLOPs/sample (est.): {m['training_flops_per_sample_estimate']['value']:,.0f}")
    print("=" * 78 + "\n")


def main():
    args = build_arg_parser().parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    archs = list(_ARCH_REGISTRY) if args.arch == 'both' else [args.arch]

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": vars(args),
        "metric_legend": {
            str(mid): {"key": key, "category": category, "unit": unit, "description": desc}
            for mid, key, category, unit, desc in METRIC_CATALOG
        },
        "device_info": _tag("device_info", get_device_info(device)),
        "replay_buffer_memory": profile_buffer_memory(args),
        "architectures": {arch: profile_one_arch(arch, args, device) for arch in archs},
    }

    print_summary(report)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(_PROJECT_ROOT, "results_compute_profile")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"profile_{args.arch}_{timestamp}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
