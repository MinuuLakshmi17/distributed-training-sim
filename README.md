# distributed-training-sim

A dependency-free, CPU-only educational simulator for synchronous data-parallel training. It trains a tiny linear-regression model and compares two collective communication designs: ring allreduce and a parameter server.

## What is implemented

- **Tiny model + partitioned data:** deterministic synthetic regression data, round-robin worker partitions, mean squared-error objective, analytic gradients.
- **Ring allreduce:** simulated reduce-scatter/allgather communication cost with message, byte, round, and latency/bandwidth estimates.
- **Parameter server:** simulated worker-to-server gradient uploads and server-to-worker updates.
- **Gradient bucketing:** splits gradient vectors into fixed-size chunks and accounts for per-bucket communication overhead.
- **Stragglers:** slower simulated worker compute and configurable extra wait delay for synchronous steps.
- **Failure/recovery scenario:** a configured worker can fail on a selected step; its contribution is excluded for that step, so the step trains on the surviving workers' data rescaled to a consistent magnitude (the missing shard cannot be reconstructed), and the worker rejoins from the current global model for the next step.
- **Checkpointing:** atomic JSON checkpoint writes and checkpoint loading/inspection.
- **Verification:** tests for collective arithmetic, exact agreement with sequential full-batch SGD, uneven data partitions, checkpoint round trips, bucketing, straggler delay, and failure recovery.

## Quick start

Requires Python 3.10+. The simulator itself has no third-party runtime dependencies.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m pytest -q
```

Run ring allreduce with buckets, a straggler, and a simulated worker failure:

```bash
python -m dtrain_sim.cli train --strategy ring --workers 4 --steps 30 \
  --bucket-size 2 --straggler 2 --straggler-delay 0.05 \
  --fail-worker 1 --fail-step 5 --checkpoint checkpoint.json
```

Compare against parameter-server aggregation:

```bash
python -m dtrain_sim.cli train --strategy parameter-server --workers 4 --steps 30
python examples/compare_strategies.py
```

Inspect a saved checkpoint:

```bash
python -m dtrain_sim.cli checkpoint checkpoint.json
```

## Interpreting the output

- `final_loss` should decrease on the included synthetic dataset.
- `messages` and `bytes_sent` are counts from the simulator's analytical communication model.
- `communication_seconds` and `simulated_training_seconds` are **modeled estimates**, not wall-clock network measurements or benchmarks.
- For a fixed global batch and learning rate, one distributed step is tested against one sequential full-batch SGD step.

## Design notes and limitations

This is a simulator, not a distributed deep-learning framework. Workers execute in one Python process on CPU; no GPUs, sockets, real parameter-server process, or real network transfer are used. Ring allreduce computes the mathematically exact mean locally while estimating the costs of a ring collective. The network model assumes uniform latency and bandwidth, fixed-size scalar payloads, and idealized ring chunking. The parameter-server model uses a simplified two-message-per-worker accounting. Compute time is an intentionally simple deterministic model. Failure is injected at a step boundary, rather than during an in-flight collective; the simulator excludes that worker for that step and restores it from the global parameters afterward. Checkpoints are written at configured intervals, but automatic resume-training from a checkpoint is not yet implemented. There is no asynchronous training, optimizer state, real elastic membership, topology-aware routing, or statistical benchmarking.

## Project structure

```text
dtrain_sim/
  model.py       # linear model, loss and gradients
  collectives.py # ring, bucketing, parameter-server cost models
  training.py    # worker partitioning, orchestration, failures, checkpoints
  cli.py         # command-line interface
examples/
  compare_strategies.py
tests/
  test_simulator.py
```

## License

MIT
