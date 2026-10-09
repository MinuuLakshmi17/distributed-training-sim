"""Training orchestration, worker simulation, stragglers and checkpoint recovery."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, random
from pathlib import Path
from typing import Sequence
from .model import Example, LinearModel
from .collectives import (CommunicationStats, NetworkConfig, parameter_server_reduce,
                          ring_allreduce, bucketed_ring_allreduce)

@dataclass
class Worker:
    worker_id: int
    data: list[Example]
    speed_multiplier: float = 1.0
    alive: bool = True

    def __post_init__(self):
        if self.speed_multiplier <= 0:
            raise ValueError("speed_multiplier must be positive")

@dataclass
class TrainResult:
    model: LinearModel
    losses: list[float]
    communication: CommunicationStats
    step_times: list[float]
    recovered_failures: int = 0
    strategy: str = "ring"

    def summary(self) -> dict:
        return {"strategy": self.strategy, "steps": len(self.losses)-1,
                "initial_loss": self.losses[0], "final_loss": self.losses[-1],
                "messages": self.communication.messages, "bytes_sent": self.communication.bytes_sent,
                "communication_seconds": round(self.communication.simulated_seconds, 8),
                "simulated_training_seconds": round(sum(self.step_times), 8),
                "recovered_failures": self.recovered_failures,
                "final_parameters": [round(v, 6) for v in self.model.parameters()]}

def partition_data(data: Sequence[Example], workers: int) -> list[list[Example]]:
    if workers < 1: raise ValueError("workers must be >= 1")
    parts = [[] for _ in range(workers)]
    for i, row in enumerate(data): parts[i % workers].append(row)
    if any(not p for p in parts): raise ValueError("each worker must receive at least one example")
    return parts

def sequential_step(model: LinearModel, data: Sequence[Example], learning_rate: float) -> None:
    model.apply_gradient(model.gradient(data), learning_rate)

def train(data: Sequence[Example], initial_model: LinearModel, steps: int = 20,
          learning_rate: float = 0.05, workers: int = 2, strategy: str = "ring",
          bucket_size: int | None = None, straggler_workers: set[int] | None = None,
          straggler_delay: float = 0.0, failed_worker_steps: dict[int, int] | None = None,
          checkpoint_path: str | Path | None = None, checkpoint_every: int = 1,
          network: NetworkConfig = NetworkConfig()) -> TrainResult:
    if not data: raise ValueError("training data must not be empty")
    if steps < 0 or learning_rate <= 0: raise ValueError("steps must be >= 0 and learning_rate positive")
    if strategy not in {"ring", "parameter-server"}: raise ValueError("strategy must be ring or parameter-server")
    if checkpoint_every < 1: raise ValueError("checkpoint_every must be >= 1")
    parts = partition_data(data, workers)
    worker_list = [Worker(i, p) for i,p in enumerate(parts)]
    model = LinearModel(list(initial_model.weights), initial_model.bias)
    losses = [model.loss(data)]
    comm = CommunicationStats(); times=[]; recovered=0
    stragglers = straggler_workers or set(); failures = failed_worker_steps or {}
    for step in range(1, steps+1):
        grads=[]; worker_compute=[]
        for worker in worker_list:
            if not worker.alive: continue
            if failures.get(worker.worker_id) == step:
                worker.alive = False; recovered += 1
                continue
            # Failed workers are replaced from the latest global checkpoint before
            # the next step. This step excludes a worker that failed during its work.
            grad = model.gradient(worker.data)
            # Collectives average worker vectors, so scale local mean gradients
            # to preserve the exact global-example mean for uneven partitions.
            # (This exactness holds only when all workers participate; see the
            # failure path below.)
            scale = len(worker.data) * workers / len(data)
            grads.append([v * scale for v in grad])
            worker_compute.append((0.002 * len(worker.data) * (2.0 if worker.worker_id in stragglers else 1.0)))
        if not grads:
            # Recovery restores every worker from the global model checkpoint.
            for worker in worker_list: worker.alive = True
            grads = [[v * (len(w.data) * workers / len(data)) for v in model.gradient(w.data)] for w in worker_list]
            worker_compute = [0.002*len(w.data)*(2.0 if w.worker_id in stragglers else 1.0) for w in worker_list]
        if len(grads) != workers:
            # A failed worker's shard is genuinely missing from this step:
            # no rescaling can reconstruct its data. Rescale survivors so the
            # update magnitude stays consistent with a full step (equivalent
            # to treating the missing shard as zero gradient), and document
            # that this step trains on partial data.
            factor = len(grads) / workers
            grads = [[v * factor for v in g] for g in grads]
        if strategy == "parameter-server": reduced, c = parameter_server_reduce(grads, network)
        elif bucket_size: reduced, c = bucketed_ring_allreduce(grads, bucket_size, network)
        else: reduced, c = ring_allreduce(grads, network)
        comm.add(c); model.apply_gradient(reduced, learning_rate)
        # Synchronous step waits for the slowest participating worker; delay is simulated.
        delay = straggler_delay if any(w.worker_id in stragglers and w.alive for w in worker_list) else 0.0
        times.append(max(worker_compute, default=0.0) + c.simulated_seconds + delay)
        losses.append(model.loss(data))
        if checkpoint_path and step % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, step, losses)
        # A failed worker is restored from the updated global parameters.
        for worker in worker_list: worker.alive = True
    return TrainResult(model, losses, comm, times, recovered, strategy)

def save_checkpoint(path: str | Path, model: LinearModel, step: int, losses: Sequence[float]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "parameters": model.parameters(), "losses": list(losses)}
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)

def load_checkpoint(path: str | Path) -> tuple[LinearModel, int, list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LinearModel.from_parameters(payload["parameters"]), int(payload["step"]), list(payload["losses"])

def synthetic_data(n: int = 120, seed: int = 7) -> list[Example]:
    if n < 2: raise ValueError("n must be >= 2")
    rng = random.Random(seed); out=[]
    for i in range(n):
        x = (i / (n-1))*2.0 - 1.0
        out.append(Example([x], 2.0*x + 0.5))
    rng.shuffle(out)
    return out
