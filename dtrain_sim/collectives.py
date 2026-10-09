"""Deterministic communication-cost models for parameter-server and ring allreduce."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass
class CommunicationStats:
    messages: int = 0
    bytes_sent: int = 0
    simulated_seconds: float = 0.0
    rounds: int = 0

    def add(self, other: "CommunicationStats") -> None:
        self.messages += other.messages
        self.bytes_sent += other.bytes_sent
        self.simulated_seconds += other.simulated_seconds
        self.rounds += other.rounds

@dataclass(frozen=True)
class NetworkConfig:
    latency_seconds: float = 0.001
    bandwidth_bytes_per_second: float = 100_000_000.0
    bytes_per_value: int = 8

    def __post_init__(self):
        if self.latency_seconds < 0 or self.bandwidth_bytes_per_second <= 0 or self.bytes_per_value <= 0:
            raise ValueError("invalid network configuration")

def _transfer_cost(nbytes: int, config: NetworkConfig) -> float:
    return config.latency_seconds + nbytes / config.bandwidth_bytes_per_second

def parameter_server_reduce(gradients: Sequence[Sequence[float]], config: NetworkConfig = NetworkConfig()) -> tuple[list[float], CommunicationStats]:
    if not gradients:
        raise ValueError("at least one worker gradient required")
    width = len(gradients[0])
    if any(len(g) != width for g in gradients):
        raise ValueError("gradient dimensions differ")
    n = len(gradients)
    total = [sum(g[j] for g in gradients) / n for j in range(width)]
    payload = width * config.bytes_per_value
    stats = CommunicationStats(messages=2*n, bytes_sent=2*n*payload,
        simulated_seconds=2*n*_transfer_cost(payload, config), rounds=2)
    return total, stats

def ring_allreduce(gradients: Sequence[Sequence[float]], config: NetworkConfig = NetworkConfig()) -> tuple[list[float], CommunicationStats]:
    """Simulate ring reduce-scatter + allgather cost; compute exact mean locally.

    Cost assumes evenly sized chunks (last chunk may be smaller), one send per worker
    per phase-step. This is a deterministic cost model, not a real network benchmark.
    """
    if not gradients:
        raise ValueError("at least one worker gradient required")
    width = len(gradients[0])
    if any(len(g) != width for g in gradients):
        raise ValueError("gradient dimensions differ")
    n = len(gradients)
    result = [sum(g[j] for g in gradients) / n for j in range(width)]
    if n == 1:
        return result, CommunicationStats()
    chunks = [len(range(i, width, n)) for i in range(n)]
    messages = 2 * n * (n - 1)
    # Across all workers, each phase transfers (N-1) full-vector equivalents.
    # This aggregate formula is exact for the standard ring payload model.
    bytes_sent = 2 * (n - 1) * width * config.bytes_per_value
    seconds = messages * config.latency_seconds + bytes_sent / config.bandwidth_bytes_per_second
    return result, CommunicationStats(messages, bytes_sent, seconds, rounds=2*(n-1))

def bucketed_ring_allreduce(gradients: Sequence[Sequence[float]], bucket_size: int,
                            config: NetworkConfig = NetworkConfig()) -> tuple[list[float], CommunicationStats]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if not gradients:
        raise ValueError("at least one worker gradient required")
    width = len(gradients[0])
    if any(len(g) != width for g in gradients):
        raise ValueError("gradient dimensions differ")
    result = [0.0] * width
    stats = CommunicationStats()
    for start in range(0, width, bucket_size):
        end = min(width, start + bucket_size)
        reduced, bucket_stats = ring_allreduce([g[start:end] for g in gradients], config)
        result[start:end] = reduced
        stats.add(bucket_stats)
    return result, stats
