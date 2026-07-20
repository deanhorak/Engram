"""Immutable, versioned worker capability registry."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import ActionProposal, _identifier, _probability


def _identifiers(values: tuple[str, ...], name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _identifier(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


@dataclass(frozen=True)
class WorkerCapability:
    worker_id: str
    generation: int
    strategies: tuple[str, ...]
    model_ids: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()
    adapter: str = "python"
    local: bool = True
    trust: float = 0.5
    enabled: bool = True

    def __post_init__(self) -> None:
        _identifier(self.worker_id, "worker_id")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation <= 0:
            raise ValueError("worker generation must be a positive integer")
        _identifiers(self.strategies, "strategies", allow_empty=False)
        _identifiers(self.model_ids, "model_ids")
        _identifiers(self.tool_ids, "tool_ids")
        _identifier(self.adapter, "adapter")
        if not isinstance(self.local, bool) or not isinstance(self.enabled, bool):
            raise ValueError("local and enabled must be booleans")
        _probability(self.trust, "trust")

    def supports(self, proposal: ActionProposal) -> bool:
        return (
            self.enabled
            and proposal.strategy in self.strategies
            and (proposal.model_id is None or proposal.model_id in self.model_ids)
            and (proposal.tool_id is None or proposal.tool_id in self.tool_ids)
        )


@dataclass(frozen=True)
class WorkerRegistry:
    revision: int = 0
    workers: tuple[WorkerCapability, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("registry revision must be a non-negative integer")
        identities = [(worker.worker_id, worker.generation) for worker in self.workers]
        if len(set(identities)) != len(identities):
            raise ValueError("worker ID/generation pairs must be unique")
        by_worker: dict[str, list[int]] = {}
        for worker in self.workers:
            by_worker.setdefault(worker.worker_id, []).append(worker.generation)
        if any(generations != sorted(generations) for generations in by_worker.values()):
            raise ValueError("worker generations must be ordered")

    def register(self, worker: WorkerCapability) -> "WorkerRegistry":
        generations = [item.generation for item in self.workers if item.worker_id == worker.worker_id]
        if generations and worker.generation <= max(generations):
            raise ValueError("new worker generation must exceed every registered generation")
        return WorkerRegistry(self.revision + 1, self.workers + (worker,))

    def disable(self, worker_id: str) -> "WorkerRegistry":
        current = self.latest(worker_id)
        return self.register(replace(current, generation=current.generation + 1, enabled=False))

    def latest(self, worker_id: str) -> WorkerCapability:
        matches = [worker for worker in self.workers if worker.worker_id == worker_id]
        if not matches:
            raise KeyError(worker_id)
        return max(matches, key=lambda worker: worker.generation)

    def get(self, worker_id: str, generation: int) -> WorkerCapability:
        for worker in self.workers:
            if worker.worker_id == worker_id and worker.generation == generation:
                return worker
        raise KeyError((worker_id, generation))

    def resolve(self, proposal: ActionProposal) -> WorkerCapability:
        if proposal.worker_id is not None:
            worker = (
                self.get(proposal.worker_id, proposal.worker_generation)
                if proposal.worker_generation is not None
                else self.latest(proposal.worker_id)
            )
            if worker.generation != self.latest(worker.worker_id).generation:
                raise ValueError("requested worker generation is no longer current")
            if not worker.supports(proposal):
                raise ValueError("requested worker does not support the action")
            return worker
        latest = {worker_id: self.latest(worker_id) for worker_id in {w.worker_id for w in self.workers}}
        eligible = [worker for worker in latest.values() if worker.supports(proposal)]
        if not eligible:
            raise ValueError("no enabled worker supports the action")
        return sorted(eligible, key=lambda worker: (-worker.trust, not worker.local, worker.worker_id))[0]


__all__ = ["WorkerCapability", "WorkerRegistry"]
