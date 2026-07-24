"""Trained-model Milestone 3 attention substitutions on native BitNet packages."""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from engram.evaluation.native_bitnet_kernel import _load_frozen_sequences
from engram.evaluation.native_bitnet_parity import (
    _logit_metrics,
    _tensor_metrics,
    _torch_modules,
)
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.utils import atomic_json


def _attention_replacement_class():
    torch, nn, functional = _torch_modules()
    from transformers.models.bitnet.modeling_bitnet import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    class NativeBitNetAttentionReplacement(nn.Module):
        """Exact local, recurrent, retrieval, or hybrid causal attention."""

        def __init__(
            self,
            source,
            *,
            mode: str,
            local_window: int,
            recurrent_decay: float,
            retrieval_top_k: int,
            older_weight: float,
            lsh_tables: int = 4,
            lsh_bits: int = 8,
            lsh_radius: int = 1,
            retrieval_candidates: int = 12,
            lsh_seed: int = 314159,
            page_size: int = 8,
            page_bound: str = "sphere",
            sink_tokens: int = 2,
            native_attention_library: str | Path | None = None,
        ) -> None:
            super().__init__()
            if mode not in {
                "local",
                "recurrent",
                "retrieval",
                "hybrid",
                "indexed_hybrid",
                "bounded_hybrid",
                "streaming_hybrid",
                "native_streaming",
            }:
                raise ValueError(f"unsupported attention replacement mode: {mode}")
            if local_window <= 0 or retrieval_top_k <= 0:
                raise ValueError(
                    "attention window and retrieval top-k must be positive"
                )
            if not 0.0 <= recurrent_decay <= 1.0:
                raise ValueError("recurrent decay must be in [0, 1]")
            if not 0.0 <= older_weight <= 1.0:
                raise ValueError("older weight must be in [0, 1]")
            if lsh_tables <= 0 or not 1 <= lsh_bits <= 16:
                raise ValueError(
                    "LSH tables must be positive and bits must be in [1, 16]"
                )
            if not 0 <= lsh_radius <= 2:
                raise ValueError("LSH radius must be in [0, 2]")
            if retrieval_candidates < retrieval_top_k:
                raise ValueError(
                    "retrieval candidates must be at least retrieval top-k"
                )
            if page_size <= 0:
                raise ValueError("page size must be positive")
            if page_bound not in {"box", "sphere"}:
                raise ValueError("page bound must be 'box' or 'sphere'")
            if not 0 <= sink_tokens <= retrieval_top_k:
                raise ValueError("sink tokens must be in [0, retrieval_top_k]")
            self.config = source.config
            self.layer_idx = source.layer_idx
            self.head_dim = source.head_dim
            self.num_key_value_groups = source.num_key_value_groups
            self.scaling = source.scaling
            self.attention_dropout = source.attention_dropout
            self.is_causal = True
            self.q_proj = source.q_proj
            self.k_proj = source.k_proj
            self.v_proj = source.v_proj
            self.o_proj = source.o_proj
            self.attn_sub_norm = source.attn_sub_norm
            self.mode = mode
            self.local_window = int(local_window)
            self.recurrent_decay = float(recurrent_decay)
            self.retrieval_top_k = int(retrieval_top_k)
            self.older_weight = float(older_weight)
            self.lsh_tables = int(lsh_tables)
            self.lsh_bits = int(lsh_bits)
            self.lsh_radius = int(lsh_radius)
            self.retrieval_candidates = int(retrieval_candidates)
            self.page_size = int(page_size)
            self.page_bound = page_bound
            self.sink_tokens = int(sink_tokens)
            self.native_attention_library = native_attention_library
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(lsh_seed) + 104729 * int(self.layer_idx))
            projections = torch.randn(
                self.lsh_tables,
                self.lsh_bits,
                self.head_dim,
                generator=generator,
                dtype=torch.float32,
            )
            projections /= projections.norm(dim=-1, keepdim=True).clamp_min_(1e-12)
            self.register_buffer("_lsh_projections", projections, persistent=False)
            self.index_stats: dict[str, int] = {}

        @staticmethod
        def _positive_features(values):
            return functional.elu(values.float()) + 1.0

        def _score_mask(self, scores, attention_mask):
            if attention_mask is not None:
                scores = scores + attention_mask[..., : scores.shape[-1]]
            return scores

        def _local(self, query, key, value, attention_mask):
            length = query.shape[-2]
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
            positions = torch.arange(length, device=query.device)
            visible = positions[None, :] <= positions[:, None]
            visible &= positions[None, :] > (positions[:, None] - self.local_window)
            scores = scores.masked_fill(
                ~visible.reshape(1, 1, length, length),
                float("-inf"),
            )
            scores = self._score_mask(scores, attention_mask)
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            return torch.matmul(weights, value), weights

        def _recurrent(self, query, key, value):
            query_features = self._positive_features(query)
            key_features = self._positive_features(key)
            batch, heads, length, width = query.shape
            numerator = torch.zeros(
                batch,
                heads,
                width,
                value.shape[-1],
                dtype=torch.float32,
                device=query.device,
            )
            normalizer = torch.zeros(
                batch,
                heads,
                width,
                dtype=torch.float32,
                device=query.device,
            )
            rows = []
            for position in range(length):
                feature = key_features[:, :, position]
                current_value = value[:, :, position].float()
                numerator = self.recurrent_decay * numerator + torch.einsum(
                    "bhd,bhv->bhdv",
                    feature,
                    current_value,
                )
                normalizer = self.recurrent_decay * normalizer + feature
                query_feature = query_features[:, :, position]
                denominator = torch.einsum(
                    "bhd,bhd->bh",
                    query_feature,
                    normalizer,
                ).clamp_min_(1e-6)
                row = (
                    torch.einsum(
                        "bhd,bhdv->bhv",
                        query_feature,
                        numerator,
                    )
                    / denominator[..., None]
                )
                rows.append(row)
            return torch.stack(rows, dim=2)

        def _older_reads(self, query, key, value):
            batch, heads, length, width = query.shape
            value_width = value.shape[-1]
            query_features = self._positive_features(query)
            key_features = self._positive_features(key)
            numerator = torch.zeros(
                batch,
                heads,
                width,
                value_width,
                dtype=torch.float32,
                device=query.device,
            )
            normalizer = torch.zeros(
                batch,
                heads,
                width,
                dtype=torch.float32,
                device=query.device,
            )
            recurrent_rows = []
            retrieval_rows = []
            has_older = []
            for position in range(length):
                evicted = position - self.local_window
                if evicted >= 0:
                    feature = key_features[:, :, evicted]
                    evicted_value = value[:, :, evicted].float()
                    numerator = self.recurrent_decay * numerator + torch.einsum(
                        "bhd,bhv->bhdv",
                        feature,
                        evicted_value,
                    )
                    normalizer = self.recurrent_decay * normalizer + feature
                query_feature = query_features[:, :, position]
                denominator = torch.einsum(
                    "bhd,bhd->bh",
                    query_feature,
                    normalizer,
                ).clamp_min_(1e-6)
                recurrent = (
                    torch.einsum(
                        "bhd,bhdv->bhv",
                        query_feature,
                        numerator,
                    )
                    / denominator[..., None]
                )
                if evicted >= 0:
                    older_key = key[:, :, : evicted + 1].float()
                    older_value = value[:, :, : evicted + 1].float()
                    scores = (
                        torch.einsum(
                            "bhd,bhtd->bht",
                            query[:, :, position].float(),
                            older_key,
                        )
                        * self.scaling
                    )
                    count = min(self.retrieval_top_k, evicted + 1)
                    top_scores, indices = torch.topk(scores, count, dim=-1)
                    gather = indices[..., None].expand(
                        batch,
                        heads,
                        count,
                        value_width,
                    )
                    selected = torch.gather(older_value, 2, gather)
                    retrieval = torch.einsum(
                        "bht,bhtv->bhv",
                        torch.softmax(top_scores, dim=-1),
                        selected,
                    )
                else:
                    retrieval = torch.zeros_like(recurrent)
                recurrent_rows.append(recurrent)
                retrieval_rows.append(retrieval)
                has_older.append(evicted >= 0)
            mask = torch.tensor(
                has_older,
                device=query.device,
                dtype=torch.bool,
            ).reshape(1, 1, length, 1)
            return (
                torch.stack(recurrent_rows, dim=2),
                torch.stack(retrieval_rows, dim=2),
                mask,
            )

        def _local_retrieval(self, query, key, value):
            """Jointly normalize exact local keys and exact top older keys."""

            rows = []
            length = query.shape[-2]
            for position in range(length):
                local_start = max(0, position - self.local_window + 1)
                visible_key = key[:, :, local_start : position + 1]
                visible_value = value[:, :, local_start : position + 1]
                if local_start > 0:
                    older_key = key[:, :, :local_start]
                    older_value = value[:, :, :local_start]
                    older_scores = (
                        torch.einsum(
                            "bhd,bhtd->bht",
                            query[:, :, position],
                            older_key,
                        )
                        * self.scaling
                    )
                    count = min(self.retrieval_top_k, local_start)
                    _, indices = torch.topk(older_scores, count, dim=-1)
                    key_gather = indices[..., None].expand(
                        *indices.shape,
                        key.shape[-1],
                    )
                    value_gather = indices[..., None].expand(
                        *indices.shape,
                        value.shape[-1],
                    )
                    selected_key = torch.gather(older_key, 2, key_gather)
                    selected_value = torch.gather(older_value, 2, value_gather)
                    visible_key = torch.cat((visible_key, selected_key), dim=2)
                    visible_value = torch.cat(
                        (visible_value, selected_value),
                        dim=2,
                    )
                scores = (
                    torch.einsum(
                        "bhd,bhtd->bht",
                        query[:, :, position],
                        visible_key,
                    )
                    * self.scaling
                )
                weights = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                rows.append(
                    torch.einsum(
                        "bht,bhtv->bhv",
                        weights,
                        visible_value,
                    )
                )
            return torch.stack(rows, dim=2)

        def _hash_codes(self, vectors):
            projections = self._lsh_projections.to(vectors.device)
            products = torch.einsum("...d,tbd->...tb", vectors.float(), projections)
            powers = (1 << torch.arange(self.lsh_bits, device=vectors.device)).long()
            return ((products >= 0).long() * powers).sum(dim=-1)

        def _neighbor_codes(self, code: int) -> tuple[int, ...]:
            result = [code]
            for radius in range(1, self.lsh_radius + 1):
                for bits in combinations(range(self.lsh_bits), radius):
                    neighbor = code
                    for bit in bits:
                        neighbor ^= 1 << bit
                    result.append(neighbor)
            return tuple(result)

        def _indexed_local_retrieval(self, query, key, value):
            """Online sign-LSH postings followed by exact candidate reranking."""

            batch, heads, length, _ = query.shape
            key_codes = self._hash_codes(key)
            query_codes = self._hash_codes(query)
            output_batches = []
            stats = {
                "queries_with_older": 0,
                "oracle_slots": 0,
                "oracle_hits": 0,
                "candidate_slots_per_head": 0,
                "candidate_slots_unique_kv_group": 0,
                "selected_slots_unique_kv_group": 0,
                "indexed_older_tokens": 0,
                "posting_probes": 0,
            }
            for batch_index in range(batch):
                postings = [
                    [dict() for _ in range(self.lsh_tables)] for _ in range(heads)
                ]
                batch_rows = []
                for position in range(length):
                    local_start = max(0, position - self.local_window + 1)
                    newly_older = local_start - 1
                    if newly_older >= 0:
                        stats["indexed_older_tokens"] += 1
                        for head in range(heads):
                            for table in range(self.lsh_tables):
                                code = int(
                                    key_codes[
                                        batch_index, head, newly_older, table
                                    ].item()
                                )
                                postings[head][table].setdefault(code, []).append(
                                    newly_older
                                )
                    head_rows = []
                    candidate_sets: list[set[int]] = []
                    selected_sets: list[set[int]] = []
                    for head in range(heads):
                        candidates: dict[int, int] = {}
                        if local_start > 0:
                            stats["queries_with_older"] += 1
                            for table in range(self.lsh_tables):
                                query_code = int(
                                    query_codes[
                                        batch_index, head, position, table
                                    ].item()
                                )
                                for code in self._neighbor_codes(query_code):
                                    stats["posting_probes"] += 1
                                    for candidate in postings[head][table].get(
                                        code, ()
                                    ):
                                        candidates[candidate] = (
                                            candidates.get(candidate, 0) + 1
                                        )
                            ranked = sorted(
                                candidates,
                                key=lambda candidate: (
                                    -candidates[candidate],
                                    -candidate,
                                ),
                            )[: self.retrieval_candidates]
                            # Fixed anchors cover sparse buckets without scanning old keys.
                            for anchor in (0, local_start - 1):
                                if (
                                    len(ranked) < self.retrieval_candidates
                                    and anchor not in ranked
                                    and 0 <= anchor < local_start
                                ):
                                    ranked.append(anchor)
                            candidate_indices = torch.tensor(
                                ranked,
                                device=query.device,
                                dtype=torch.long,
                            )
                            exact_older_scores = (
                                torch.matmul(
                                    key[batch_index, head, :local_start],
                                    query[batch_index, head, position],
                                )
                                * self.scaling
                            )
                            oracle_count = min(self.retrieval_top_k, local_start)
                            oracle = set(
                                int(index)
                                for index in torch.topk(
                                    exact_older_scores,
                                    oracle_count,
                                ).indices.tolist()
                            )
                            candidate_set = set(ranked)
                            stats["oracle_slots"] += len(oracle)
                            stats["oracle_hits"] += len(oracle & candidate_set)
                            if candidate_indices.numel():
                                candidate_scores = exact_older_scores[candidate_indices]
                                selected_count = min(
                                    self.retrieval_top_k,
                                    candidate_indices.numel(),
                                )
                                selected_indices = candidate_indices[
                                    torch.topk(
                                        candidate_scores,
                                        selected_count,
                                    ).indices
                                ]
                            else:
                                selected_indices = candidate_indices
                        else:
                            candidate_indices = torch.empty(
                                0,
                                device=query.device,
                                dtype=torch.long,
                            )
                            selected_indices = candidate_indices
                        candidate_set = set(int(v) for v in candidate_indices.tolist())
                        selected_set = set(int(v) for v in selected_indices.tolist())
                        candidate_sets.append(candidate_set)
                        selected_sets.append(selected_set)
                        stats["candidate_slots_per_head"] += len(candidate_set)
                        local_indices = torch.arange(
                            local_start,
                            position + 1,
                            device=query.device,
                        )
                        visible_indices = torch.cat((local_indices, selected_indices))
                        visible_key = key[batch_index, head, visible_indices]
                        visible_value = value[batch_index, head, visible_indices]
                        scores = (
                            torch.matmul(
                                visible_key,
                                query[batch_index, head, position],
                            )
                            * self.scaling
                        )
                        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                            query.dtype
                        )
                        head_rows.append(torch.matmul(weights, visible_value))
                    for kv_head in range(heads // self.num_key_value_groups):
                        first = kv_head * self.num_key_value_groups
                        last = first + self.num_key_value_groups
                        stats["candidate_slots_unique_kv_group"] += len(
                            set().union(*candidate_sets[first:last])
                        )
                        stats["selected_slots_unique_kv_group"] += len(
                            set().union(*selected_sets[first:last])
                        )
                    batch_rows.append(torch.stack(head_rows))
                output_batches.append(torch.stack(batch_rows, dim=1))
            self.index_stats = stats
            return torch.stack(output_batches)

        def _bounded_local_retrieval(self, query, key, value):
            """Exact top-k via page bounding boxes and branch-and-bound."""

            batch, heads, length, _ = query.shape
            output_batches = []
            stats = {
                "queries_with_older": 0,
                "older_slots": 0,
                "candidate_slots_per_head": 0,
                "candidate_slots_unique_kv_group": 0,
                "selected_slots_unique_kv_group": 0,
                "bound_page_reads_unique_kv_group": 0,
                "total_pages": 0,
                "opened_pages_per_head": 0,
            }
            for batch_index in range(batch):
                batch_rows = []
                for position in range(length):
                    local_start = max(0, position - self.local_window + 1)
                    pages = [
                        (start, min(start + self.page_size, local_start))
                        for start in range(0, local_start, self.page_size)
                    ]
                    stats["older_slots"] += local_start * heads
                    stats["total_pages"] += len(pages)
                    head_rows = []
                    candidate_sets: list[set[int]] = []
                    selected_sets: list[set[int]] = []
                    for head in range(heads):
                        selected_indices = torch.empty(
                            0,
                            device=query.device,
                            dtype=torch.long,
                        )
                        candidates: list[int] = []
                        if pages:
                            stats["queries_with_older"] += 1
                            query_row = query[batch_index, head, position].float()
                            page_bounds = []
                            for start, end in pages:
                                page_keys = key[batch_index, head, start:end].float()
                                if self.page_bound == "box":
                                    minimum = page_keys.amin(dim=0)
                                    maximum = page_keys.amax(dim=0)
                                    bound = torch.where(
                                        query_row >= 0,
                                        query_row * maximum,
                                        query_row * minimum,
                                    ).sum()
                                else:
                                    center = page_keys.mean(dim=0)
                                    radius = (page_keys - center).norm(dim=-1).max()
                                    bound = torch.dot(query_row, center) + (
                                        query_row.norm() * radius
                                    )
                                page_bounds.append(float(bound.item() * self.scaling))
                            order = sorted(
                                range(len(pages)),
                                key=lambda index: -page_bounds[index],
                            )
                            scored: list[tuple[float, int]] = []
                            for order_index, page_index in enumerate(order):
                                start, end = pages[page_index]
                                page_scores = (
                                    torch.matmul(
                                        key[batch_index, head, start:end].float(),
                                        query_row,
                                    )
                                    * self.scaling
                                )
                                candidates.extend(range(start, end))
                                scored.extend(
                                    (float(score), start + offset)
                                    for offset, score in enumerate(page_scores.tolist())
                                )
                                scored.sort(reverse=True)
                                stats["opened_pages_per_head"] += 1
                                if (
                                    len(scored) >= self.retrieval_top_k
                                    and order_index + 1 < len(order)
                                    and page_bounds[order[order_index + 1]]
                                    <= scored[self.retrieval_top_k - 1][0]
                                ):
                                    break
                            selected_indices = torch.tensor(
                                [index for _, index in scored[: self.retrieval_top_k]],
                                device=query.device,
                                dtype=torch.long,
                            )
                        candidate_set = set(candidates)
                        selected_set = set(int(v) for v in selected_indices.tolist())
                        candidate_sets.append(candidate_set)
                        selected_sets.append(selected_set)
                        stats["candidate_slots_per_head"] += len(candidate_set)
                        local_indices = torch.arange(
                            local_start,
                            position + 1,
                            device=query.device,
                        )
                        visible_indices = torch.cat((local_indices, selected_indices))
                        visible_key = key[batch_index, head, visible_indices]
                        visible_value = value[batch_index, head, visible_indices]
                        scores = (
                            torch.matmul(
                                visible_key,
                                query[batch_index, head, position],
                            )
                            * self.scaling
                        )
                        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                            query.dtype
                        )
                        head_rows.append(torch.matmul(weights, visible_value))
                    for kv_head in range(heads // self.num_key_value_groups):
                        first = kv_head * self.num_key_value_groups
                        last = first + self.num_key_value_groups
                        stats["bound_page_reads_unique_kv_group"] += len(pages)
                        stats["candidate_slots_unique_kv_group"] += len(
                            set().union(*candidate_sets[first:last])
                        )
                        stats["selected_slots_unique_kv_group"] += len(
                            set().union(*selected_sets[first:last])
                        )
                    batch_rows.append(torch.stack(head_rows))
                output_batches.append(torch.stack(batch_rows, dim=1))
            self.index_stats = stats
            return torch.stack(output_batches)

        def _streaming_local_retrieval(self, query, key, value):
            """Exact local attention plus bounded sinks and online heavy hitters."""

            batch, heads, length, _ = query.shape
            output_batches = []
            heavy_capacity = self.retrieval_candidates - self.sink_tokens
            stats = {
                "queries_with_older": 0,
                "selected_slots_unique_kv_group": 0,
                "candidate_slots_unique_kv_group": 0,
                "candidate_slots_per_head": 0,
                "heavy_insertions": 0,
                "heavy_evictions": 0,
            }
            for batch_index in range(batch):
                mass = torch.zeros(heads, length, dtype=torch.float32)
                heavy: list[dict[int, float]] = [dict() for _ in range(heads)]
                batch_rows = []
                for position in range(length):
                    local_start = max(0, position - self.local_window + 1)
                    newly_older = local_start - 1
                    head_rows = []
                    candidate_sets: list[set[int]] = []
                    selected_sets: list[set[int]] = []
                    for head in range(heads):
                        if newly_older >= self.sink_tokens:
                            heavy[head][newly_older] = float(
                                mass[head, newly_older].item()
                            )
                            stats["heavy_insertions"] += 1
                            if len(heavy[head]) > heavy_capacity:
                                victim = min(
                                    heavy[head],
                                    key=lambda index: (
                                        heavy[head][index],
                                        index,
                                    ),
                                )
                                del heavy[head][victim]
                                stats["heavy_evictions"] += 1
                        older = list(range(min(self.sink_tokens, local_start)))
                        older.extend(
                            sorted(
                                heavy[head],
                                key=lambda index: (
                                    -heavy[head][index],
                                    index,
                                ),
                            )
                        )
                        candidate_indices = torch.tensor(
                            older,
                            device=query.device,
                            dtype=torch.long,
                        )
                        if older:
                            stats["queries_with_older"] += 1
                        stats["candidate_slots_per_head"] += len(older)
                        candidate_sets.append(set(older))
                        if candidate_indices.numel() > self.retrieval_top_k:
                            candidate_scores = (
                                torch.matmul(
                                    key[batch_index, head, candidate_indices],
                                    query[batch_index, head, position],
                                )
                                * self.scaling
                            )
                            older_indices = candidate_indices[
                                torch.topk(
                                    candidate_scores,
                                    self.retrieval_top_k,
                                ).indices
                            ]
                        else:
                            older_indices = candidate_indices
                        selected_sets.append(
                            set(int(value) for value in older_indices.tolist())
                        )
                        local_indices = torch.arange(
                            local_start,
                            position + 1,
                            device=query.device,
                        )
                        visible_indices = torch.cat((local_indices, older_indices))
                        visible_key = key[batch_index, head, visible_indices]
                        visible_value = value[batch_index, head, visible_indices]
                        scores = (
                            torch.matmul(
                                visible_key,
                                query[batch_index, head, position],
                            )
                            * self.scaling
                        )
                        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
                        for offset, index in enumerate(visible_indices.tolist()):
                            mass[head, index] += weights[offset].cpu()
                            if index in heavy[head]:
                                heavy[head][index] = float(mass[head, index].item())
                        head_rows.append(
                            torch.matmul(weights.to(query.dtype), visible_value)
                        )
                    for kv_head in range(heads // self.num_key_value_groups):
                        first = kv_head * self.num_key_value_groups
                        last = first + self.num_key_value_groups
                        stats["candidate_slots_unique_kv_group"] += len(
                            set().union(*candidate_sets[first:last])
                        )
                        stats["selected_slots_unique_kv_group"] += len(
                            set().union(*selected_sets[first:last])
                        )
                    batch_rows.append(torch.stack(head_rows))
                output_batches.append(torch.stack(batch_rows, dim=1))
            self.index_stats = stats
            return torch.stack(output_batches)

        def _native_streaming(self, query, key, value):
            from engram.runtime.native_attention import NativeStreamingAttention

            output_batches = []
            totals: dict[str, int] = {}
            for batch_index in range(query.shape[0]):
                batch_rows = []
                with NativeStreamingAttention(
                    query_heads=query.shape[1],
                    key_value_heads=key.shape[1],
                    head_dimension=query.shape[-1],
                    local_window=self.local_window,
                    older_candidates=self.retrieval_candidates,
                    older_top_k=self.retrieval_top_k,
                    sink_tokens=self.sink_tokens,
                    scale=self.scaling,
                    library=self.native_attention_library,
                ) as attention:
                    for position in range(query.shape[2]):
                        output, metrics = attention.step(
                            query[batch_index, :, position].float().cpu().numpy(),
                            key[batch_index, :, position].float().cpu().numpy(),
                            value[batch_index, :, position].float().cpu().numpy(),
                        )
                        batch_rows.append(torch.from_numpy(output))
                        for name in (
                            "candidate_key_bytes",
                            "selected_value_bytes",
                            "local_kv_bytes",
                        ):
                            totals[name] = totals.get(name, 0) + int(
                                getattr(metrics, name)
                            )
                        totals["state_bytes"] = max(
                            totals.get("state_bytes", 0),
                            int(metrics.state_bytes),
                        )
                        totals["scratch_bytes"] = max(
                            totals.get("scratch_bytes", 0),
                            int(metrics.scratch_bytes),
                        )
                output_batches.append(torch.stack(batch_rows, dim=1))
            self.index_stats = totals
            return torch.stack(output_batches).to(query.device)

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            **_kwargs,
        ):
            if past_key_values is not None:
                raise ValueError(
                    "Milestone 3 attention substitutions currently require "
                    "full-sequence evaluation with use_cache=False"
                )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(
                query,
                key,
                *position_embeddings,
            )
            native_key = key
            native_value = value
            if self.mode != "native_streaming":
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)
            if self.mode == "local":
                output, weights = self._local(query, key, value, attention_mask)
            elif self.mode == "recurrent":
                output = self._recurrent(query, key, value)
                weights = None
            elif self.mode == "hybrid":
                output = self._local_retrieval(query, key, value)
                weights = None
            elif self.mode == "indexed_hybrid":
                output = self._indexed_local_retrieval(query, key, value)
                weights = None
            elif self.mode == "bounded_hybrid":
                output = self._bounded_local_retrieval(query, key, value)
                weights = None
            elif self.mode == "streaming_hybrid":
                output = self._streaming_local_retrieval(query, key, value)
                weights = None
            elif self.mode == "native_streaming":
                output = self._native_streaming(query, native_key, native_value)
                weights = None
            else:
                local, local_weights = self._local(
                    query,
                    key,
                    value,
                    attention_mask,
                )
                recurrent, retrieval, has_older = self._older_reads(
                    query,
                    key,
                    value,
                )
                older = retrieval
                blended = (1.0 - self.older_weight) * local + self.older_weight * older
                output = torch.where(has_older, blended, local)
                weights = local_weights
            output = output.to(hidden_states.dtype)
            output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            output = self.attn_sub_norm(output)
            return self.o_proj(output), weights

    return NativeBitNetAttentionReplacement


def _head_analysis(attentions, *, local_window: int) -> dict[str, Any]:
    torch, _, _ = _torch_modules()
    layers = []
    retrieval_head_fractions = []
    for layer_index, weights in enumerate(attentions):
        if weights is None:
            continue
        probabilities = weights.float()
        length = probabilities.shape[-1]
        positions = torch.arange(length, device=probabilities.device)
        local = positions[None, :] > positions[:, None] - local_window
        causal = positions[None, :] <= positions[:, None]
        local &= causal
        older = causal & ~local
        older_mass = (
            probabilities * older.reshape(1, 1, length, length).to(probabilities.dtype)
        ).sum(dim=-1)
        valid = positions >= local_window
        per_head = older_mass[:, :, valid].mean(dim=(0, 2))
        entropy = -(
            probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        fraction = float((per_head >= 0.2).float().mean().item())
        retrieval_head_fractions.append(fraction)
        layers.append(
            {
                "layer": layer_index,
                "mean_attention_entropy": float(entropy.mean().item()),
                "mean_older_mass": float(per_head.mean().item()),
                "maximum_head_older_mass": float(per_head.max().item()),
                "retrieval_head_fraction_at_20_percent_older_mass": fraction,
            }
        )
    return {
        "local_window": local_window,
        "layers": layers,
        "mean_retrieval_head_fraction": (
            sum(retrieval_head_fractions) / len(retrieval_head_fractions)
            if retrieval_head_fractions
            else 0.0
        ),
    }


def evaluate_native_bitnet_attention_substitution(
    package: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    library: str | Path | None = None,
    threads: int | None = None,
    native_projections: bool = False,
    sequence_count: int = 2,
    prediction_positions: int = 32,
    record_offset: int = 0,
    modes: Sequence[str] = ("local", "recurrent", "retrieval", "hybrid"),
    layers: Sequence[int] | None = None,
    local_window: int = 16,
    recurrent_decay: float = 0.99,
    retrieval_top_k: int = 4,
    older_weight: float = 0.5,
    retrieval_candidates: int = 12,
    lsh_tables: int = 4,
    lsh_bits: int = 8,
    lsh_radius: int = 1,
    lsh_seed: int = 314159,
    page_size: int = 8,
    page_bound: str = "sphere",
    sink_tokens: int = 2,
    native_attention_library: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate trained all-layer attention replacements after the MLP pass."""

    if prediction_positions % sequence_count:
        raise ValueError("prediction positions must divide evenly across sequences")
    predictions_per_sequence = prediction_positions // sequence_count
    tokens_per_sequence = predictions_per_sequence + 1
    texts, dataset_evidence = _load_frozen_sequences(
        dataset,
        sequence_count=sequence_count,
        record_offset=record_offset,
    )
    torch, _, functional = _torch_modules()
    with NativeBitNetRuntime(
        package,
        library=library,
        threads=threads,
        native_projections=native_projections,
    ) as runtime:
        num_attention_heads = int(runtime.model.config.num_attention_heads)
        num_key_value_heads = int(runtime.model.config.num_key_value_heads)
        head_dim = int(runtime.model.config.hidden_size // num_attention_heads)
        runtime.model.config._attn_implementation = "eager"
        encoded = []
        for index, text in enumerate(texts):
            tokens = runtime.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < tokens_per_sequence:
                raise ValueError(f"attention sequence {index} has too few tokens")
            encoded.append([int(value) for value in tokens[:tokens_per_sequence]])
        input_ids = torch.tensor(encoded, dtype=torch.long)
        runtime.kernel.clear_metrics()
        with torch.inference_mode():
            started = time.perf_counter()
            baseline = runtime.forward(
                input_ids,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )
            baseline_seconds = time.perf_counter() - started
        baseline_calls = list(runtime.kernel.calls)
        head_analysis = _head_analysis(
            baseline.attentions,
            local_window=local_window,
        )
        selected_layers = (
            list(range(len(runtime.model.model.layers)))
            if layers is None
            else list(dict.fromkeys(int(value) for value in layers))
        )
        if any(
            value < 0 or value >= len(runtime.model.model.layers)
            for value in selected_layers
        ):
            raise ValueError("attention layer is outside the model")
        Replacement = _attention_replacement_class()
        results = {}
        baseline_logits = baseline.logits[:, :-1]
        baseline_hidden = baseline.hidden_states[-1][:, :-1]
        labels = input_ids[:, 1:]
        baseline_nll = functional.cross_entropy(
            baseline_logits.float().reshape(-1, baseline_logits.shape[-1]),
            labels.reshape(-1),
        )
        originals = {
            index: runtime.model.model.layers[index].self_attn
            for index in selected_layers
        }
        try:
            for mode in dict.fromkeys(str(value) for value in modes):
                for index, original in originals.items():
                    runtime.model.model.layers[index].self_attn = Replacement(
                        original,
                        mode=mode,
                        local_window=local_window,
                        recurrent_decay=recurrent_decay,
                        retrieval_top_k=retrieval_top_k,
                        older_weight=older_weight,
                        retrieval_candidates=retrieval_candidates,
                        lsh_tables=lsh_tables,
                        lsh_bits=lsh_bits,
                        lsh_radius=lsh_radius,
                        lsh_seed=lsh_seed,
                        page_size=page_size,
                        page_bound=page_bound,
                        sink_tokens=sink_tokens,
                        native_attention_library=native_attention_library,
                    )
                runtime.kernel.clear_metrics()
                with torch.inference_mode():
                    started = time.perf_counter()
                    candidate = runtime.forward(
                        input_ids,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    elapsed = time.perf_counter() - started
                candidate_logits = candidate.logits[:, :-1]
                candidate_nll = functional.cross_entropy(
                    candidate_logits.float().reshape(
                        -1,
                        candidate_logits.shape[-1],
                    ),
                    labels.reshape(-1),
                )
                results[mode] = {
                    "logits": _logit_metrics(
                        baseline_logits,
                        candidate_logits,
                    ),
                    "final_hidden": _tensor_metrics(
                        baseline_hidden,
                        candidate.hidden_states[-1][:, :-1],
                    ),
                    "nll_delta": float((candidate_nll - baseline_nll).item()),
                    "elapsed_seconds": elapsed,
                    "mlp_calls": len(runtime.kernel.calls),
                }
                if mode in {
                    "indexed_hybrid",
                    "bounded_hybrid",
                    "streaming_hybrid",
                    "native_streaming",
                }:
                    totals: dict[str, int | float] = {}
                    for index in selected_layers:
                        replacement = runtime.model.model.layers[index].self_attn
                        for name, value in replacement.index_stats.items():
                            totals[name] = int(totals.get(name, 0)) + int(value)
                    if mode not in {"streaming_hybrid", "native_streaming"}:
                        totals["oracle_recall"] = (
                            1.0
                            if mode == "bounded_hybrid"
                            else (
                                int(totals["oracle_hits"]) / int(totals["oracle_slots"])
                                if totals.get("oracle_slots")
                                else 1.0
                            )
                        )
                    results[mode]["index"] = totals
                for index, original in originals.items():
                    runtime.model.model.layers[index].self_attn = original
        finally:
            for index, original in originals.items():
                runtime.model.model.layers[index].self_attn = original

    thresholds = {
        "maximum_teacher_student_kl": 0.05,
        "minimum_teacher_top1_agreement": 0.9,
        "maximum_nll_delta": 0.05,
        "maximum_final_hidden_relative_l2": 0.1,
        "minimum_unique_sequences": 8,
        "minimum_prediction_positions": 256,
        "minimum_index_oracle_recall": 0.95,
    }
    semantic_checks = {}
    for mode, result in results.items():
        semantic_checks[mode] = {
            "teacher_student_kl": result["logits"]["mean_kl_divergence"]
            <= thresholds["maximum_teacher_student_kl"],
            "teacher_top1_agreement": result["logits"]["top1_agreement"]
            >= thresholds["minimum_teacher_top1_agreement"],
            "nll_delta": result["nll_delta"] <= thresholds["maximum_nll_delta"],
            "final_hidden_relative_l2": result["final_hidden"]["relative_l2"]
            <= thresholds["maximum_final_hidden_relative_l2"],
            "unique_sequences": dataset_evidence["unique_sequences"]
            >= thresholds["minimum_unique_sequences"],
            "prediction_positions": prediction_positions
            >= thresholds["minimum_prediction_positions"],
        }
        if mode in {"indexed_hybrid", "bounded_hybrid"}:
            semantic_checks[mode]["index_oracle_recall"] = (
                result["index"]["oracle_recall"]
                >= thresholds["minimum_index_oracle_recall"]
            )
    semantic_confirmation_passed = len(results) == 1 and all(
        next(iter(semantic_checks.values())).values()
    )
    bounded_attention_confirmation_passed = semantic_confirmation_passed and next(
        iter(results)
    ) in {"streaming_hybrid", "native_streaming"}
    dense_slots = sum(range(1, tokens_per_sequence + 1))
    selected_slots = sum(
        min(position + 1, local_window)
        + min(max(0, position + 1 - local_window), retrieval_top_k)
        for position in range(tokens_per_sequence)
    )
    key_bytes_per_slot = num_key_value_heads * head_dim * 2
    value_bytes_per_slot = key_bytes_per_slot
    evaluated_layer_count = len(selected_layers)
    dense_kv_bytes = (
        evaluated_layer_count
        * sequence_count
        * dense_slots
        * (key_bytes_per_slot + value_bytes_per_slot)
    )
    selected_value_bytes = sequence_count * selected_slots * value_bytes_per_slot
    exact_topk_key_scan_bytes = sequence_count * dense_slots * key_bytes_per_slot
    indexed_traffic = None
    bounded_traffic = None
    streaming_traffic = None
    native_streaming_traffic = None
    if "indexed_hybrid" in results:
        index = results["indexed_hybrid"]["index"]
        local_slots = (
            evaluated_layer_count
            * sequence_count
            * sum(
                min(position + 1, local_window)
                for position in range(tokens_per_sequence)
            )
            * num_key_value_heads
        )
        local_kv_bytes = local_slots * head_dim * 2 * 2
        candidate_key_bytes = (
            int(index["candidate_slots_unique_kv_group"]) * head_dim * 2
        )
        retrieved_value_bytes = (
            int(index["selected_slots_unique_kv_group"]) * head_dim * 2
        )
        posting_bytes = (
            int(index["indexed_older_tokens"]) * num_key_value_heads * lsh_tables * 4
        )
        total_indexed_bytes = (
            local_kv_bytes + candidate_key_bytes + retrieved_value_bytes + posting_bytes
        )
        indexed_traffic = {
            "local_kv_bytes": local_kv_bytes,
            "candidate_exact_key_bytes": candidate_key_bytes,
            "retrieved_value_bytes": retrieved_value_bytes,
            "posting_id_write_bytes": posting_bytes,
            "total_logical_bytes": total_indexed_bytes,
            "fraction_of_dense_kv": total_indexed_bytes / dense_kv_bytes,
            "candidate_oracle_recall": index["oracle_recall"],
            "candidate_slots_unique_kv_group": index["candidate_slots_unique_kv_group"],
            "selected_slots_unique_kv_group": index["selected_slots_unique_kv_group"],
        }
    if "bounded_hybrid" in results:
        index = results["bounded_hybrid"]["index"]
        local_slots = (
            evaluated_layer_count
            * sequence_count
            * sum(
                min(position + 1, local_window)
                for position in range(tokens_per_sequence)
            )
            * num_key_value_heads
        )
        local_kv_bytes = local_slots * head_dim * 2 * 2
        candidate_key_bytes = (
            int(index["candidate_slots_unique_kv_group"]) * head_dim * 2
        )
        retrieved_value_bytes = (
            int(index["selected_slots_unique_kv_group"]) * head_dim * 2
        )
        metadata_bytes_per_page = (
            head_dim * 2 * 4 if page_bound == "box" else head_dim * 4 + 4
        )
        bound_metadata_bytes = (
            int(index["bound_page_reads_unique_kv_group"]) * metadata_bytes_per_page
        )
        total_bounded_bytes = (
            local_kv_bytes
            + candidate_key_bytes
            + retrieved_value_bytes
            + bound_metadata_bytes
        )
        bounded_traffic = {
            "local_kv_bytes": local_kv_bytes,
            "page_bound_metadata_bytes": bound_metadata_bytes,
            "metadata_bytes_per_page": metadata_bytes_per_page,
            "candidate_exact_key_bytes": candidate_key_bytes,
            "retrieved_value_bytes": retrieved_value_bytes,
            "total_logical_bytes": total_bounded_bytes,
            "fraction_of_dense_kv": total_bounded_bytes / dense_kv_bytes,
            "candidate_oracle_recall": 1.0,
            "candidate_slots_unique_kv_group": index["candidate_slots_unique_kv_group"],
            "selected_slots_unique_kv_group": index["selected_slots_unique_kv_group"],
            "opened_page_fraction": (
                int(index["opened_pages_per_head"])
                / (
                    int(index["total_pages"]) * num_attention_heads
                    if index["total_pages"]
                    else 1
                )
            ),
        }
    if "streaming_hybrid" in results:
        index = results["streaming_hybrid"]["index"]
        local_slots = (
            evaluated_layer_count
            * sequence_count
            * sum(
                min(position + 1, local_window)
                for position in range(tokens_per_sequence)
            )
            * num_key_value_heads
        )
        local_kv_bytes = local_slots * head_dim * 2 * 2
        older_candidate_key_bytes = (
            int(index["candidate_slots_unique_kv_group"]) * head_dim * 2
        )
        older_selected_value_bytes = (
            int(index["selected_slots_unique_kv_group"]) * head_dim * 2
        )
        # One float score and one uint32 position per retained older entry.
        cache_metadata_bytes = (
            evaluated_layer_count
            * sequence_count
            * num_key_value_heads
            * retrieval_candidates
            * 8
        )
        total_streaming_bytes = (
            local_kv_bytes
            + older_candidate_key_bytes
            + older_selected_value_bytes
            + cache_metadata_bytes
        )
        streaming_traffic = {
            "local_kv_bytes": local_kv_bytes,
            "older_candidate_key_bytes": older_candidate_key_bytes,
            "older_selected_value_bytes": older_selected_value_bytes,
            "bounded_cache_metadata_bytes": cache_metadata_bytes,
            "total_logical_bytes": total_streaming_bytes,
            "fraction_of_dense_kv": total_streaming_bytes / dense_kv_bytes,
            "maximum_older_entries_per_head": retrieval_candidates,
            "sink_tokens": sink_tokens,
            "heavy_hitter_tokens": retrieval_candidates - sink_tokens,
            "exact_rerank_top_k": retrieval_top_k,
            "candidate_slots_unique_kv_group": index["candidate_slots_unique_kv_group"],
            "selected_slots_unique_kv_group": index["selected_slots_unique_kv_group"],
        }
    if "native_streaming" in results:
        index = results["native_streaming"]["index"]
        native_total = (
            int(index["local_kv_bytes"])
            + int(index["candidate_key_bytes"])
            + int(index["selected_value_bytes"])
        )
        native_dense = (
            evaluated_layer_count
            * sequence_count
            * dense_slots
            * num_attention_heads
            * head_dim
            * 2
            * 4
        )
        native_streaming_traffic = {
            **index,
            "dtype": "float32",
            "total_logical_read_bytes": native_total,
            "dense_query_head_logical_read_bytes": native_dense,
            "fraction_of_dense_query_head_reads": native_total / native_dense,
        }
    report = {
        "schema_version": 1,
        "experiment": "native_bitnet_milestone_3_attention_substitution",
        "status": (
            "frozen_bounded_attention_confirmation"
            if bounded_attention_confirmation_passed
            else (
                "frozen_semantic_confirmation"
                if sequence_count >= 8 and prediction_positions >= 256
                else "trained_model_development_evaluation"
            )
        ),
        "package": str(Path(package).resolve()),
        "dataset": {
            **dataset_evidence,
            "role": (
                "frozen_confirmation"
                if sequence_count >= 8 and prediction_positions >= 256
                else "development_attention_evaluation"
            ),
            "tokenizer_policy": "pinned_model_tokenizer_with_mistral_regex_fix",
            "sequences": sequence_count,
            "tokens_per_sequence": tokens_per_sequence,
            "prediction_positions": prediction_positions,
        },
        "configuration": {
            "native_packed_attention_projections": bool(native_projections),
            "layers": selected_layers,
            "modes": list(modes),
            "local_window": local_window,
            "recurrent_decay": recurrent_decay,
            "retrieval_top_k": retrieval_top_k,
            "older_weight": older_weight,
            "retrieval_candidates": retrieval_candidates,
            "lsh_tables": lsh_tables,
            "lsh_bits": lsh_bits,
            "lsh_radius": lsh_radius,
            "lsh_seed": lsh_seed,
            "page_size": page_size,
            "page_bound": page_bound,
            "sink_tokens": sink_tokens,
            "operator_contracts": {
                "local": "exact causal softmax over the bounded local window",
                "recurrent": "causal ELU+1 normalized recurrent linear attention",
                "retrieval": "local softmax plus separately normalized exact older top-k",
                "hybrid": "one exact sparse softmax over local keys and exact older top-k keys",
                "indexed_hybrid": (
                    "online multi-table sign-LSH postings, exact candidate "
                    "reranking, then one softmax over local and selected older keys"
                ),
                "bounded_hybrid": (
                    "exact page upper bounds with branch-and-bound top-k, "
                    "then one softmax over local and selected older keys"
                ),
                "streaming_hybrid": (
                    "exact local window plus fixed attention sinks and a "
                    "bounded online cumulative-attention heavy-hitter cache"
                ),
                "native_streaming": (
                    "stateful native C++ W/C/K sink and heavy-hitter cache "
                    "with exact candidate reranking"
                ),
            },
        },
        "head_analysis": head_analysis,
        "baseline": {
            "nll": float(baseline_nll.item()),
            "elapsed_seconds": baseline_seconds,
            "mlp_calls": len(baseline_calls),
        },
        "substitutions": results,
        "thresholds": thresholds,
        "semantic_checks": semantic_checks,
        "semantic_confirmation_passed": semantic_confirmation_passed,
        "bounded_attention_confirmation_passed": (
            bounded_attention_confirmation_passed
        ),
        "attention_traffic_model": {
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "bf16_bytes_per_element": 2,
            "evaluated_layers": evaluated_layer_count,
            "dense_visible_kv_slots": (
                evaluated_layer_count * sequence_count * dense_slots
            ),
            "hybrid_selected_kv_slots": sequence_count * selected_slots,
            "dense_kv_bytes": dense_kv_bytes,
            "exact_topk_key_scan_bytes": (
                evaluated_layer_count * exact_topk_key_scan_bytes
            ),
            "selected_value_bytes": evaluated_layer_count * selected_value_bytes,
            "current_exact_hybrid_bytes": evaluated_layer_count
            * (exact_topk_key_scan_bytes + selected_value_bytes),
            "current_exact_hybrid_fraction_of_dense_kv": (
                evaluated_layer_count
                * (exact_topk_key_scan_bytes + selected_value_bytes)
                / dense_kv_bytes
            ),
            "indexed_retrieval_implemented": (
                indexed_traffic is not None or bounded_traffic is not None
            ),
            "indexed_hybrid": indexed_traffic,
            "bounded_hybrid": bounded_traffic,
            "streaming_hybrid": streaming_traffic,
            "native_streaming": native_streaming_traffic,
            "hardware_dram_counter_measured": False,
        },
        "decision": (
            "milestone_3_bounded_attention_pass_native_optimization_next"
            if bounded_attention_confirmation_passed
            else (
                "semantic_progression_pass_systems_pending"
                if semantic_confirmation_passed
                else "measure_before_attention_distillation"
            )
        ),
        "scope_caveat": (
            (
                "The streaming hybrid retains a fixed local window, attention "
                "sinks, and bounded heavy-hitter cache; it never scans evicted "
                "keys. This establishes bounded trained-model attention quality, "
                "not native latency or hardware DRAM traffic. The current "
                "per-head cache implementation is a Python evaluation reference."
            )
            if {"streaming_hybrid", "native_streaming"} & set(results)
            else (
                "This run measures deterministic attention operators on the "
                "qualified MLP package. Scan-based exact retrieval cannot "
                "qualify the Milestone 3 systems gate without bounded retrieval "
                "and physical traffic evidence."
            )
        ),
    }
    atomic_json(Path(out), report)
    return report


__all__ = ["evaluate_native_bitnet_attention_substitution"]
