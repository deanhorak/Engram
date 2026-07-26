"""Cache-line accounting for a practical native-BitNet DIP layout.

The practical layout is deliberately dual:

* the DIP index duplicates gate/up in coordinate-major order, with every
  coordinate row padded to a cache-line boundary; and
* the existing native-BitNet artifact remains record-major for exact
  candidate completion and selected down-record reads.

The duplication is not hidden.  It makes dynamic top-input scans sequential
and candidate completion deterministic: every selected coordinate row and
every candidate record occupies an integral number of cache lines.  The
functions below report both serialized package bytes and a cold, one-use
cache-line schedule.  They do not claim measured hardware DRAM traffic.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from engram.models.native_bitnet import native_bitnet_repack_traffic


_TRITS_PER_BYTE = 5
_BF16_BYTES = 2
_DOWN_NORM_BYTES = 2
_INDEX_HEADER_BYTES = 128
_INDEX_DIRECTORY_ENTRY_BYTES = 32
_INDEX_LAYER_HEADER_BYTES = 128


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def native_bitnet_dip_physical_accounting(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_counts: Sequence[int],
    candidate_counts: Sequence[int],
    top_ks: Sequence[int],
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Return serialized-layout and cold cache-line bytes for a DIP schedule.

    ``input_counts``, ``candidate_counts``, and ``top_ks`` contain one value
    per layer.  Candidate gate/up completion is charged as a full
    record-major read even though only omitted input coordinates are logically
    needed.  This is the cache-line-honest consequence of dynamic input
    selection.  Down is read only for the final ``top_k`` records.

    The current native artifact has no padding between individual records.
    This proof therefore fails closed unless one packed record is itself an
    integral number of cache lines.  The pinned 2B BitNet dimensions satisfy
    that contract: 2,560 trits occupy exactly 512 bytes, or eight 64-byte
    cache lines.
    """

    hidden = _positive_integer(hidden_size, "hidden_size")
    intermediate = _positive_integer(intermediate_size, "intermediate_size")
    cache_line = _positive_integer(cache_line_bytes, "cache_line_bytes")
    if cache_line < 64 or cache_line % 64:
        raise ValueError("cache_line_bytes must be a positive multiple of 64")
    inputs = tuple(input_counts)
    candidates = tuple(candidate_counts)
    selected = tuple(top_ks)
    if not inputs or len(inputs) != len(candidates) or len(inputs) != len(selected):
        raise ValueError("DIP schedules must be non-empty and have equal lengths")
    for name, values, maximum in (
        ("input_counts", inputs, hidden),
        ("candidate_counts", candidates, intermediate),
        ("top_ks", selected, intermediate),
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > maximum
            for value in values
        ):
            raise ValueError(f"{name} values must be positive and within width")
    if any(top_k > candidate for top_k, candidate in zip(selected, candidates)):
        raise ValueError("top_ks must not exceed candidate_counts")

    layer_count = len(inputs)
    record_payload_bytes = math.ceil(hidden / _TRITS_PER_BYTE)
    if record_payload_bytes % cache_line:
        raise ValueError(
            "packed native-BitNet record width must be cache-line aligned "
            "for schedule-independent physical accounting"
        )
    coordinate_payload_bytes = math.ceil(intermediate / _TRITS_PER_BYTE)
    coordinate_stride_bytes = _align(coordinate_payload_bytes, cache_line)
    norm_value_bytes = (
        _DOWN_NORM_BYTES if hidden <= (1 << (8 * _DOWN_NORM_BYTES)) - 1 else 4
    )

    base = native_bitnet_repack_traffic(
        hidden,
        intermediate,
        layer_count=layer_count,
        cache_line_bytes=cache_line,
    )
    index_header_block_bytes = _align(_INDEX_HEADER_BYTES, cache_line)
    index_directory_payload_bytes = layer_count * _INDEX_DIRECTORY_ENTRY_BYTES
    index_directory_block_bytes = _align(index_directory_payload_bytes, cache_line)
    coordinate_stream_bytes_per_projection = hidden * coordinate_stride_bytes
    down_norm_payload_bytes_per_layer = intermediate * norm_value_bytes
    down_norm_stream_bytes_per_layer = _align(
        down_norm_payload_bytes_per_layer, cache_line
    )
    index_layer_payload_bytes = (
        _INDEX_LAYER_HEADER_BYTES
        + 2 * coordinate_stream_bytes_per_projection
        + down_norm_stream_bytes_per_layer
    )
    index_layer_block_bytes = _align(index_layer_payload_bytes, cache_line)
    index_layer_tail_padding_bytes = index_layer_block_bytes - index_layer_payload_bytes
    serialized_index_bytes = (
        index_header_block_bytes
        + index_directory_block_bytes
        + layer_count * index_layer_block_bytes
    )

    coordinate_gate_up_payload_bytes_per_layer = 2 * hidden * coordinate_payload_bytes
    coordinate_gate_up_row_padding_bytes_per_layer = (
        2 * hidden * (coordinate_stride_bytes - coordinate_payload_bytes)
    )
    base_gate_up_bytes_per_layer = 2 * intermediate * record_payload_bytes
    combined_serialized_bytes = (
        int(base["serialized_artifact_bytes"]) + serialized_index_bytes
    )
    dense_q4_bytes = int(base["dense_q4_source_mlp_bytes"])

    layer_reports: list[dict[str, Any]] = []
    layer_cold_bytes = 0
    for layer, (input_count, candidate_count, top_k) in enumerate(
        zip(inputs, candidates, selected)
    ):
        partial_coordinate_scan_bytes = 2 * input_count * coordinate_stride_bytes
        partial_coordinate_payload_bytes = 2 * input_count * coordinate_payload_bytes
        partial_coordinate_padding_bytes = (
            partial_coordinate_scan_bytes - partial_coordinate_payload_bytes
        )
        candidate_completion_bytes = (
            2 * candidate_count * record_payload_bytes if input_count < hidden else 0
        )
        selected_down_bytes = top_k * record_payload_bytes
        gain_scan_bytes = _align(intermediate * _BF16_BYTES, cache_line)
        down_norm_scan_bytes = down_norm_stream_bytes_per_layer

        # Two base-artifact lines cover its layer header and projection-scale
        # line; two lines cover the authenticated DIP policy/index header.
        layer_header_and_scale_bytes = 4 * cache_line
        complete = (
            partial_coordinate_scan_bytes
            + candidate_completion_bytes
            + selected_down_bytes
            + gain_scan_bytes
            + down_norm_scan_bytes
            + layer_header_and_scale_bytes
        )
        layer_cold_bytes += complete
        dense_layer_bytes = (3 * hidden * intermediate + 1) // 2
        omitted_payload_lower_bound = (
            2 * candidate_count * math.ceil((hidden - input_count) / _TRITS_PER_BYTE)
        )
        layer_reports.append(
            {
                "layer": layer,
                "input_coordinates": input_count,
                "candidate_count": candidate_count,
                "top_k": top_k,
                "partial_coordinate_scan_bytes": partial_coordinate_scan_bytes,
                "partial_coordinate_payload_bytes": (partial_coordinate_payload_bytes),
                "partial_coordinate_padding_bytes": (partial_coordinate_padding_bytes),
                "candidate_completion_record_bytes": (candidate_completion_bytes),
                "candidate_completion_omitted_payload_lower_bound_bytes": (
                    omitted_payload_lower_bound
                ),
                "candidate_completion_amplification_over_omitted_payload": (
                    candidate_completion_bytes / omitted_payload_lower_bound
                    if omitted_payload_lower_bound
                    else 1.0
                ),
                "gate_up_trits_duplicated_between_partial_and_completion": (
                    2 * candidate_count * input_count
                    if candidate_completion_bytes
                    else 0
                ),
                "selected_down_record_bytes": selected_down_bytes,
                "gain_scan_bytes": gain_scan_bytes,
                "down_norm_scan_bytes": down_norm_scan_bytes,
                "layer_header_and_scale_bytes": layer_header_and_scale_bytes,
                "complete_modelled_cold_bytes": complete,
                "dense_q4_bytes": dense_layer_bytes,
                "fraction_of_dense_q4": complete / dense_layer_bytes,
            }
        )

    base_global_metadata_bytes = int(base["header_cache_aligned_bytes"]) + int(
        base["directory_cache_aligned_bytes"]
    )
    index_global_metadata_bytes = index_header_block_bytes + index_directory_block_bytes
    global_metadata_cold_bytes = (
        base_global_metadata_bytes + index_global_metadata_bytes
    )
    complete_cold_bytes = layer_cold_bytes + global_metadata_cold_bytes
    fraction_of_dense_q4 = complete_cold_bytes / dense_q4_bytes

    return {
        # Version 2 is source-bound by the 128-byte index header and embeds
        # the complete per-layer RMS policy in each 128-byte layer header.
        # Reports produced before that layout change used a misleading v1
        # label (and, in some cases, 64-byte headers) and must not qualify as
        # frozen physical evidence.
        "format": "native_bitnet_dip_dual_layout_v2",
        "layout": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "layer_count": layer_count,
            "cache_line_bytes": cache_line,
            "trits_per_byte": _TRITS_PER_BYTE,
            "base_record_payload_bytes": record_payload_bytes,
            "base_record_cache_lines": record_payload_bytes // cache_line,
            "coordinate_payload_bytes": coordinate_payload_bytes,
            "coordinate_stride_bytes": coordinate_stride_bytes,
            "coordinate_cache_lines": coordinate_stride_bytes // cache_line,
            "coordinate_row_padding_bytes": (
                coordinate_stride_bytes - coordinate_payload_bytes
            ),
            "index_header_bytes": _INDEX_HEADER_BYTES,
            "index_header_block_bytes": index_header_block_bytes,
            "index_directory_entry_bytes": _INDEX_DIRECTORY_ENTRY_BYTES,
            "index_directory_payload_bytes": index_directory_payload_bytes,
            "index_directory_block_bytes": index_directory_block_bytes,
            "index_layer_header_bytes": _INDEX_LAYER_HEADER_BYTES,
            "coordinate_stream_bytes_per_projection": (
                coordinate_stream_bytes_per_projection
            ),
            "down_norm_value_bytes": norm_value_bytes,
            "down_norm_payload_bytes_per_layer": (down_norm_payload_bytes_per_layer),
            "down_norm_stream_bytes_per_layer": (down_norm_stream_bytes_per_layer),
            "index_layer_payload_bytes": index_layer_payload_bytes,
            "index_layer_tail_padding_bytes": index_layer_tail_padding_bytes,
            "index_layer_block_bytes": index_layer_block_bytes,
        },
        "serialization": {
            "base_record_artifact_bytes": int(base["serialized_artifact_bytes"]),
            "coordinate_index_bytes": serialized_index_bytes,
            "combined_serialized_bytes": combined_serialized_bytes,
            "dense_q4_bytes": dense_q4_bytes,
            "combined_fraction_of_dense_q4": (
                combined_serialized_bytes / dense_q4_bytes
            ),
            "base_gate_up_bytes_per_layer": base_gate_up_bytes_per_layer,
            "coordinate_gate_up_payload_bytes_per_layer": (
                coordinate_gate_up_payload_bytes_per_layer
            ),
            "coordinate_gate_up_row_padding_bytes_per_layer": (
                coordinate_gate_up_row_padding_bytes_per_layer
            ),
            "gate_up_values_are_duplicated": True,
            "duplication_reason": (
                "coordinate-major rows make arbitrary top-input scans "
                "sequential; record-major rows make arbitrary candidate "
                "completion cache-line deterministic"
            ),
        },
        "traffic": {
            "layers": layer_reports,
            "base_global_header_directory_bytes": (base_global_metadata_bytes),
            "index_global_header_directory_bytes": (index_global_metadata_bytes),
            "global_header_directory_bytes": global_metadata_cold_bytes,
            "layer_cold_bytes": layer_cold_bytes,
            "complete_modelled_cold_bytes": complete_cold_bytes,
            "dense_q4_bytes": dense_q4_bytes,
            "fraction_of_dense_q4": fraction_of_dense_q4,
            "passes_45_percent": fraction_of_dense_q4 <= 0.45,
            "accounting_policy": (
                "charge every selected coordinate-row cache line, complete "
                "record-major gate/up rows for every candidate, down rows only "
                "for final top-K, full gain/down-norm scans, and serialized "
                "headers/directories once; no selected-weight cache reuse"
            ),
            "measured_hardware_dram_bytes": False,
            "excluded": (
                "activation and candidate scratch, sorting instructions, "
                "allocator/runtime metadata, and hardware prefetch effects"
            ),
        },
    }


__all__ = ["native_bitnet_dip_physical_accounting"]
