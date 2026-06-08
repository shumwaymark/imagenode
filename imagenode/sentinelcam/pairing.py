"""Host-side crop/meta pairing for the OAK A2 pipeline.

PROMOTED 2026-06-06 from `dev/oak_harness/pairing.py` (verbatim — the module was
written depthai-free precisely so it graduates unchanged, per its original
docstring). This is the production-bound copy for the Phase-3 outpost intake;
production home `sentinelcam/pairing.py`. The harness keeps its own identical copy
for the benchmark rig.

The device emits two parallel streams for every kept detection: a crop image
(through the crop ImageManip) and a metadata `Buffer` sidecar (which bypasses
the manip). The manip is LOSSY — it can drop a crop via the
`setMaxOutputFrameSize` guard (Hard-Won Rule 2) — while the meta path is not.
So the host must pair the two streams by a correlation key, never by FIFO
arrival order, and must treat an orphaned meta (a meta whose crop was dropped)
as a first-class drop metric.

This module is deliberately free of any `depthai` or `oak_camera` import: it
operates on plain integers and a structurally-typed meta. That keeps it
unit-testable without hardware and lets it graduate unchanged into the outpost
intake loop (Phase 3.2) and the EventSampler (Phase 3.5).

Two pairing modes, identical machinery, only the key function differs:

  SEQNUM     key = seqnum.  This is the historically-shipped behavior and it is
             KNOWN-BROKEN for multi-detection frames: when one frame emits
             several crops they all carry one identical seqnum, so a crop
             dropped mid-frame leaves an orphaned meta with the SAME key as the
             survivor. The resync's `key < crop_key` test is `N < N` (false),
             the survivor mispairs FIFO with the wrong detection's meta, and the
             real orphan only surfaces a frame later. Confirmed live in
             Retest_June1.txt. Retained here ONLY so the harness can reproduce
             and quantify the bug for A/B against the fix.

  COMPOSITE  key = seqnum * COMPOSITE_SEQ_STRIDE + det_idx.  The device stamps
             this composite onto each crop frame's sequence number; the host
             reconstructs the same key from the meta. Now every crop of a frame
             has a UNIQUE key, so a mid-frame drop shows up as a true gap that
             the resync discards correctly. This is the fix (§ 3.2).
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Protocol


# Must match the device Script's stamping factor in
# sentinelcam.oak_camera._build_script_source. The plan (§ 3.2) specifies
# seqnum*100+det_idx; 100 comfortably exceeds the per-frame detection count of
# MobileNet-SSD and keeps the composite human-readable in logs (the low two
# decimal digits are the det_idx).
COMPOSITE_SEQ_STRIDE = 100


class MetaLike(Protocol):
    """Structural type for the crop metadata sidecar.

    `oak_camera.CropMeta` satisfies this without importing it here. Pairing only
    needs the correlation fields (`seqnum`, `det_idx`); the rest is the contract
    downstream consumers rely on after pairing — `label` for demux, and the
    `class_name`/`label_name` helpers the harness and intake loop use directly.
    """

    seqnum: int
    det_idx: int
    label: int

    @property
    def class_name(self) -> Optional[str]: ...

    @property
    def label_name(self) -> str: ...


class PairMode(enum.Enum):
    SEQNUM = "seqnum"
    COMPOSITE = "composite"


@dataclass
class PairStats:
    """Cumulative pairing outcomes — every counter is a real, nameable event."""

    metas_in: int = 0           # metas pushed from q_meta
    crops_in: int = 0           # crops offered from q_crops
    paired: int = 0             # crop matched to its meta
    orphaned_meta: int = 0      # meta discarded because its own crop was dropped
    crop_no_meta: int = 0       # crop arrived with no matching meta (reverse desync)

    def as_dict(self) -> dict:
        return {
            "metas_in": self.metas_in,
            "crops_in": self.crops_in,
            "paired": self.paired,
            "orphaned_meta": self.orphaned_meta,
            "crop_no_meta": self.crop_no_meta,
        }


def _seqnum_key(meta: MetaLike) -> int:
    return meta.seqnum


def _composite_key(meta: MetaLike) -> int:
    return meta.seqnum * COMPOSITE_SEQ_STRIDE + meta.det_idx


class CropPairer:
    """Pairs crops to metas by a monotonic correlation key.

    Usage per intake tick: drain `q_meta` into `push_meta()` FIRST (the device
    sends meta before img+cfg and XLink preserves per-queue order, so a paired
    crop always finds its meta already queued), then for each crop call
    `pair(crop_key)` where `crop_key = crop_frame.getSequenceNum()`.

    The resync is the same in both modes; only `_key` differs. Metas are held in
    arrival order, which for a correct device is also key order — the resync
    relies on that monotonicity to discard orphans.
    """

    def __init__(self, mode: PairMode = PairMode.COMPOSITE) -> None:
        self.mode = mode
        self._key: Callable[[MetaLike], int] = (
            _composite_key if mode is PairMode.COMPOSITE else _seqnum_key
        )
        self._metas: Deque[MetaLike] = deque()
        self.stats = PairStats()

    def push_meta(self, meta: MetaLike) -> None:
        self._metas.append(meta)
        self.stats.metas_in += 1

    def pair(self, crop_key: int) -> Optional[MetaLike]:
        """Return the meta matching this crop, or None if the crop is unpaired.

        Discards (and counts) any queued meta whose key precedes the crop's: that
        meta's own crop was dropped at the manip guard. A None return means the
        reverse desync — a crop whose meta never arrived — which should not happen
        because meta bypasses the lossy manip; it is counted as `crop_no_meta`.
        """
        self.stats.crops_in += 1

        while self._metas and self._key(self._metas[0]) < crop_key:
            self._metas.popleft()
            self.stats.orphaned_meta += 1

        if not self._metas:
            self.stats.crop_no_meta += 1
            return None

        if self._key(self._metas[0]) > crop_key:
            # Head meta belongs to a later crop; this crop's own meta never
            # arrived. Skip the crop without consuming a meta it doesn't own.
            self.stats.crop_no_meta += 1
            return None

        meta = self._metas.popleft()
        self.stats.paired += 1
        return meta

    def flush_orphans(self) -> int:
        """Discard any metas still queued at shutdown (their crops never came).

        Returns the count, also folded into `orphaned_meta`. Call once when the
        run ends so end-of-stream stragglers are accounted, not silently dropped.
        """
        n = len(self._metas)
        self._metas.clear()
        self.stats.orphaned_meta += n
        return n

    @property
    def pending_metas(self) -> int:
        return len(self._metas)


def decode_composite(crop_key: int) -> tuple[int, int]:
    """Split a composite crop key back into (seqnum, det_idx)."""
    return divmod(crop_key, COMPOSITE_SEQ_STRIDE)


def make_composite(seqnum: int, det_idx: int) -> int:
    """Build the composite key the device stamps onto a crop frame."""
    return seqnum * COMPOSITE_SEQ_STRIDE + det_idx
