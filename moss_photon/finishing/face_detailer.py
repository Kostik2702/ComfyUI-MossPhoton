"""Identity face detailer pipeline (SPEC 4.2 item 3, 5 and 7; decisions from the MP-56 spike; audit N01-N05, R02).

Per face: size gate -> square crop -> ArcFace reading of the face in it -> redraw (the backend) -> LAB colour match on the
skin -> composite through a FACE-shaped feathered mask -> gates on the new crop -> accept rule. numpy only; the heavy parts
(detector, ArcFace, the diffusion redraw) are a ``Backend`` object, so the whole pipeline runs in unit tests with fakes.

Backend contract (duck typed)::

    backend.arcface_available            bool
    backend.detect(img)                  -> [(x0, y0, x1, y1, conf)]   img float32 [H, W, C] 0..1
    backend.read_faces(rgb_u8)           -> [FaceReading]              EVERY face of an [S, S, 3] uint8 crop
    backend.redraw(crop, seed)           -> float32 [S, S, 3]          same size as ``crop``, 0..1

Rules that never bend:

* outside the mask the result is bit-equal to the input (``geometry.composite_region`` copies, and the pipeline checks);
* with reference photos a redraw is kept only if the ArcFace score of THE SAME FACE rises by at least ``min_delta`` and the
  new crop passes the same visibility/pose gates as the old one, else the original crop stays;
* WITHOUT references nothing is redrawn unless the user opted in (``accept_without_references``); then the new crop still
  has to pass the gates and the redraw is marked ``unscorable``;
* a face that is tiny, in profile, occluded, not found by ArcFace, or bound ambiguously is not redrawn;
* nothing is ever picked as "best of N".
"""
import dataclasses
import time
from typing import NamedTuple, Optional

import numpy as np

from ..core import errors
from . import colormatch, geometry

NODE = geometry.NODE

_FACE_SEED_STRIDE = 7919      # per-face seed offset (prime), as in the QwenPhoton detailer
_IMAGE_SEED_STRIDE = 104729   # per-image seed offset (prime)

DECISIONS = ("accepted", "accepted_unscorable", "rejected", "skipped_large", "skipped_unscorable",
             "skipped_other_character", "skipped_ambiguous")
CHARACTER_LABELS = ("primary", "other_1", "other_2", "other_3")
ROLLBACKS = ("score_delta", "no_face_before", "no_face_after", "no_landmarks_after", "low_detector_confidence", "profile", "face_moved")
MAX_CHARACTERS = len(CHARACTER_LABELS)

_REASON_TEXT = {
    "too_small": "too small to redraw honestly",
    "no_face_in_crop": "ArcFace finds no face where the detector saw one (occluded or not a face)",
    "low_detector_confidence": "ArcFace detector confidence is low (occluded or unclear)",
    "no_landmarks": "ArcFace gave no landmarks, so pose and identity of this face cannot be tied together",
    "profile": "the head is turned away (profile)",
}


class FaceReading(NamedTuple):
    """One face as ArcFace sees it, in the coordinates of the crop it was read from.

    ``embedding`` is L2-normalised float32; ``bbox`` is ``(x0, y0, x1, y1)``; ``kps`` is the five insightface landmarks
    ``[5, 2]`` (eye, eye, nose, mouth, mouth) or ``None``. The box and the landmarks are what ties a "before" reading to
    the "after" reading of the SAME face."""

    embedding: "np.ndarray"
    det_score: float
    bbox: tuple
    kps: Optional["np.ndarray"]


@dataclasses.dataclass(frozen=True)
class Settings:
    """Everything the pass depends on; written to the report as is.

    The sampler values are the MP-56 result (turbo, 8 steps, cfg 1, denoise 0.25, 1.8x crop at 1024 px, 8 % feather).
    The thresholds below decide which faces the pass ATTEMPTS and when it ROLLS BACK; none is a calibrated measure of
    identity (SPEC 7). ``min_delta`` = 0.01: the ArcFace score of one and the same face, re-read after 1-3 px shifts, JPEG
    q85/q95 and +-3 % rescale of the crop, moved by std 0.003-0.004 and at most 0.012 (3 MP-56 faces, 14 perturbations
    each, 2026-10-25), so a gain below 0.01 cannot be told from re-measurement noise.
    """

    size_gate_px: float = 110.0
    crop_factor: float = 1.8
    work_px: int = 1024
    steps: int = 8
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    denoise: float = 0.25
    feather: float = 0.08
    mask_dilate: float = 1.15
    min_face_px: int = 48
    max_faces: int = 8
    det_conf: float = 0.45
    match_min: float = 0.30
    match_margin: float = 0.05
    min_delta: float = 0.01
    arcface_det_min: float = 0.5
    yaw_max: float = 0.6
    track_iou_min: float = 0.5
    kps_shift_max: float = 0.35
    color_match: bool = True
    accept_without_references: bool = False

    def to_dict(self, seed):
        d = dataclasses.asdict(self)
        d["seed"] = int(seed)
        return d


def to_u8(rgb):
    """Float 0..1 ``[..., 3]`` -> uint8 (only for the ArcFace reading; pixels written back are never quantised)."""
    return np.rint(np.clip(rgb[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def mean_similarity(embedding, references):
    """Mean cosine of a normalised ``embedding`` to the rows of ``references`` (``[n, d]``, rows normalised).

    This is the MP-56 / c3_identity ``sim_gt`` score: the mean over the reference photos, not a centroid.
    """
    return float(np.mean(references @ embedding))


# --------------------------------------------------------------------------- binding faces to characters (N01)
def best_assignment(scores, threshold):
    """Globally optimal one-to-one assignment of characters to faces.

    ``scores[i][j]`` = score of face i against character j. Every character takes at most one face and every face at
    most one character; a pair below ``threshold`` is never allowed (the character takes the dummy "nobody" instead).
    The sum of the used scores is maximised exactly (exhaustive over characters, at most ``MAX_CHARACTERS`` of them, so
    it is the Hungarian optimum without the machinery). Returns ``[character index or None]`` per face; ties keep the
    first found, which is deterministic."""
    n_faces = len(scores)
    n_chars = len(scores[0]) if n_faces else 0
    if n_chars > MAX_CHARACTERS:
        raise ValueError("at most %d characters" % MAX_CHARACTERS)
    best = {"total": -1.0, "owner": [None] * n_faces}

    def rec(c, used, total, owner):
        if c == n_chars:
            if total > best["total"] + 1e-12:
                best["total"], best["owner"] = total, list(owner)
            return
        rec(c + 1, used, total, owner)  # character c matches nobody
        for i in range(n_faces):
            if i not in used and scores[i][c] >= threshold:
                owner[i] = c
                rec(c + 1, used | {i}, total + float(scores[i][c]), owner)
                owner[i] = None

    rec(0, frozenset(), 0.0, [None] * n_faces)
    return best["owner"]


def assign_faces(scores, match_min, match_margin):
    """Bind faces to characters; returns ``(owners, status)`` per face.

    ``owners[i]`` is the character index or ``None``. ``status[i]`` is ``"bound"``, ``"unmatched"`` (no character takes
    this face above ``match_min``) or ``"ambiguous"`` (the optimum gives it a character, but a rival is within
    ``match_margin``: another character scores about as high on this face, or another face scores about as high on
    this character). An ambiguous face gets NO character: it is left untouched (audit N01)."""
    owners = best_assignment(scores, match_min)
    status = []
    for i, c in enumerate(owners):
        if c is None:
            status.append("unmatched")
            continue
        s = float(scores[i][c])
        row_rivals = [float(scores[i][k]) for k in range(len(scores[i])) if k != c]
        col_rivals = [float(scores[k][c]) for k in range(len(scores)) if k != i]
        if any(s - r < match_margin for r in row_rivals + col_rivals):
            status.append("ambiguous")
        else:
            status.append("bound")
    return [o if st == "bound" else None for o, st in zip(owners, status)], status


# --------------------------------------------------------------------------- readings and gates (N03)
def pick_reading(readings, expected_bbox, min_iou):
    """The reading whose box overlaps ``expected_bbox`` best (IoU at least ``min_iou``), else ``None``.

    Never "the largest face": the face the detector found is the face that is scored, before and after."""
    best, best_iou = None, 0.0
    for r in readings:
        iou = geometry.bbox_iou(r.bbox, expected_bbox)
        if iou > best_iou:
            best, best_iou = r, iou
    return best if best is not None and best_iou >= float(min_iou) else None


def gate_reason(reading, settings):
    """Why a reading is not good enough to score (``None`` = it is): not found, low detector confidence, no landmarks,
    or turned away. The same gates run on the crop before the redraw and on the crop after it."""
    if reading is None:
        return "no_face_in_crop"
    if reading.det_score < settings.arcface_det_min:
        return "low_detector_confidence"
    yaw = geometry.yaw_proxy(reading.kps)
    if yaw is None:
        return "no_landmarks"
    if yaw > settings.yaw_max:
        return "profile"
    return None


# --------------------------------------------------------------------------- the report record of one face
def _new_record(index, image_index, bbox, settings):
    short = geometry.short_side(bbox)
    return {
        "index": index, "image_index": image_index, "bbox": [round(float(v), 2) for v in bbox[:4]],
        "short_side_px": round(short, 2),
        "size_gate": {"threshold_px": float(settings.size_gate_px), "passed": short < float(settings.size_gate_px)},
        "decision": None, "reason": None, "rollback": None, "unscorable": False, "unscorable_reason": None,
        "character": None, "score_before": None, "score_after": None, "crop": None, "denoise": None, "steps": None,
        "seed": None, "color_match": None, "time_s": 0.0,
    }


def _issue(code, param, message, fix, **detail):
    return errors.make_issue(code, NODE, param, message, fix, detail=detail or None)


def _skip_unscorable(rec, issues, reason, text, fix="none needed: the face is left as it is (never faked)"):
    rec["decision"], rec["unscorable"], rec["unscorable_reason"], rec["reason"] = "skipped_unscorable", True, reason, text
    issues.append(_issue("unscorable_face", "image", text, fix, face=rec["index"], reason=reason))


def _skip_binding(rec, issues, decision, text):
    rec["decision"], rec["reason"] = decision, text
    issues.append(_issue("face_not_this_character", "reference_images", "face %d %s; left untouched" % (rec["index"], text),
                         "none needed; connect the right reference photos, or lower character_match_min if this IS the "
                         "character", face=rec["index"]))


def _check_image(img):
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] < 3 or img.dtype != np.float32:
        raise errors.OutputError(errors.make_issue(
            "output_error", NODE, "image", "expected a float32 image [H, W, C>=3], got %s"
            % (("%s %s" % (img.dtype, img.shape)) if isinstance(img, np.ndarray) else type(img).__name__),
            "connect an IMAGE [B,H,W,C]"))
    if not np.isfinite(img).all():
        raise errors.OutputError(errors.make_issue(
            "output_error", NODE, "image", "the image holds NaN or Inf (%d value(s))" % int((~np.isfinite(img)).sum()),
            "fix the upstream node that produced it; nothing was redrawn"))


def _analyse(img, detections, settings, backend, image_index, clock, issues):
    """Phase A: size gate, crop, and the ArcFace reading of the detector's face in it, on the UNCHANGED frame.

    Returns ``(records, work)``: one report record per face in detector order, and the entries still in play."""
    H, W = int(img.shape[0]), int(img.shape[1])
    records, work = [], []
    for i, det in enumerate(detections):
        t0 = clock()
        rec = _new_record(i, image_index, det[:4], settings)
        short = rec["short_side_px"]
        if short < settings.min_face_px:
            _skip_unscorable(rec, issues, "too_small", "face %d is %.0f px on its short side (< %d px): %s"
                             % (i, short, settings.min_face_px, _REASON_TEXT["too_small"]))
        elif short >= settings.size_gate_px:
            rec["decision"] = "skipped_large"
            rec["reason"] = "short side %.0f px >= size gate %.0f px" % (short, settings.size_gate_px)
            issues.append(_issue("face_large_skipped", "size_gate_px",
                                 "face %d is %.0f px on its short side, at or above the gate of %.0f px; left untouched"
                                 % (i, short, settings.size_gate_px),
                                 "raise size_gate_px to detail larger faces (MP-56: at 2 MP the pass lowered ArcFace)",
                                 face=i, short_side_px=round(short, 2)))
        else:
            box = geometry.crop_box(det[:4], settings.crop_factor, W, H)
            bx, by, side = box
            rec["crop"] = {"x": box[0], "y": box[1], "side": box[2]}
            face_box = (det[0] - bx, det[1] - by, det[2] - bx, det[3] - by)
            reading = None
            if backend.arcface_available:
                reading = pick_reading(backend.read_faces(to_u8(img[by:by + side, bx:bx + side])), face_box,
                                       settings.track_iou_min)
                reason = gate_reason(reading, settings)
                if reason is not None:
                    _skip_unscorable(rec, issues, reason, "face %d: %s; not redrawn" % (i, _REASON_TEXT[reason]))
            else:
                rec["unscorable"], rec["unscorable_reason"] = True, "arcface_unavailable"
            if rec["decision"] is None:
                work.append({"rec": rec, "box": box, "face_box": face_box, "reading": reading})
        rec["time_s"] = round(clock() - t0, 3)
        records.append(rec)
    return records, work


def _bind(work, characters, settings, issues):
    """Bind the faces still in play to the connected characters (global assignment with a margin); returns the entries
    bound to the PRIMARY character, the one the MODEL carries."""
    scores = [[mean_similarity(w["reading"].embedding, ref) for ref in characters] for w in work]
    owners, status = assign_faces(scores, settings.match_min, settings.match_margin)
    keep = []
    for w, row, owner, st in zip(work, scores, owners, status):
        rec = w["rec"]
        if st == "ambiguous":
            _skip_binding(rec, issues, "skipped_ambiguous",
                          "binds to a character only ambiguously (scores %s, margin %.2f needed)"
                          % ("/".join("%.3f" % v for v in row), settings.match_margin))
        elif owner is None:
            _skip_binding(rec, issues, "skipped_other_character",
                          "matches none of the connected characters (best score %.3f, cut-off %.2f)"
                          % (max(row), settings.match_min))
        elif owner != 0:
            rec["character"] = CHARACTER_LABELS[min(owner, MAX_CHARACTERS - 1)]
            _skip_binding(rec, issues, "skipped_other_character",
                          "is bound to %s, not to the character the MODEL carries" % rec["character"])
        else:
            rec["character"] = CHARACTER_LABELS[0]
            rec["score_before"] = round(row[0], 4)
            w["score_before"] = row[0]
            keep.append(w)
    return keep


def _after_gates(backend, cand, before, settings):
    """Gates on the NEW crop. Returns ``(rollback code or None, after reading or None)``: the same face must still be
    there (box overlap and landmark shift against the "before" reading), visible and not turned away."""
    after_all = backend.read_faces(to_u8(cand))
    after = pick_reading(after_all, before.bbox, settings.track_iou_min)
    reason = gate_reason(after, settings)
    if reason is not None:
        return ("no_face_after" if reason == "no_face_in_crop" else
                "no_landmarks_after" if reason == "no_landmarks" else reason), after
    if geometry.kps_shift(before.kps, after.kps) > settings.kps_shift_max:
        return "face_moved", after
    return None, after


def detail_image(img, settings, backend, characters, *, seed, image_index=0, clock=time.perf_counter):
    """Run the pass on ONE image ``[H, W, C]`` float32 (finite).

    ``characters`` is a list of reference-embedding matrices ``[n_photos, d]`` (rows L2-normalised; index 0 is the
    character the MODEL carries, the others only exist so their faces are left alone) or ``[]`` for "no references".
    Returns ``(out, faces, issues)``: ``out`` is ``img`` itself when nothing changed, else a new array; ``faces`` are
    the report face records (dicts, largest face first); ``issues`` are ``Issue`` objects for the report. ``img`` is never
    modified. Raises ``output_error`` for a non-finite or malformed image, or if a pixel outside the mask would differ
    (the result is discarded), and ``missing_guard_dependency`` if references are given but ArcFace is not available.
    """
    _check_image(img)
    scored = bool(characters)
    if scored and not backend.arcface_available:
        raise errors.MissingGuardDependency(errors.make_issue(
            "missing_guard_dependency", NODE, "reference_images",
            "reference photos need the ArcFace scorer (insightface), which is not available",
            "install insightface and the buffalo_l pack, or disconnect reference_images"))
    H, W = int(img.shape[0]), int(img.shape[1])
    issues = []
    detections = geometry.select_faces(backend.detect(img), max_faces=settings.max_faces, min_conf=settings.det_conf)
    if not detections:
        issues.append(_issue("no_face_found", "image", "image %d: no face at detector confidence >= %.2f"
                             % (image_index, settings.det_conf), "none needed: the image is returned unchanged",
                             image=image_index))
        return img, [], issues

    records, work = _analyse(img, detections, settings, backend, image_index, clock, issues)
    if scored:
        work = _bind(work, characters, settings, issues)
    elif not settings.accept_without_references:
        # N04: nothing to measure against -> pass-through. Accepting an unchecked redraw is an explicit opt-in.
        for w in work:
            _skip_unscorable(w["rec"], issues, "no_references",
                             "face %d: no reference photos, so a redraw could not be checked; left as it is"
                             % w["rec"]["index"],
                             "connect reference_images (and consent_reference_photos), or turn on accept_without_references")
        work = []

    # Phase B: redraw, colour match, composite, gates, accept rule; one face at a time on the evolving frame.
    cur = img
    touched = np.zeros((H, W), dtype=bool)
    full_mask = np.zeros((H, W), dtype=np.float32)
    for w in work:
        rec, (bx, by, side), face_box = w["rec"], w["box"], w["face_box"]
        t0 = clock()
        face_seed = (int(seed) + image_index * _IMAGE_SEED_STRIDE + rec["index"] * _FACE_SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF
        rec["seed"], rec["denoise"], rec["steps"] = face_seed, settings.denoise, settings.steps
        crop = cur[by:by + side, bx:bx + side, :3].copy()  # a copy: the backend may do anything to what it gets
        before = w["reading"]
        if before is not None and touched[by:by + side, bx:bx + side].any():
            # an earlier accepted face overlaps this crop: "before" must describe the crop as it is now
            before = pick_reading(backend.read_faces(to_u8(crop)), face_box, settings.track_iou_min)
            if scored:
                w["score_before"] = None if before is None else mean_similarity(before.embedding, characters[0])
                rec["score_before"] = None if before is None else round(w["score_before"], 4)
            if before is None:  # the neighbour's paste changed this crop so much that its face can no longer be read
                rec["decision"], rec["rollback"] = "rejected", "no_face_before"
                rec["reason"] = "the face can no longer be read after an overlapping face was pasted; the original crop is kept"
                issues.append(_issue("detailer_rejected", "reference_images", "face %d: %s" % (rec["index"], rec["reason"]),
                                     "none needed: the original crop was kept", face=rec["index"], rollback="no_face_before"))
                rec["time_s"] = round(rec["time_s"] + clock() - t0, 3)
                continue
        new = backend.redraw(crop, face_seed)
        if new.shape != crop.shape or not np.isfinite(new).all():
            raise errors.OutputError(errors.make_issue(
                "output_error", NODE, "image", "face %d: the redraw returned %s (expected finite %s)"
                % (rec["index"], getattr(new, "shape", None), crop.shape),
                "report this as a bug of the redraw backend"))
        new = np.clip(new.astype(np.float32, copy=False), 0.0, 1.0)
        base_mask = geometry.ellipse_mask(side, face_box, settings.mask_dilate, settings.feather)
        if settings.color_match:
            skin = geometry.skin_mask(side, face_box, None if before is None else before.kps)
            new, rec["color_match"] = colormatch.lab_match(new, crop, skin)
        mask = base_mask * colormatch.seam_factor(crop, new, base_mask)
        region = cur[by:by + side, bx:bx + side]
        full_new = region.copy()
        full_new[..., :3] = new
        cand = geometry.composite_region(region, full_new, mask)

        rollback, after = None, None
        if backend.arcface_available and before is not None:
            rollback, after = _after_gates(backend, cand, before, settings)
        score_after = None if after is None or not scored else mean_similarity(after.embedding, characters[0])
        rec["score_after"] = None if score_after is None else round(score_after, 4)
        if rollback is None and scored:
            sb = w.get("score_before")
            if sb is None or score_after is None or score_after - sb < settings.min_delta:
                rollback = "score_delta"
        if rollback is not None:
            rec["decision"], rec["rollback"] = "rejected", rollback
            rec["reason"] = _rollback_text(rollback, w.get("score_before"), score_after, settings)
            issues.append(_issue("detailer_rejected", "reference_images", "face %d: %s" % (rec["index"], rec["reason"]),
                                 "none needed: the original crop was kept", face=rec["index"], rollback=rollback))
            accept = False
        elif scored:
            accept = True
            rec["decision"] = "accepted"
            rec["reason"] = "ArcFace score rose %.4f -> %.4f (needed +%.2f), same face, gates passed" % (
                w["score_before"], score_after, settings.min_delta)
        else:
            accept = True
            rec["decision"] = "accepted_unscorable"
            rec["unscorable"] = True
            rec["unscorable_reason"] = rec["unscorable_reason"] or "no_references"
            rec["reason"] = "kept without an identity score (%s, accept_without_references)" % rec["unscorable_reason"]
            issues.append(_issue("unscorable_face", "reference_images",
                                 "face %d was redrawn and kept without an identity score (%s)"
                                 % (rec["index"], rec["unscorable_reason"]),
                                 "connect reference photos and turn on consent_reference_photos to enable the accept rule",
                                 face=rec["index"], reason=rec["unscorable_reason"]))
        if accept:
            if cur is img:
                cur = img.copy()
            cur[by:by + side, bx:bx + side] = cand
            touched[by:by + side, bx:bx + side] |= mask > 0.0
            full_mask[by:by + side, bx:bx + side] = np.maximum(full_mask[by:by + side, bx:bx + side], base_mask)
        rec["time_s"] = round(rec["time_s"] + clock() - t0, 3)

    geometry.assert_outside_mask_equal(img, cur, full_mask)
    return cur, records, issues


def _rollback_text(code, before, after, settings):
    if code == "score_delta":
        if before is None or after is None:
            return "no usable ArcFace score before/after; the original crop is kept"
        return ("ArcFace score %.4f -> %.4f did not rise by the required %.2f; the original crop is kept"
                % (before, after, settings.min_delta))
    return "the redrawn crop failed a gate (%s); the original crop is kept" % code


SECTION_VERSION = 1


def build_section(settings, seed, *, images, faces, issues, detector, arcface, references, prompt_sha256,
                  trigger_used, time_s, recipe):
    """The report's ``detailer`` section (validated by ``report.with_detailer``). Pure; JSON-safe values only.

    ``detector`` is ``{"available", "model"}``, ``arcface`` is ``{"available", "pack"}``, ``references`` is
    ``{"connected", "consent", "used", "n_images", "n_with_face", "n_characters"}``, ``recipe`` is
    ``recipe_block`` of this run. No prompt text, no path and no reference photo is written (R5): the prompt is its
    sha256 only.
    """
    decisions = [f["decision"] for f in faces]
    return {
        "version": SECTION_VERSION,
        "images": int(images),
        "settings": settings.to_dict(seed),
        "dependencies": {"detector": dict(detector), "arcface": dict(arcface)},
        "references": dict(references),
        "prompt": {"sha256": prompt_sha256, "trigger_used": bool(trigger_used)},
        "recipe": dict(recipe),
        "faces": list(faces),
        "counts": {
            "faces": len(faces),
            "accepted": sum(d in ("accepted", "accepted_unscorable") for d in decisions),
            "rejected": decisions.count("rejected"),
            "skipped": sum(d.startswith("skipped") for d in decisions),
            "unscorable": sum(bool(f["unscorable"]) for f in faces),
        },
        "time_s": round(float(time_s), 3),
        "warnings": [i.to_dict() for i in issues],
    }


def recipe_block(settings, seed, *, model, lora, vae_sha256, input_sha256, prompt_sha256, reference_sha256):
    """The detailer's own recipe (audit F05): what, together with the input pixels, determines the output pixels.

    Separate from the sampler's ``recipe_hash`` (which describes the picture BEFORE this node). ``model`` is the
    checkpoint identity ``{"sha256", "basename", "size"}``, ``lora`` the adapter list from the identity profile, ``vae_sha256``
    the VAE weights hash or ``None``, ``input_sha256`` one hash per input image, ``reference_sha256`` a hash of the
    reference EMBEDDINGS (never of the photos) or ``None``. Nothing private is written."""
    from ..core import canon
    material = {"settings": settings.to_dict(seed), "model": dict(model), "lora": [dict(x) for x in lora],
                "vae_sha256": vae_sha256, "input_sha256": list(input_sha256), "prompt_sha256": prompt_sha256,
                "reference_sha256": reference_sha256}
    out = dict(material)
    out["recipe_hash"] = canon.sha256_hex(canon.canonical_json(material))
    return out


def reference_matrix(backend, photos_u8):
    """``(matrix [n, d] | None, n_with_face)``: ArcFace embeddings of the LARGEST face of each reference photo.

    ``photos_u8`` are ``[H, W, 3]`` uint8 arrays. Photos without a face are skipped (and counted by the difference
    between ``len(photos_u8)`` and ``n_with_face``); the rows are L2-normalised. The photos are only read.
    """
    rows = []
    for photo in photos_u8:
        readings = backend.read_faces(photo)
        if readings:
            largest = max(readings, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
            rows.append(np.asarray(largest.embedding, dtype=np.float32))
    if not rows:
        return None, 0
    return np.stack(rows), len(rows)
