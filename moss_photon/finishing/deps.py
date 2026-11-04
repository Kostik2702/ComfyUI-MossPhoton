"""Optional dependencies of MossPhotonIdentityDetailer (SPEC 4 and 8): what is installed, and the clear message when not.

Stdlib only and cheap: nothing heavy is imported here (``importlib.util.find_spec`` looks without importing), so the
pack loads, and the node module can be imported, without ultralytics or insightface.

Two dependencies, with different weight:

* the face DETECTOR (``ultralytics`` + a YOLO face model file) is required: without it there is no face to find, and
  the node refuses to run with ``missing_guard_dependency`` (its message names what to install);
* the ArcFace SCORER (``insightface`` + the ``buffalo_l`` pack) is required only for what needs it: reference photos,
  the accept rule, ``unscorable`` detection. A run without it works with every face marked unscorable.
"""
import importlib.util
import os

DETECTOR_PACKAGE = "ultralytics"
DETECTOR_MODEL = "face_yolov8m.pt"
ARCFACE_PACKAGE = "insightface"
ARCFACE_PACK = "buffalo_l"

DETECTOR_FIX = ("pip install ultralytics into the ComfyUI python and put %s into models/ultralytics/bbox/ "
                "(https://huggingface.co/Bingsu/adetailer)" % DETECTOR_MODEL)
ARCFACE_FIX = ("pip install insightface onnxruntime into the ComfyUI python and put the %s pack into "
               "models/insightface/models/%s/" % (ARCFACE_PACK, ARCFACE_PACK))


class Availability:
    """What ``probe`` found. ``*_reason`` is ``None`` when the dependency is usable, else one line saying what is missing."""

    __slots__ = ("detector_reason", "detector_path", "arcface_reason", "arcface_root")

    def __init__(self, detector_reason, detector_path, arcface_reason, arcface_root):
        self.detector_reason = detector_reason
        self.detector_path = detector_path
        self.arcface_reason = arcface_reason
        self.arcface_root = arcface_root

    @property
    def detector_ok(self):
        return self.detector_reason is None

    @property
    def arcface_ok(self):
        return self.arcface_reason is None

    def describe(self):
        """One line for a node description or a log."""
        parts = ["detector " + ("ready (%s)" % DETECTOR_MODEL if self.detector_ok else "MISSING: " + self.detector_reason),
                 "ArcFace " + ("ready (%s)" % ARCFACE_PACK if self.arcface_ok else "missing: " + self.arcface_reason)]
        return "; ".join(parts)


def _models_dir():
    """ComfyUI's models directory, or ``None`` outside ComfyUI (``folder_paths`` is a top-level ComfyUI module)."""
    try:
        import folder_paths
        return folder_paths.models_dir
    except Exception:  # noqa: BLE001 - not running inside ComfyUI
        return None


def _registered_bbox_dirs():
    try:
        import folder_paths
        return list(folder_paths.get_folder_paths("ultralytics_bbox"))
    except Exception:  # noqa: BLE001 - the Impact Subpack folder type is optional
        return []


def probe(models_dir=None, find_spec=importlib.util.find_spec, registered_bbox_dirs=None):
    """Look for the two dependencies. ``models_dir`` defaults to ComfyUI's; ``find_spec`` is injectable for tests.

    Never raises and never imports ultralytics, insightface or onnxruntime.
    """
    if models_dir is None:
        models_dir = _models_dir()
    if registered_bbox_dirs is None:
        registered_bbox_dirs = _registered_bbox_dirs()

    detector_reason, detector_path = None, None
    if find_spec(DETECTOR_PACKAGE) is None:
        detector_reason = "python package %s is not installed" % DETECTOR_PACKAGE
    elif models_dir is None:
        detector_reason = "the ComfyUI models folder is unknown, so %s cannot be located" % DETECTOR_MODEL
    else:
        candidates = list(registered_bbox_dirs) + [
            os.path.join(models_dir, "ultralytics", "bbox"), os.path.join(models_dir, "ultralytics")]
        for folder in candidates:
            path = os.path.join(folder, DETECTOR_MODEL)
            if os.path.isfile(path):
                detector_path = path
                break
        if detector_path is None:
            detector_reason = "face model %s not found in models/ultralytics/bbox" % DETECTOR_MODEL

    arcface_reason, arcface_root = None, None
    if find_spec(ARCFACE_PACKAGE) is None:
        arcface_reason = "python package %s is not installed" % ARCFACE_PACKAGE
    elif find_spec("onnxruntime") is None:
        arcface_reason = "python package onnxruntime is not installed"
    elif models_dir is None:
        arcface_reason = "the ComfyUI models folder is unknown, so the %s pack cannot be located" % ARCFACE_PACK
    else:
        root = os.path.join(models_dir, "insightface")
        if os.path.isdir(os.path.join(root, "models", ARCFACE_PACK)):
            arcface_root = root
        else:
            arcface_reason = "pack %s not found in models/insightface/models" % ARCFACE_PACK
    return Availability(detector_reason, detector_path, arcface_reason, arcface_root)
