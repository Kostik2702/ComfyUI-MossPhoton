"""Shared helpers of the MP-9 integration tests (tiny 24-layer MOSS on CPU, no weights).

``run`` drives ``solver.execute`` the way the nodes do; ``stock_run`` is the node-level stock path
(``nodes.KSampler`` after the native ``SkipLayerGuidanceDiTSimple``); ``make_profile`` builds a
PHOTON_IDENTITY_PROFILE for a LoRA chain by hand from the contract fixture, WITHOUT importing the LoRA module (M4).
"""
import copy
import hashlib
import json
import os

from . import _comfy_env, tiny_moss

F32 = ("torch.float32",)
CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "contracts")
NODE = "MossPhotonSampler"


def read_contract(name):
    with open(os.path.join(CONTRACTS, name), "r", encoding="utf-8") as f:
        return json.load(f)


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_request(tiny, **kw):
    """A ``SampleRequest`` on the tiny model: reference preset, seed 42, 23/9-token prompts, no preview, float32
    reference dtype (the fixture is float32, the real reference dtype is bf16)."""
    from moss_photon.core import solver
    fields = dict(
        node_id=NODE, model=tiny.patcher, positive=tiny.make_cond(23), negative=tiny.make_cond(9),
        latent={"samples": tiny.latent.clone()}, seed=tiny_moss.SEED, preset_id="reference", cfg=3.0,
        preview=False, reference_dtypes=F32)
    fields.update(kw)
    return solver.SampleRequest(**fields)


def run(tiny, **kw):
    from moss_photon.core import solver
    return solver.execute(make_request(tiny, **kw))


def native_slg(patcher, layer=22, start=0.0, end=1.0):
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    return SkipLayerGuidanceDiTSimple.execute(
        patcher, start, end, double_layers="", single_layers=str(layer)).args[0]


def stock_run(tiny, patcher, *, positive, negative, steps, cfg=3.0, seed=tiny_moss.SEED, scheduler="sgm_uniform",
              latent=None, counter=None):
    """The stock node-level path: ``nodes.KSampler().sample`` (m6). Returns the final samples tensor.

    ``counter`` (a ``solver.ForwardStats``) is installed on a clone of ``patcher`` first, so forwards can be
    compared without changing the numbers."""
    import nodes
    from moss_photon.core import solver
    model = patcher
    if counter is not None:
        model = patcher.clone()
        solver.install_counter(model, counter)
    latent = latent if latent is not None else {"samples": tiny.latent.clone()}
    return nodes.KSampler().sample(model, seed, steps, cfg, "euler", scheduler, positive, negative, latent,
                                   denoise=1.0)[0]["samples"]


def make_lora_dict(patcher, skip=(), seed=11):
    import torch
    from tests.integration.test_adapter_cpu import make_lora
    return make_lora(patcher, torch, set(skip), seed)


def apply_lora(patcher, lora_sd, strength=1.0):
    """The native ``load_lora_for_models`` (what LoraLoaderModelOnly does)."""
    import comfy.sd
    return comfy.sd.load_lora_for_models(patcher, None, lora_sd, strength, 0)[0]


def entries(patcher):
    return sum(len(v) for v in patcher.patches.values())


def make_profile(model, *, base_model, role="character", baseline=0, name="synthetic.safetensors", strength=1.0,
                 attach=True):
    """Profile of ONE adapter whose attachment describes ``model`` (the MODEL a MossPhotonLoRAProfile node would
    return after applying it on ``base_model``); with ``attach`` the canonical string is stored on ``model``."""
    from moss_photon.core import canon, consts
    profile = copy.deepcopy(read_contract("identity_profile.json"))
    gained = entries(model) - entries(base_model)
    entry = profile["stack"][0]
    entry["role"] = role
    entry["lora"] = {"basename": name, "size": 1, "sha256": None, "metadata": {}}
    entry["strength_model"] = strength
    entry["patch_entries"] = gained
    entry["trigger_hints"] = []
    entry["trigger_present"] = False
    entry["coverage"]["ratio"] = 1.0
    entry["coverage"]["layers_missing"] = []
    profile["summary"].update({"count": 1, "roles": [role], "has_character": role == "character",
                               "patch_entries_total": entries(model)})
    profile["attachment"] = {
        "version": 1, "patches_uuid": str(model.patches_uuid), "patch_entries_total": entries(model),
        "baseline_entries": baseline,
        "entries": [{"basename": name, "size": 1, "sha256": None, "strength_model": strength, "strength_clip": 0.0,
                     "te": False, "patch_entries": gained}]}
    if attach:
        model.set_attachments(consts.ATTACH_LORA_STACK, canon.canonical_json(profile["attachment"]))
    return profile


# --------------------------------------------------------------------------- stand-ins for the slow or external parts
class NoSample:
    """Context manager: ``solver.build_guider`` returns a guider whose ``sample`` returns the latent unchanged.

    For tests whose subject is the PREFLIGHT and the LABEL (both computed before any forward): the model, its
    modifiers and patches are real, only the sampling steps are skipped."""

    def __enter__(self):
        from unittest import mock
        from moss_photon.core import solver

        class Guider:
            def sample(self, noise, latent_image, sampler, sigmas, callback=None, disable_pbar=False, seed=None):
                return latent_image.clone()

        self._patch = mock.patch.object(solver, "build_guider", lambda *a, **k: Guider())
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def known_sft():
    """Patch ``fingerprint.identify_checkpoint`` so the fixture counts as the verified SFT checkpoint."""
    from unittest import mock
    from moss_photon.core import consts, fingerprint
    family = fingerprint.KNOWN_CHECKPOINTS[consts.SFT_BF16_SHA256]["family"]
    ident = {"identity": {"sha256": consts.SFT_BF16_SHA256, "size": 10164655808,
                          "basename": "moss_image_sft_bf16.safetensors", "hash_source": "cache",
                          "classified_by": "known_table", "official": True, "variant": "bf16",
                          "verified_matrix": True}, "family": family, "warnings": []}
    return mock.patch.object(fingerprint, "identify_checkpoint", return_value=ident)


def retarget_profile(profile, model, *, baseline=0, attach=True):
    """Point a fixture profile's attachment at ``model`` (its live uuid and entry count) and store the canonical
    string on it, as the last MossPhotonLoRAProfile node of a chain would."""
    from moss_photon.core import canon, consts
    profile = copy.deepcopy(profile)
    profile["attachment"]["patches_uuid"] = str(model.patches_uuid)
    profile["attachment"]["patch_entries_total"] = entries(model)
    profile["attachment"]["baseline_entries"] = baseline
    profile["summary"]["patch_entries_total"] = entries(model)
    if attach:
        model.set_attachments(consts.ATTACH_LORA_STACK, canon.canonical_json(profile["attachment"]))
    return profile


def codes(report):
    return [w["code"] for w in report["warnings"]]


def errors_module():
    from moss_photon.core import errors
    return errors
