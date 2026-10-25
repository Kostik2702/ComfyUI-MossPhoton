"""R4: the sampler never mutates its inputs -- MODEL (patches, options, object patches, attachments), CONDITIONING, the
LATENT dict and tensor, the identity profile and the constraints -- on the success path and on a failure path."""
import copy
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


def snapshot(model):
    return {
        "patches": {k: [id(item) for item in v] for k, v in model.patches.items()},
        "patches_uuid": str(model.patches_uuid),
        "model_options": copy.deepcopy(model.model_options),
        "object_patches": dict(model.object_patches),
        "attachments": dict(model.attachments),
        "wrappers": copy.deepcopy(model.wrappers),
        "callbacks": copy.deepcopy(model.callbacks),
        "weights": {k: v.detach().clone() for k, v in list(model.model.state_dict().items())[:8]},
    }


class InputNotMutatedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        cls.torch = torch

    def assert_same(self, before, after):
        weights_b, weights_a = before.pop("weights"), after.pop("weights")
        self.assertEqual(before, after)
        for k, v in weights_b.items():
            self.assertTrue(self.torch.equal(v, weights_a[k]), k)

    def fixtures(self):
        tiny = self.tiny
        lora = ss.apply_lora(tiny.patcher, ss.make_lora_dict(tiny.patcher), 0.8)
        profile = ss.make_profile(lora, base_model=tiny.patcher, strength=0.8)
        constraints = ss.read_contract("constraints.json")
        return lora, profile, constraints

    def test_success_path(self):
        tiny = self.tiny
        lora, profile, constraints = self.fixtures()
        pos, neg = tiny.make_cond(23), tiny.make_cond(9)
        latent = {"samples": tiny.latent.clone(), "extra": "keep"}
        snaps = [snapshot(m) for m in (tiny.patcher, lora)]
        pos0 = [(t.clone(), dict(d)) for t, d in pos]
        neg0 = [(t.clone(), dict(d)) for t, d in neg]
        profile0, constraints0 = copy.deepcopy(profile), copy.deepcopy(constraints)
        latent0 = {"samples": latent["samples"].clone(), "extra": "keep"}
        with ss.known_sft():
            ss.run(tiny, model=lora, positive=pos, negative=neg, latent=latent, identity_profile=profile,
                   constraints=constraints, overrides={"steps": 6}, preset_id="photo_safe")
            ss.run(tiny, positive=pos, negative=neg, latent=latent, overrides={"steps": 6})
        for model, snap in zip((tiny.patcher, lora), snaps):
            self.assert_same(snap, snapshot(model))
        for (t, d), (t0, d0) in zip(pos + neg, pos0 + neg0):
            self.assertTrue(self.torch.equal(t, t0))
            self.assertEqual(d, d0)
        self.assertEqual(set(latent), {"samples", "extra"})
        self.assertTrue(self.torch.equal(latent["samples"], latent0["samples"]))
        self.assertEqual(profile, profile0)
        self.assertEqual(constraints, constraints0)

    def test_no_wrapper_and_no_slg_hook_leak_into_the_incoming_model(self):
        tiny = self.tiny
        ss.run(tiny, overrides={"steps": 6})
        self.assertNotIn("wrappers", tiny.patcher.model_options["transformer_options"])
        self.assertNotIn("sampler_calc_cond_batch_function", tiny.patcher.model_options)
        self.assertEqual(tiny.patcher.model_options, {"transformer_options": {}})

    def test_without_slg_the_counter_still_goes_on_a_clone(self):
        """No SLG clone exists (slg off, cfg 1): the solver must clone for the counter itself."""
        tiny = self.tiny
        snap = snapshot(tiny.patcher)
        ss.run(tiny, overrides={"steps": 6, "slg_mode": "off"})
        ss.run(tiny, cfg=1.0, overrides={"steps": 6})
        self.assert_same(snap, snapshot(tiny.patcher))
        self.assertNotIn("wrappers", tiny.patcher.model_options["transformer_options"])

    def test_failure_path_after_the_clone_exists(self):
        """A NaN stops the run after the SLG clone, the counter and the noise exist; nothing of it sticks."""
        tiny = self.tiny
        snap = snapshot(tiny.patcher)
        net = tiny.model.diffusion_model
        original = net.forward
        net.forward = lambda x, t, *a, **k: original(x, t, *a, **k) * float("nan")
        try:
            with self.assertRaises(ss.errors_module().NonFiniteOutput):
                ss.run(tiny, overrides={"steps": 6})
        finally:
            del net.forward
        self.assert_same(snap, snapshot(tiny.patcher))

    def test_failure_path_before_the_clone(self):
        tiny = self.tiny
        snap = snapshot(tiny.patcher)
        latent = {"samples": self.torch.ones(1, 4, 8, 8)}
        before = latent["samples"].clone()
        with self.assertRaises(ss.errors_module().InvalidLatent):
            ss.run(tiny, latent=latent)
        self.assertTrue(self.torch.equal(latent["samples"], before))
        self.assert_same(snap, snapshot(tiny.patcher))

    def test_the_report_is_a_fresh_object_each_run(self):
        a = ss.run(self.tiny, overrides={"steps": 6}).report
        b = ss.run(self.tiny, overrides={"steps": 6}).report
        self.assertIsNot(a, b)
        a["warnings"].append("x")
        self.assertNotIn("x", b["warnings"])
        self.assertEqual(a["recipe_hash"], b["recipe_hash"])


if __name__ == "__main__":
    unittest.main()
