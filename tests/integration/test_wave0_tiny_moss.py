"""Wave 0 integration: ComfyUI on CPU and the tiny random MOSS fixture (design 6.0, LD4/M3, 6.4 test_slg22_non_vacuous).

Skipped with a message when ComfyUI is not installed (run_tests prints the skipped count). No weights, no GPU.
"""
import hashlib
import unittest

from . import _comfy_env, tiny_moss


def _state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _native_slg(patcher, layer, start=0.0, end=1.0):
    """The stock SkipLayerGuidanceDiTSimple (uncond-only) applied on a clone: the native reference of design D3."""
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    return SkipLayerGuidanceDiTSimple.execute(
        patcher, start, end, double_layers="", single_layers=str(layer)).args[0]


class EnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _comfy_env.setup()

    def test_comfy_is_on_cpu_with_eight_threads(self):
        import torch
        import comfy.model_management
        self.assertEqual(comfy.model_management.get_torch_device().type, "cpu")
        self.assertEqual(torch.get_num_threads(), _comfy_env.THREADS)
        self.assertEqual(_comfy_env.THREADS, 8)

    def test_setup_is_idempotent(self):
        self.assertEqual(_comfy_env.setup(), _comfy_env.setup())


class TinyMossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()

    def test_tiny_model_has_24_single_blocks(self):
        layers = self.tiny.model.diffusion_model.layers
        self.assertEqual(len(layers), 24)
        self.assertEqual(tiny_moss.LAYERS, 24)
        self.assertEqual({type(b).__name__ for b in layers}, {"Block"})
        self.assertEqual(type(self.tiny.model).__name__, "MOSSImage")

    def test_tuple_shape_of_the_fixture(self):
        patcher, model, make_cond, latent, sigmas = self.tiny
        self.assertIs(patcher.model, model)
        self.assertEqual(tuple(latent.shape), (1, 128, 8, 8))
        self.assertEqual(len(sigmas), tiny_moss.STEPS + 1)
        self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
        cond = make_cond(23)
        self.assertEqual(tuple(cond[0][0].shape), (1, 23, tiny_moss.TXT))
        self.assertEqual(tiny_moss.TXT, 64)

    def test_model_sampling_is_the_moss_native_one(self):
        ms = self.tiny.patcher.get_model_object("model_sampling")
        self.assertEqual(float(ms.shift), 5.0)
        self.assertEqual(float(ms.multiplier), 1.0)

    def test_cond_is_seeded_by_length(self):
        import torch
        a, b, c = self.tiny.make_cond(9), self.tiny.make_cond(9), self.tiny.make_cond(23)
        self.assertTrue(torch.equal(a[0][0], b[0][0]))
        self.assertNotEqual(tuple(a[0][0].shape), tuple(c[0][0].shape))

    def test_weights_are_seeded(self):
        again = tiny_moss.build_tiny_moss()
        self.assertEqual(_state_hash(again.model), _state_hash(self.tiny.model))
        other = tiny_moss.build_tiny_moss(weight_seed=tiny_moss.WEIGHT_SEED + 1)
        self.assertNotEqual(_state_hash(other.model), _state_hash(self.tiny.model))

    def test_sampling_is_finite_and_deterministic(self):
        import torch
        a = tiny_moss.sample_euler(self.tiny)
        b = tiny_moss.sample_euler(self.tiny)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), (1, 128, 8, 8))
        self.assertGreater(float(a.abs().sum()), 0.0)

    def test_slg22_non_vacuous(self):
        """M3: with 24 layers skipping layer 22 in the uncond branch changes the result."""
        base = tiny_moss.sample_euler(self.tiny)
        slg_model = _native_slg(self.tiny.patcher, 22)
        with_slg = tiny_moss.sample_euler(self.tiny, slg_model)
        self.assertNotEqual(_tensor_hash(with_slg), _tensor_hash(base))
        # the incoming MODEL is never mutated (D3)
        self.assertNotIn("sampler_calc_cond_batch_function", self.tiny.patcher.model_options)
        self.assertEqual(_tensor_hash(tiny_moss.sample_euler(self.tiny)), _tensor_hash(base))

    def test_slg_layer_23_also_changes_the_result(self):
        base = tiny_moss.sample_euler(self.tiny)
        self.assertNotEqual(
            _tensor_hash(tiny_moss.sample_euler(self.tiny, _native_slg(self.tiny.patcher, 23))), _tensor_hash(base))


class FourLayerControlTest(unittest.TestCase):
    """The reason for 24 layers, kept as a control: on 4 layers SLG 22 is a bit-exact no-op (MP-5 R5)."""

    def test_slg22_is_vacuous_on_four_layers(self):
        small = tiny_moss.build_tiny_moss(layers=4)
        self.assertEqual(len(small.model.diffusion_model.layers), 4)
        base = tiny_moss.sample_euler(small)
        with_slg = tiny_moss.sample_euler(small, _native_slg(small.patcher, 22))
        self.assertEqual(_tensor_hash(with_slg), _tensor_hash(base))


if __name__ == "__main__":
    unittest.main()
