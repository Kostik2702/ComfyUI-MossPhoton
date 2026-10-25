"""Other required GPU tests of design 6.7 and the turbo mechanics G4. Opt-in, see ``_env``.

* the real SFT loaded by ``UNETLoader`` has no foreign modifiers (the bf16 reference set; on failure the architect amends
  the allowlist of design 1.2, the coder does not relax it);
* ``MossPhotonPromptLayout`` without blocks produces the CONDITIONING of ``CLIPTextEncode`` on the reference TE;
* the stock ``EmptyLatentImage`` graph equals the ``EmptyFlux2LatentImage`` graph on C01 (M8; two extra stock runs, full size);
* turbo G4 (a)-(d): counter, negative unused at cfg 1, forced SLG no-op with warning, both stack orders, outside-verified warning.
"""
import unittest

from . import _env

S = None


def setUpModule():
    global S
    S = _env.session()


class OtherRequiredTest(unittest.TestCase):
    @unittest.skipIf(_env.size() == "smoke", "two extra stock runs; part of the full size")
    def test_other_required(self):
        import extras
        out = extras.run_other_required(S.assets, S.cases, S.gallery)
        self.assertEqual([c for c in out if not c["ok"]], [])


class TurboMechanicsTest(unittest.TestCase):
    def test_g4(self):
        import extras
        out = extras.run_g4(S.assets, S.cases, S.gallery)
        self.assertEqual([c for c in out if not c["ok"]], [])


if __name__ == "__main__":
    unittest.main()
