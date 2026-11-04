"""MP-76: the panel refreshes a preset_pin that differs from the installed pin only in sha8.

The functions parsePin / pinMatches / decidePin are cut out of web/mossphoton.js and run in node (the file imports the
ComfyUI frontend, so it can not be loaded whole). Same id and version, other sha8: written; another version: never.
"""
import json
import re
import shutil
import subprocess
import unittest

from .test_route_fields import read_js

INSTALLED = "turbo_darkroom_fullbody@1.0.0#8f673374"


def cut(src, start_pattern):
    m = re.search(start_pattern, src)
    assert m, start_pattern
    end = src.index("\n}\n", m.start()) + 3 if "function" in start_pattern else src.index("\n};\n", m.start()) + 4
    return src[m.start():end]


def decide(pin, preset_id="turbo_darkroom_fullbody", other_pin=None):
    src = read_js()
    code = "\n".join((
        re.search(r"const PIN_RE = .*;", src).group(0),
        cut(src, r"export const parsePin"),
        cut(src, r"export function pinMatches"),
        cut(src, r"async function decidePin"),
    )).replace("export ", "")
    code += """
const getView = async () => (%s);
decidePin(%s, %s, %s).then((d) => console.log(JSON.stringify(d)));
""" % (json.dumps({"ok": other_pin is not None, "body": {"preset": {"pin": other_pin}}}), json.dumps(pin),
       json.dumps(preset_id), json.dumps(INSTALLED))
    out = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class PinRefreshTest(unittest.TestCase):
    def test_other_sha8_same_version_is_refreshed(self):
        d = decide("turbo_darkroom_fullbody@1.0.0#d2e9ebd3")
        self.assertEqual(d["write"], INSTALLED)
        self.assertEqual(d["note"]["kind"], "refreshed")
        self.assertEqual(d["note"]["from"], "turbo_darkroom_fullbody@1.0.0#d2e9ebd3")

    def test_other_version_is_never_rewritten(self):
        for pin in ("turbo_darkroom_fullbody@0.9.0#8f673374", "turbo_darkroom_fullbody@1.0.1#d2e9ebd3",
                    "turbo_darkroom_fullbody@2.0.0"):
            d = decide(pin)
            self.assertIsNone(d["write"], pin)
            self.assertEqual(d["note"]["kind"], "older", pin)

    def test_installed_pin_and_pin_without_sha8_are_left_alone(self):
        for pin in (INSTALLED, "turbo_darkroom_fullbody@1.0.0"):
            d = decide(pin)
            self.assertIsNone(d["write"], pin)
            self.assertIsNone(d["note"], pin)

    def test_malformed_and_empty_behave_as_before(self):
        self.assertEqual(decide("turbo_darkroom_fullbody@1.0")["note"]["kind"], "malformed")
        self.assertIsNone(decide("turbo_darkroom_fullbody@1.0")["write"])
        self.assertEqual(decide("")["note"]["kind"], "filled")
        self.assertEqual(decide("")["write"], INSTALLED)


class PinRefreshSourceTest(unittest.TestCase):
    def test_the_panel_has_a_line_for_the_refresh(self):
        src = read_js()
        self.assertIn("refreshed:", src)
        self.assertRegex(src, r"parsed\.version === parsePin\(installedPin\)\?\.version")


if __name__ == "__main__":
    unittest.main()
