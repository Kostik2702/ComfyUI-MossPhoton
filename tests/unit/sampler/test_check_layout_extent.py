"""solver.check_layout_extent (A23): edges inside [-H/2, H/2] x [-W/2, W/2], pure."""
import copy
import json
import os
import unittest

from moss_photon.core import errors, solver

NODE = "MossPhotonSampler"
CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def constraints(extent):
    return {"layout": {"items": 1, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
                       "bboxes": [list(extent)], "extent": list(extent), "experimental": True}}


class LayoutExtentTest(unittest.TestCase):
    def check(self, extent, w=1024, h=512):
        return solver.check_layout_extent(constraints(extent), w, h, node=NODE)

    def test_inside(self):
        self.assertEqual(self.check([-100, -200, 100, 200]), [])

    def test_edges_are_inclusive(self):
        self.assertEqual(self.check([-256, -512, 256, 512]), [])

    def test_y_is_checked_against_the_height_and_x_against_the_width(self):
        # [ymin, xmin, ymax, xmax]: y 300 is beyond H/2 = 256 although it fits the width
        bad = self.check([-300, 0, 10, 10])
        self.assertEqual([i.detail["edge"] for i in bad], ["ymin"])
        bad = self.check([0, -600, 10, 600])
        self.assertEqual(sorted(i.detail["edge"] for i in bad), ["xmax", "xmin"])
        self.assertEqual(self.check([0, -300, 10, 300]), [])

    def test_each_edge(self):
        for extent, edge in (([-257, 0, 0, 0], "ymin"), ([0, 0, 257, 0], "ymax"),
                             ([0, -513, 0, 0], "xmin"), ([0, 0, 0, 513], "xmax")):
            with self.subTest(edge=edge):
                issues = self.check(extent)
                self.assertEqual([i.detail["edge"] for i in issues], [edge])
                self.assertEqual(issues[0].code, "invalid_layout")

    def test_raises_as_invalid_layout(self):
        with self.assertRaises(errors.InvalidLayout):
            errors.raise_first_error(self.check([0, 0, 0, 5000]))

    def test_no_constraints_no_layout_no_extent(self):
        self.assertEqual(solver.check_layout_extent(None, 64, 64, node=NODE), [])
        self.assertEqual(solver.check_layout_extent({"layout": None}, 64, 64, node=NODE), [])
        self.assertEqual(solver.check_layout_extent({"layout": {"items": 0}}, 64, 64, node=NODE), [])

    def test_malformed_extent(self):
        for extent in ([1, 2, 3], "abc", [0, 0, 0, float("nan")], [0, 0, 0, True], [0, 0, 0, None]):
            with self.subTest(extent=extent):
                c = {"layout": {"extent": extent}}
                self.assertEqual([i.code for i in solver.check_layout_extent(c, 64, 64, node=NODE)], ["invalid_layout"])

    def test_the_contract_fixture_without_a_layout_passes(self):
        with open(os.path.join(CONTRACTS, "constraints.json"), "r", encoding="utf-8") as f:
            fixture = json.load(f)
        self.assertIsNone(fixture["layout"])
        self.assertEqual(solver.check_layout_extent(fixture, 1024, 1024, node=NODE), [])
        with_layout = copy.deepcopy(fixture)
        with_layout["layout"] = constraints([-100, -100, 100, 100])["layout"]
        self.assertEqual(solver.check_layout_extent(with_layout, 1024, 1024, node=NODE), [])
        self.assertEqual(len(solver.check_layout_extent(with_layout, 128, 128, node=NODE)), 4)

    def test_input_is_not_mutated(self):
        c = constraints([-1, -1, 1, 1])
        before = copy.deepcopy(c)
        solver.check_layout_extent(c, 8, 8, node=NODE)
        self.assertEqual(c, before)


if __name__ == "__main__":
    unittest.main()
