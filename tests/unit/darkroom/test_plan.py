"""MP-57: the pure darkroom plan. Phases by sigma progress, per-step settings, segments, the forward counts per phase
(guarantee d: cfg 1 = 1 per step, cfg > 1 = 2 per step), the character weights (cross-checked against ``phase``), the
independent seeds (guarantee b) and the refusals of the plan itself (guarantee e).

Expectations come from a second route (plain loops over the sigma array, the textbook formulas), not from ``darkroom``.
"""
import unittest

from moss_photon.core import consts, darkroom, errors, lora_plan, phase

from .support import MP57, OFF, settings_of

# the exact float32 'simple' arrays of the live model, shift 5 (research/turbo_phases/common.py SIMPLE)
S12 = [1.0, 0.9822193384170532, 0.9617158770561218, 0.9375, 0.9092148542404175, 0.8752998113632202, 0.8333333134651184,
       0.7814843058586121, 0.7148972749710083, 0.625, 0.5005995035171509, 0.314371258020401, 0.0]
S8 = [1.0, 0.9722222089767456, 0.9375, 0.8928571343421936, 0.8333333134651184, 0.75, 0.625, 0.4166666567325592, 0.0]
CHARACTER = {"stack_index": 1, "role": "character"}


def plan_of(recipe=MP57, sigmas=S12, phased=CHARACTER, seed=42, **kw):
    """The plan of the settings of preset ``recipe`` (``OFF`` = the all-neutral block) with ``overrides``."""
    overrides = kw.pop("overrides", {})
    return darkroom.build_plan(settings_of(recipe, overrides), sigmas, seed=seed, phased=phased, **kw)


def entry(phase_name, strength=1.0, index=0):
    """A resolved LoRA Phase entry without patches (the plan only reads its phase and strength)."""
    return lora_plan.Resolved(index=index, basename="x.safetensors", size=1, sha256=None, strength=strength,
                              phase=phase_name, patch_dict={})


def progress(sigmas):
    return [(sigmas[0] - s) / (sigmas[0] - sigmas[-1]) for s in sigmas[:-1]]


class PhasesTest(unittest.TestCase):
    def test_the_portrait_recipe_on_12_simple_steps(self):
        plan = plan_of()
        phases = [s.phase for s in plan.steps]
        self.assertEqual(phases, ["compose"] * 3 + ["commit"] * 4 + ["finish"] * 5)
        self.assertEqual(plan.total_steps, 12)
        self.assertEqual([s.cfg for s in plan.steps], [1.0] * 7 + [2.0] * 5)
        self.assertEqual(plan.steps[7].index, 7, "the finish guidance starts at step 7 (the ticket's onset)")

    def test_a_step_belongs_to_the_phase_its_start_progress_falls_in(self):
        plan = plan_of(overrides={"r_f": 0.3})
        r = progress(S12)
        for step in plan.steps:
            expected = "compose" if r[step.index] < 0.06 else "commit" if r[step.index] < 0.17 else \
                "hold" if r[step.index] < 0.3 else "finish"
            self.assertEqual(step.phase, expected, step.index)
        self.assertEqual({s.phase for s in plan.steps}, {"compose", "commit", "hold", "finish"})

    def test_progress_is_the_sigma_distance_of_the_whole_schedule(self):
        for sigmas in (S12, S8, [0.8, 0.5, 0.1, 0.0]):
            plan = plan_of(sigmas=sigmas)
            self.assertEqual([s.r for s in plan.steps], progress(sigmas))
        # and it equals the cumulative |d sigma| progress of the LoRA phase module on a monotone schedule
        for got, want in zip((s.r for s in plan_of().steps), phase.progress_fractions(S12)):
            self.assertAlmostEqual(got, want, places=12)

    def test_the_phase_table_has_sigma_bounds_one_minus_r_and_the_step_counts(self):
        rows = {p["name"]: p for p in darkroom.describe(plan_of())["phases"]}
        self.assertEqual(list(rows), ["compose", "commit", "finish"], "no hold phase when r_k == r_f")
        self.assertEqual((rows["compose"]["sigma_start"], rows["compose"]["sigma_end"]), (1.0, 1.0 - 0.06))
        self.assertAlmostEqual(rows["commit"]["sigma_end"], 1.0 - 0.17, places=12)
        self.assertEqual((rows["finish"]["sigma_start"], rows["finish"]["sigma_end"]), (1.0 - 0.17, 0.0))
        self.assertEqual([rows[n]["steps"] for n in rows], [3, 4, 5])
        self.assertEqual([(rows[n]["first_step"], rows[n]["last_step"]) for n in rows], [(0, 2), (3, 6), (7, 11)])

    def test_a_hold_phase_appears_between_commit_and_finish(self):
        rows = {p["name"]: p for p in darkroom.describe(plan_of(overrides={"r_f": 0.4}))["phases"]}
        self.assertEqual(list(rows), ["compose", "commit", "hold", "finish"])
        self.assertEqual(rows["hold"]["cfg"], 1.0)
        self.assertEqual(rows["hold"]["weight_first"], 1.2)

    def test_an_empty_phase_has_no_step_range(self):
        rows = {p["name"]: p for p in darkroom.describe(plan_of(overrides={"r_c": 0.0, "r_k": 0.1}))["phases"]}
        self.assertEqual((rows["compose"]["steps"], rows["compose"]["first_step"], rows["compose"]["last_step"]),
                         (0, None, None))


class ForwardsTest(unittest.TestCase):
    """Guarantee (d), the plan side: cfg 1 costs one forward per step, cfg above 1 costs two."""

    def test_forwards_per_phase_and_in_total(self):
        plan = plan_of()
        self.assertEqual(plan.expected_branch_forwards, 7 * 1 + 5 * 2)
        rows = {p["name"]: p for p in darkroom.describe(plan, batch=1)["phases"]}
        self.assertEqual([rows[n]["expected_sample_forwards"] for n in rows], [3 * 1, 4 * 1, 5 * 2])
        self.assertEqual(sum(p["expected_sample_forwards"] for p in rows.values()), plan.expected_branch_forwards)
        self.assertEqual([r["expected_sample_forwards"] for r in darkroom.describe(plan, batch=3)["phases"]],
                         [9, 12, 30])

    def test_every_cfg_above_one_is_two_forwards(self):
        for cfg in (1.0001, 1.5, 2.0, 7.0):
            plan = plan_of(overrides={"finish_cfg": cfg})
            self.assertEqual(plan.expected_branch_forwards, 7 + 5 * 2, cfg)
        plan = plan_of(overrides={"compose_cfg": 1.5, "finish_cfg": 1.0, "r_f": 0.6})
        self.assertEqual(plan.expected_branch_forwards, 2 * 11 + 1 * 1)

    def test_the_negative_is_used_only_when_a_phase_runs_above_cfg_one(self):
        self.assertTrue(plan_of().uses_negative)
        self.assertFalse(plan_of(OFF, phased=None).uses_negative)

    def test_a_reexpose_run_has_more_steps_and_the_plan_counts_them(self):
        plan = plan_of(overrides={"reexpose": True, "r_k": 0.3, "r_f": 0.3, "sigma_r": 0.8})
        self.assertEqual(plan.total_steps, len(plan.steps))
        self.assertEqual(plan.expected_branch_forwards, sum(1 if s.cfg == 1.0 else 2 for s in plan.steps))


class WeightsTest(unittest.TestCase):
    def test_the_character_weights_equal_the_phase_module_schedule(self):
        plan = plan_of()
        sched = phase.build_schedule(S12, phase.make_plan(1, 1.0, 1.2, 0.06, 0.17))
        self.assertEqual([s.weight for s in plan.steps], list(sched.applied))
        self.assertEqual(sorted(set(s.weight for s in plan.steps)), sorted(set(sched.applied)))

    def test_weights_follow_smoothstep_at_six_levels(self):
        plan = plan_of(overrides={"w_compose": 0.4, "w_commit": 1.6, "r_c": 0.1, "r_k": 0.5, "r_f": 0.5})
        r = progress(S12)
        for step in plan.steps:
            t = min(1.0, max(0.0, (r[step.index] - 0.1) / 0.4))
            s = 3 * t * t - 2 * t ** 3
            level = int(round(s * 5))
            want = 0.4 + 1.2 * level / 5
            self.assertAlmostEqual(step.weight, want, places=12, msg=step.index)
        self.assertEqual(plan.steps[0].weight, 0.4)
        self.assertEqual(plan.steps[-1].weight, 1.6)

    def test_equal_weights_are_one_constant(self):
        plan = plan_of(overrides={"w_commit": 1.0})
        self.assertEqual({s.weight for s in plan.steps}, {1.0})

    def test_a_constant_profile_has_no_weights(self):
        plan = plan_of(OFF, phased=None)
        self.assertEqual({s.weight for s in plan.steps}, {None})
        self.assertFalse(plan.uses_adapter)

    def test_a_phased_adapter_with_equal_weights_still_runs_through_level_clones(self):
        plan = plan_of(OFF)
        self.assertTrue(plan.uses_adapter)
        self.assertEqual({s.weight for s in plan.steps}, {1.0})
        self.assertFalse(plan.neutral, "the character is applied by the engine, not by the MODEL")


def boost(reason=None, strength=1.0):
    return darkroom.ConstantBoost(stack_index=1, basename="A.safetensors", size=10, sha256=None, strength=strength,
                                  patch_dict=None if reason else {"k": 1}, patch_entries=3, reason=reason)


class ConstantProfileBoostTest(unittest.TestCase):
    """MP-67: a constant character profile boosts the commit phase with an EXTRA patch of the same LoRA."""

    def test_the_default_recipe_plans_the_extra_patch_with_the_weights_of_the_phased_path(self):
        phased = plan_of()
        const = plan_of(phased=None, constant_boost=boost())
        self.assertEqual(const.boost_mode, darkroom.BOOST_EXTRA)
        self.assertFalse(const.uses_adapter)
        self.assertEqual([s.weight for s in const.steps], [s.weight for s in phased.steps])
        self.assertEqual([(g.first, g.stop, g.weight, g.cfg) for g in const.segments],
                         [(g.first, g.stop, g.weight, g.cfg) for g in phased.segments])
        self.assertEqual(const.character, {"mode": "constant_profile_extra_patch", "stack_index": 1,
                                           "profile_phase_plan_replaced": False, "base_strength": 1.0,
                                           "boost_skipped": None})

    def test_a_skipped_boost_runs_the_plan_without_weights_and_says_why(self):
        plan = plan_of(phased=None, constant_boost=boost("file_not_found"))
        self.assertIsNone(plan.boost_mode)
        self.assertEqual({s.weight for s in plan.steps}, {None})
        self.assertEqual(plan.character["boost_skipped"], "file_not_found")
        self.assertEqual(plan.character["mode"], "constant_profile")
        self.assertEqual([(g.first, g.stop) for g in plan.segments], [(0, 7), (7, 12)], "only the guidance switch cuts")

    def test_no_boost_needs_no_loader_result(self):
        plan = plan_of(OFF, phased=None, constant_boost=boost())
        self.assertIsNone(plan.boost_mode)
        self.assertIsNone(plan.character["boost_skipped"])

    def test_the_phased_profile_wins_over_a_constant_boost(self):
        plan = plan_of(constant_boost=boost())
        self.assertEqual(plan.boost_mode, darkroom.BOOST_PHASED)

    def test_the_adapter_of_the_boost_carries_the_profile_entry(self):
        adapter = boost().adapter()
        self.assertEqual((adapter.stack_index, adapter.basename, adapter.patch_dict), (1, "A.safetensors", {"k": 1}))


class SegmentsTest(unittest.TestCase):
    def test_segments_cover_every_step_once_and_cut_the_sigma_path_exactly(self):
        plan = plan_of()
        covered = [i for seg in plan.segments for i in range(seg.first, seg.stop)]
        self.assertEqual(covered, list(range(12)))
        for seg in plan.segments:
            path = plan.paths[seg.path]
            self.assertEqual(list(path[seg.local:seg.local_stop + 1]), S12[seg.first:seg.stop + 1])
            self.assertEqual(len({(s.cfg, s.weight, s.model, s.sampler) for s in plan.steps[seg.first:seg.stop]}), 1)

    def test_the_portrait_segments_are_the_weight_levels_and_the_finish(self):
        self.assertEqual([(g.first, g.stop, g.cfg, g.weight) for g in plan_of().segments],
                         [(0, 4, 1.0, 1.0), (4, 5, 1.0, 1.04), (5, 6, 1.0, 1.1199999999999999), (6, 7, 1.0, 1.2),
                          (7, 12, 2.0, 1.2)])

    def test_one_clone_per_weight_level_and_one_for_the_finish_at_most(self):
        plan = plan_of()
        self.assertLessEqual(len(plan.segments), phase.PHASE_LEVELS + 1)

    def test_all_phases_neutral_is_one_segment(self):
        plan = plan_of(OFF, phased=None)
        self.assertEqual(len(plan.segments), 1)
        self.assertTrue(plan.neutral)
        self.assertEqual((plan.segments[0].first, plan.segments[0].stop), (0, 12))
        self.assertEqual(plan.paths, (tuple(S12),))

    def test_a_forced_split_changes_only_the_cuts(self):
        whole = plan_of(OFF, phased=None)
        cut = plan_of(OFF, phased=None, split_at=(3, 8))
        self.assertEqual([(g.first, g.stop) for g in cut.segments], [(0, 3), (3, 8), (8, 12)])
        self.assertEqual(whole.steps, cut.steps)
        self.assertEqual(whole.expected_branch_forwards, cut.expected_branch_forwards)


class FinishFeaturesTest(unittest.TestCase):
    def test_noise_injection_changes_the_sampler_of_the_finish_steps_only(self):
        plan = plan_of(overrides={"eta": 0.3})
        self.assertEqual([s.sampler for s in plan.steps], ["euler"] * 7 + ["euler_ancestral_rf"] * 5)
        self.assertEqual([(g.sampler, g.eta) for g in plan.segments][-1], ("euler_ancestral_rf", 0.3))
        self.assertEqual({g.eta for g in plan.segments[:-1]}, {0.0})

    def test_the_model_switch_moves_the_finish_steps_to_the_second_model(self):
        plan = plan_of(overrides={"switch": True}, has_finish_model=True)
        self.assertEqual([s.model for s in plan.steps], ["main"] * 7 + ["finish"] * 5)
        self.assertEqual(plan.segments[-1].model, "finish")

    def test_slg_acts_only_where_the_guidance_is_above_one_and_never_in_the_finish(self):
        plan = plan_of(overrides={"compose_slg": True, "compose_cfg": 1.5, "r_f": 0.4})
        self.assertEqual([s.slg for s in plan.steps], [True] * 10 + [False] * 2)
        plan = plan_of(overrides={"compose_slg": True})
        self.assertEqual({s.slg for s in plan.steps}, {False}, "compose cfg 1 has no unconditional pass to skip")
        row = darkroom.describe(plan_of(overrides={"compose_slg": True, "compose_cfg": 1.5}))["phases"][0]
        self.assertTrue(row["slg"])

    def test_slg_layers_are_checked_against_the_model(self):
        with self.assertRaises(errors.InvalidOverride):
            plan_of(overrides={"compose_slg": True, "compose_cfg": 1.5}, num_layers=4)
        plan_of(overrides={"compose_slg": True, "compose_cfg": 1.5}, num_layers=24)

    def test_the_sa2_plan(self):
        plan = plan_of(overrides={"schedule": "sa2_8"}, sigmas=darkroom.sa2_values())
        self.assertEqual(plan.total_steps, 8)
        self.assertEqual([round(s.r, 4) for s in plan.steps],
                         [0.0, 0.04, 0.08, 0.12, 0.17, 0.24, 0.375, 0.5833])

    def test_the_sa2_knots_resample_for_another_step_count(self):
        """MP-84: sa2_8 with another step count is a plan, not an error (the solver says the spacing is unmeasured)."""
        self.assertEqual(darkroom.sa2_values(), list(darkroom.SA2_8))
        for n in (1, 4, 12, 30, 1000):
            values = darkroom.sa2_values(n)
            self.assertEqual(len(values), n + 1)
            self.assertEqual((values[0], values[-1]), (1.0, 0.0))
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])), n)
            plan = plan_of(overrides={"schedule": "sa2_8"}, sigmas=values)
            self.assertEqual(plan.total_steps, n)


class ReexposeTest(unittest.TestCase):
    def plan(self, **ov):
        return plan_of(overrides=dict({"reexpose": True, "r_k": 0.3, "r_f": 0.3, "sigma_r": 0.6}, **ov))

    def test_two_sigma_paths_and_the_tail_below_sigma_r(self):
        plan = self.plan()
        k = next(i for i, r in enumerate(progress(S12)) if r >= 0.3)
        self.assertEqual(k, 9)
        self.assertEqual(plan.paths[0], tuple(S12[:k + 1]))
        self.assertEqual(plan.paths[1], (0.6, 0.5005995035171509, 0.314371258020401, 0.0))
        self.assertEqual(plan.total_steps, k + 3)
        self.assertEqual(plan.reexpose_step, k)
        self.assertEqual([g.reexpose_before for g in plan.segments], [False] * (len(plan.segments) - 1) + [True])

    def test_the_steps_after_the_jump_get_the_settings_of_their_own_progress(self):
        plan = self.plan()
        after = plan.steps[plan.reexpose_step:]
        self.assertEqual([round(s.r, 4) for s in after], [0.4, 0.4994, 0.6856])
        self.assertEqual({s.phase for s in after}, {"finish"})

    def test_a_reexpose_above_the_commit_end_walks_the_commit_phase_again(self):
        plan = self.plan(sigma_r=0.9)
        after = plan.steps[plan.reexpose_step:]
        self.assertEqual((after[0].phase, round(after[0].r, 4), after[0].weight), ("commit", 0.1, 1.0))
        self.assertEqual({s.phase for s in after[:1]}, {"commit"})

    def test_the_reexpose_rng_is_logged_and_independent_of_the_main_seed_stream(self):
        block = darkroom.describe(self.plan())
        self.assertEqual(block["rng"]["main_seed"], 42)
        self.assertEqual(block["rng"]["reexpose"], {"enabled": True, "offset": darkroom.REEXPOSE_SEED_OFFSET,
                                                    "seed": 42 + darkroom.REEXPOSE_SEED_OFFSET})
        self.assertEqual(block["rng"]["noise"], {"enabled": False, "offset": darkroom.NOISE_SEED_OFFSET, "seed": None})
        self.assertEqual(len(block["paths"]), 2)
        off = darkroom.describe(plan_of())
        self.assertEqual(off["rng"]["reexpose"]["seed"], None)

    def test_noise_seed_is_logged_when_eta_is_on(self):
        block = darkroom.describe(plan_of(overrides={"eta": 0.2}))
        self.assertEqual(block["rng"]["noise"]["seed"], 42 + darkroom.NOISE_SEED_OFFSET)

    def test_reexpose_before_the_first_step_or_after_the_last_is_left_out_with_a_note(self):
        """MP-84: a re-expose point outside the run turns re-expose off for the run (it was ``invalid_override``)."""
        for ov in ({"r_c": 0.0, "r_k": 0.0, "r_f": 0.0, "w_commit": 1.0}, {"r_k": 0.99, "r_f": 0.99, "w_commit": 1.0}):
            with self.subTest(ov=ov):
                plan = plan_of(overrides=dict({"reexpose": True}, **ov))
                self.assertFalse(plan.settings.reexpose)
                self.assertEqual(len(plan.paths), 1)
                self.assertEqual([n[:2] for n in plan.notes if n[1] == "reexpose"],
                                 [("darkroom_override_ignored", "reexpose")])


class DegradedPlanTest(unittest.TestCase):
    """MP-84 (owner 2026-11-02): a setting the plan cannot use is left out and named in ``notes``; only a schedule that cannot
    be walked at all is refused (``PlanRefusalTest``)."""

    def test_a_weight_change_without_a_character_adapter_skips_the_boost(self):
        plan = plan_of(phased=None)
        self.assertIsNone(plan.boost_mode)
        self.assertFalse(plan.plan_boost)
        self.assertEqual(plan.character["boost_skipped"], "no_identity_lora")
        self.assertEqual({s.weight for s in plan.steps}, {None})
        self.assertTrue(plan.uses_negative)
        plan_of(phased=None, overrides={"w_commit": 1.0})

    def test_finish_settings_on_a_run_without_finish_steps_are_noted_and_skipped(self):
        plan = plan_of(overrides={"r_f": 1.0})
        self.assertFalse(any(s.phase == "finish" for s in plan.steps))
        self.assertEqual([n[:2] for n in plan.notes], [("darkroom_override_ignored", "r_f")])
        self.assertEqual(plan_of(OFF, overrides={"r_f": 1.0}, phased=None).notes, ())  # neutral finish needs no step

    def test_the_model_switch_without_the_second_model_runs_on_the_main_model(self):
        plan = plan_of(overrides={"switch": True})
        self.assertEqual({s.model for s in plan.steps}, {"main"})
        self.assertFalse(plan.settings.switch)
        self.assertEqual([n[:2] for n in plan.notes], [("darkroom_override_ignored", "switch")])

    def test_sigma_r_above_the_first_sigma_turns_reexpose_off(self):
        plan = plan_of(sigmas=[0.5, 0.2, 0.0], overrides={"reexpose": True, "sigma_r": 0.9, "r_k": 0.3, "r_f": 0.3})
        self.assertFalse(plan.settings.reexpose)
        self.assertEqual(len(plan.paths), 1)
        self.assertEqual([n[:2] for n in plan.notes], [("darkroom_override_ignored", "reexpose")])


class LoraEntriesTest(unittest.TestCase):
    """MP-84: LoRA Phase entries split the segments where the set of acting entries changes and take the commit boost."""

    def segments_of(self, *entries, **kw):
        plan = plan_of(phased=None, lora_entries=entries, overrides=kw.pop("overrides", {"w_commit": 1.0}), **kw)
        return plan, [(g.first, g.stop) for g in plan.segments]

    def test_no_entry_is_the_plan_of_a_run_without_a_plan(self):
        with_none, _ = self.segments_of()
        self.assertEqual(with_none.lora_entries, ())
        plain = plan_of(phased=None, overrides={"w_commit": 1.0})
        self.assertEqual([(g.first, g.stop) for g in with_none.segments], [(g.first, g.stop) for g in plain.segments])

    def test_an_entry_for_all_phases_splits_nothing(self):
        plan, cuts = self.segments_of(entry("all"))
        self.assertEqual(cuts, [(0, 7), (7, 12)], "the cuts of the guidance only (compose+commit, finish)")
        self.assertFalse(plan.neutral)

    def test_each_phase_set_cuts_where_the_acting_set_changes(self):
        for phase_name, expected in (("composition", [(0, 3), (3, 7), (7, 12)]),
                                     ("identity", [(0, 7), (7, 12)]),   # MP-86: acts from the first step, like all
                                     ("texture", [(0, 7), (7, 12)]),
                                     ("composition+identity", [(0, 7), (7, 12)])):
            with self.subTest(phase=phase_name):
                plan, cuts = self.segments_of(entry(phase_name))
                acting = [lora_plan.active(plan.lora_entries, s.phase) for s in plan.steps]
                self.assertEqual(cuts, expected)
                for g in plan.segments:
                    self.assertEqual(len({acting[i] for i in range(g.first, g.stop)}), 1, "one acting set per segment")

    def test_zero_strength_never_acts_and_splits_nothing(self):
        plan, cuts = self.segments_of(entry("identity", strength=0.0))
        self.assertEqual(cuts, [(0, 7), (7, 12)])
        self.assertEqual({lora_plan.active(plan.lora_entries, p) for p in darkroom.PHASE_NAMES}, {()})

    def test_identity_entries_take_the_boost(self):
        """MP-86: the character phase; the boost is additive, so ``all`` (an ordinary loader) and ``composition+identity``
        (style and slider LoRAs) never take it."""
        plan = plan_of(phased=None, lora_entries=(entry("identity"),))   # the preset's own w_commit 1.2
        self.assertTrue(plan.plan_boost)
        self.assertIsNone(plan.boost_mode)
        self.assertEqual(plan.character["mode"], "lora_plan_extra_patch")
        self.assertIsNone(plan.character["boost_skipped"])
        self.assertEqual(plan.character["boosted_entries"], ["x.safetensors"])
        self.assertEqual({s.weight for s in plan.steps if s.phase == "compose"}, {1.0})
        self.assertEqual({s.weight for s in plan.steps if s.phase == "finish"}, {1.2})

    def test_a_negative_identity_entry_takes_no_boost_and_is_not_named(self):
        alone = plan_of(phased=None, lora_entries=(entry("identity", strength=-2.0),))
        self.assertFalse(alone.plan_boost)
        self.assertEqual(alone.character["boost_skipped"], "no_identity_lora")
        self.assertNotIn("boosted_entries", alone.character)
        self.assertEqual({s.weight for s in alone.steps}, {None})
        named = lora_plan.Resolved(index=1, basename="char.safetensors", size=1, sha256=None, strength=1.0,
                                   phase="identity", patch_dict={})
        both = plan_of(phased=None, lora_entries=(entry("identity", strength=-2.0), named))
        self.assertEqual(both.character["boosted_entries"], ["char.safetensors"])

    def test_composition_plus_identity_and_all_take_no_boost(self):
        for phase_name in ("composition+identity", "all"):
            with self.subTest(phase=phase_name):
                plan = plan_of(phased=None, lora_entries=(entry(phase_name),))
                self.assertFalse(plan.plan_boost)
                self.assertNotIn("boosted_entries", plan.character)
                self.assertEqual(plan.character["boost_skipped"], "no_identity_lora")

    def test_the_character_info_names_only_the_boosted_entries(self):
        named = lora_plan.Resolved(index=1, basename="char.safetensors", size=1, sha256=None, strength=1.0,
                                   phase="identity", patch_dict={})
        extras = lora_plan.Resolved(index=2, basename="style.safetensors", size=1, sha256=None, strength=0.6,
                                    phase="texture", patch_dict={})
        plan = plan_of(phased=None, lora_entries=(entry("composition"), named, extras))
        self.assertEqual(plan.character["boosted_entries"], ["char.safetensors"])
        self.assertNotIn("boosted_entries", plan_of(phased=None, lora_entries=(entry("texture"),)).character)
        self.assertNotIn("boosted_entries", plan_of(phased=None).character)

    def test_a_known_distill_under_identity_takes_no_boost(self):
        distill = lora_plan.Resolved(index=0, basename="d.safetensors", size=1, sha256=next(iter(consts.KNOWN_DISTILL_LORAS)),
                                     strength=1.0, phase="identity", patch_dict={})
        plan = plan_of(phased=None, lora_entries=(distill,))
        self.assertFalse(plan.plan_boost)
        self.assertEqual(plan.character["boost_skipped"], "no_identity_lora")

    def test_another_phase_takes_no_boost_and_says_it_was_skipped(self):
        for phase_name in ("all", "composition", "texture"):
            with self.subTest(phase=phase_name):
                plan = plan_of(phased=None, lora_entries=(entry(phase_name),))
                self.assertFalse(plan.plan_boost)
                self.assertEqual(plan.character["boost_skipped"], "no_identity_lora")
                self.assertEqual({s.weight for s in plan.steps}, {None})

    def test_the_finish_model_has_its_own_entries(self):
        plan = plan_of(phased=None, overrides={"switch": True, "w_commit": 1.0}, has_finish_model=True,
                       finish_lora_entries=(entry("texture"),), lora_entries=(entry("composition"),))
        finish = [s for s in plan.steps if s.model == "finish"]
        self.assertEqual({s.phase for s in finish}, {"finish"})
        self.assertEqual([lora_plan.active(plan.lora_entries, s.phase) for s in plan.steps if s.phase == "compose"][0], (0,))
        self.assertEqual(lora_plan.active(plan.finish_lora_entries, "finish"), (0,))
        self.assertEqual(lora_plan.active(plan.finish_lora_entries, "compose"), ())


class PlanRefusalTest(unittest.TestCase):
    """Guarantee (e): the plan refuses a schedule that cannot be walked, before any forward."""

    def test_a_schedule_that_does_not_descend_or_is_too_short(self):
        for sigmas in ([1.0], [0.5, 0.5], [0.2, 0.9]):
            with self.subTest(sigmas=sigmas):
                with self.assertRaises(errors.InvalidOverride):
                    plan_of(sigmas=sigmas, overrides={"w_commit": 1.0})

    def test_a_second_model_without_the_switch_is_no_refusal_and_is_not_used(self):
        """MP-67: a setting conflict degrades (the solver warns ``finish_model_ignored``); the plan never needs the model."""
        plan = plan_of(has_finish_model=True)
        self.assertEqual({s.model for s in plan.steps}, {"main"})


if __name__ == "__main__":
    unittest.main()
