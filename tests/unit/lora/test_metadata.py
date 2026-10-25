"""Metadata whitelist, trained_on, architecture and base_match (design 2.4, 6.5)."""
import unittest

from moss_photon.core import lora

CHARACTER_B = {
    "ss_network_dim": "64", "ss_base_model_version": "moss_image",
    "ss_sd_model_name": "urn:air:moss:diffusion_model:huggingface:Comfy-Org/MOSS-Image@main/diffusion_models/"
                        "moss_image_base_int8_convrot.safetensors",
    "software": '{"name": "ai-toolkit", "repo": "https://github.com/ostris/ai-toolkit", "version": "0.13.30"}',
    "modelspec.title": "Character B", "ss_optimizer": "automagic", "modelspec.thumbnail": "data:image/jpeg;base64,AAAA",
}
KREA = {"modelspec.title": "Krea2_Cheekbones_pairs", "ss_architecture": "krea2drv",
        "modelspec.architecture": "Krea-2/lora", "modelspec.implementation": "https://github.com/krea-ai/krea-2"}
CINEMA = {"ss_base_model_version": "zimage", "ss_sd_model_name": "urn:air:zimageturbo:repository:huggingface:ostris/Z-Image-De-Turbo@main.tar",
          "modelspec.architecture": "zimageturbo/lora"}
LUISAC = {"ss_architecture": "mossimage", "modelspec.architecture": "MOSS-Image/lora", "modelspec.title": "MOSS_LUISAC_REALOGIC"}


class MetadataTest(unittest.TestCase):
    def test_ss_base_model_version_is_read_and_the_whitelist_is_closed(self):
        p = lora.parse_lora_metadata(CHARACTER_B)
        self.assertEqual(p["whitelist"]["ss_base_model_version"], "moss_image")
        self.assertEqual(set(p["whitelist"]), {"ss_base_model_version", "ss_sd_model_name", "software", "ss_network_dim",
                                               "modelspec.title"})
        self.assertEqual(p["whitelist"]["software"], "ai-toolkit 0.13.30")
        self.assertNotIn("data:image", repr(p))

    def test_model_name_is_trimmed_to_a_basename(self):
        p = lora.parse_lora_metadata({"ss_sd_model_name": "C:\\models\\unet\\moss_image_sft_bf16.safetensors"})
        self.assertEqual(p["whitelist"]["ss_sd_model_name"], "moss_image_sft_bf16.safetensors")
        p = lora.parse_lora_metadata(CHARACTER_B)
        self.assertEqual(p["whitelist"]["ss_sd_model_name"], "moss_image_base_int8_convrot.safetensors")

    def test_trained_on(self):
        self.assertEqual(lora.parse_lora_metadata(CHARACTER_B)["trained_on"], "base")
        self.assertEqual(lora.parse_lora_metadata({"ss_sd_model_name": "x/moss_image_sft_int8_convrot.safetensors"})["trained_on"], "sft")
        self.assertEqual(lora.parse_lora_metadata({"ss_sd_model_name": "MOSS-Image-Base"})["trained_on"], "base")
        self.assertEqual(lora.parse_lora_metadata({"ss_sd_model_name": "base_and_sft_mix"})["trained_on"], "unknown")
        self.assertEqual(lora.parse_lora_metadata({"ss_sd_model_name": "my_custom_model"})["trained_on"], "unknown")
        self.assertEqual(lora.parse_lora_metadata({"ss_sd_model_name": "database"})["trained_on"], "unknown")
        self.assertEqual(lora.parse_lora_metadata({})["trained_on"], "unknown")
        self.assertEqual(lora.parse_lora_metadata(None)["trained_on"], "unknown")

    def test_architecture(self):
        self.assertEqual(lora.parse_lora_metadata(CHARACTER_B)["architecture"], "moss")
        self.assertEqual(lora.parse_lora_metadata({"ss_base_model_version": "MOSS-Image"})["architecture"], "moss")
        self.assertEqual(lora.parse_lora_metadata(LUISAC)["architecture"], "moss")
        self.assertEqual(lora.parse_lora_metadata({})["architecture"], "unknown")
        self.assertEqual(lora.parse_lora_metadata({"software": "ai-toolkit"})["architecture"], "unknown")

    def test_krea_and_zimage_metadata_is_an_architecture_mismatch(self):
        krea = lora.parse_lora_metadata(KREA)
        self.assertEqual((krea["architecture"], krea["architecture_value"]), ("other", "krea2drv"))
        cinema = lora.parse_lora_metadata(CINEMA)
        self.assertEqual((cinema["architecture"], cinema["architecture_value"]), ("other", "zimage"))

    def test_one_moss_value_among_others_wins(self):
        self.assertEqual(lora.parse_lora_metadata({"ss_base_model_version": "moss_image",
                                                   "modelspec.architecture": "something"})["architecture"], "moss")

    def test_base_match(self):
        self.assertIs(lora.base_match_of("base", "moss-image-sft"), False)
        self.assertIs(lora.base_match_of("sft", "moss-image-sft"), True)
        self.assertIs(lora.base_match_of("base", "moss-image-base"), True)
        self.assertIsNone(lora.base_match_of("unknown", "moss-image-sft"))
        self.assertIsNone(lora.base_match_of("base", "unknown-moss"))
        self.assertIsNone(lora.base_match_of("base", None))


if __name__ == "__main__":
    unittest.main()
