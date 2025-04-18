# sprints/04_attachment_theory_demo/code/diffusion_visualizer.py
"""
Diffusion-based visualizer to map attachment states/probabilities to artistic overlays.
"""
from typing import List, Dict
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline


class DiffusionVisualizer:
    """
    Uses a Stable Diffusion pipeline to generate artistic overlays representing attachment states.
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = None
    ):
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # Load pipeline with appropriate dtype for device
        if self.device.startswith("cuda" or "mps"):
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
        else:
            # CPU inference requires float32
            print("Using CPU for inference")
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipe.to(self.device)
        # Map attachment state indices to textual prompts
        self.prompt_map: Dict[int, str] = {
            0: "a gentle pastel scene of calm closeness, art by Studio Lonn",
            1: "a fragmented high-contrast collage reflecting anxious attachment",
            2: "a broken mosaic with jagged edges reflecting avoidant detachment",
        }

    def generate_overlay(
        self, base_image: Image.Image, state_probs: List[float]
    ) -> Image.Image:
        """
        Generate an overlay image based on attachment state probabilities.

        Args:
            base_image: PIL.Image with segmentation mask or frame.
            state_probs: list of 3 floats summing to 1 for [secure, anxious, avoidant].

        Returns:
            Blended PIL.Image of overlay and base.
        """
        # Select highest-probability state
        top_idx = max(range(len(state_probs)), key=state_probs.__getitem__)
        prompt = self.prompt_map.get(top_idx, self.prompt_map[0])
        # Generate image via diffusion
        with torch.autocast(self.device):
            result = self.pipe(prompt, num_inference_steps=25, guidance_scale=7.5)
        overlay = result.images[0].resize(base_image.size)
        # Blend overlay onto base image (RGBA)
        base_rgba = base_image.convert("RGBA")
        overlay_rgba = overlay.convert("RGBA")
        blended = Image.blend(base_rgba, overlay_rgba, alpha=0.6)
        return blended


if __name__ == "__main__":
    import os

    # Determine script and results paths
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "results"))
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Load a base segmentation example
    base_path = os.path.join(results_dir, "segmentation_example.png")
    base = Image.open(base_path)
    # Example state probabilities (secure/anxious/avoidant)
    probs = [0.7, 0.2, 0.1]
    viz = DiffusionVisualizer()
    out = viz.generate_overlay(base, probs)
    out_path = os.path.join(results_dir, "attachment_demo_overlay.png")
    out.save(out_path)
    print(f"Saved overlay to {out_path}")
