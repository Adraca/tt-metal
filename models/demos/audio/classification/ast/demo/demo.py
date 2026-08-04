# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Demo script for Audio Spectrogram Transformer (AST) bring-up on Tenstorrent hardware.

Bounty: tenstorrent/tt-metal#52054
Model: MIT/ast-finetuned-audioset-10-10-0.4593

This script:
    1. Loads the HuggingFace AST model and feature extractor.
    2. Runs a CPU reference forward pass to produce golden logits.
    3. Unfolds the spectrogram into patches and runs TTNN inference.
    4. Validates PCC between CPU and TTNN outputs.
    5. Compares top-k predictions between both backends.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import ASTFeatureExtractor, ASTForAudioClassification

import ttnn

# Add parent directory so we can import the tt module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt.ttnn_ast import create_ast_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 1
NUM_CLASSES = 527
SPEC_FREQ_BINS = 128
SPEC_TIME_FRAMES = 1024
PATCH_SIZE = 16
PATCH_STRIDE = 10
PCC_THRESHOLD = 0.99
TOP_K = 5
TOP_K_OVERLAP_MIN = 4

# Unfold math:
#   freq_patches  = (128  - 16) // 10 + 1 = 12
#   time_patches  = (1024 - 16) // 10 + 1 = 101
#   total_patches = 12 * 101 = 1212
FREQ_PATCHES = (SPEC_FREQ_BINS - PATCH_SIZE) // PATCH_STRIDE + 1  # 12
TIME_PATCHES = (SPEC_TIME_FRAMES - PATCH_SIZE) // PATCH_STRIDE + 1  # 101
TOTAL_PATCHES = FREQ_PATCHES * TIME_PATCHES  # 1212
PATCH_DIM = PATCH_SIZE * PATCH_SIZE  # 256


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_pcc(golden: torch.Tensor, predicted: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors.

    Handles edge cases where one or both tensors have zero variance,
    which would otherwise produce NaN from ``torch.corrcoef``.

    Args:
        golden: Reference tensor (any shape, will be flattened).
        predicted: Predicted tensor (same number of elements as *golden*).

    Returns:
        PCC value in [-1.0, 1.0], or 1.0 / 0.0 for degenerate cases.
    """
    golden_flat = golden.float().flatten()
    predicted_flat = predicted.float().flatten()

    if golden_flat.std() == 0 and predicted_flat.std() == 0:
        return 1.0
    if golden_flat.std() == 0 or predicted_flat.std() == 0:
        return 0.0

    return torch.corrcoef(torch.stack([golden_flat, predicted_flat]))[0, 1].item()


def load_audio(audio_path: str | None) -> torch.Tensor:
    """Load audio waveform from *audio_path* or generate dummy noise.

    Args:
        audio_path: Path to a .wav / .flac / .mp3 file, or ``None`` for
            a 1-second random waveform at 16 kHz.

    Returns:
        1-D ``torch.Tensor`` of audio samples.
    """
    if audio_path is not None:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(audio_path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            # Mix to mono if necessary
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            print(f"[INFO] Loaded audio from {audio_path}  "
                  f"(samples={waveform.shape[0]}, sr={SAMPLE_RATE})")
            return waveform
        except ImportError:
            print("[WARN] torchaudio not installed — falling back to dummy audio.")
        except Exception as exc:
            print(f"[WARN] Failed to load '{audio_path}': {exc} — falling back to dummy audio.")

    # Deterministic dummy audio for reproducibility
    generator = torch.Generator().manual_seed(42)
    dummy = torch.randn(SAMPLE_RATE * AUDIO_DURATION_SEC, generator=generator)
    print(f"[INFO] Using dummy audio (samples={dummy.shape[0]}, sr={SAMPLE_RATE})")
    return dummy


def unfold_spectrogram(input_values: torch.Tensor) -> torch.Tensor:
    """Unfold a mel-spectrogram into a sequence of flattened patches.

    The AST model treats the spectrogram as a 2-D image and extracts
    non-overlapping (stride < kernel) patches, similar to ViT.

    Input shape:  [B, 1024, 128]  (time × freq from the feature extractor)
    Reshape to:   [B, 1, 128, 1024]  (CHW with freq=H, time=W)
    Unfold:       kernel=(16,16), stride=(10,10) → [B, 256, 1212]
    Transpose:    → [B, 1212, 256]

    Args:
        input_values: Spectrogram tensor of shape ``[B, 1024, 128]``.

    Returns:
        Patch tensor of shape ``[B, 1212, 256]``.
    """
    batch_size = input_values.shape[0]

    # Reshape: [B, 1024, 128] → [B, 1, 128, 1024]  (freq as height, time as width)
    spec_2d = input_values.unsqueeze(1).transpose(2, 3)
    assert spec_2d.shape == (batch_size, 1, SPEC_FREQ_BINS, SPEC_TIME_FRAMES), (
        f"Expected shape [{batch_size}, 1, {SPEC_FREQ_BINS}, {SPEC_TIME_FRAMES}], "
        f"got {list(spec_2d.shape)}"
    )

    # Unfold into patches: [B, 256, 1212]
    patches = F.unfold(
        spec_2d,
        kernel_size=(PATCH_SIZE, PATCH_SIZE),
        stride=(PATCH_STRIDE, PATCH_STRIDE),
    )
    assert patches.shape == (batch_size, PATCH_DIM, TOTAL_PATCHES), (
        f"Expected shape [{batch_size}, {PATCH_DIM}, {TOTAL_PATCHES}], "
        f"got {list(patches.shape)}"
    )

    # Transpose to sequence-first: [B, 1212, 256]
    patches = patches.transpose(1, 2).contiguous()
    return patches


def format_top_k(logits: torch.Tensor, id2label: dict, k: int = TOP_K) -> list[tuple[int, str, float]]:
    """Return the top-*k* (class_idx, label, probability) from *logits*.

    Args:
        logits: Raw logits of shape ``[1, NUM_CLASSES]``.
        id2label: Mapping from class index to human-readable label.
        k: Number of top predictions to return.

    Returns:
        List of ``(class_idx, label, probability)`` tuples, sorted by
        descending probability.
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(k, dim=-1)
    results = []
    for i in range(k):
        idx = top_indices[0, i].item()
        label = id2label.get(idx, f"class_{idx}")
        prob = top_probs[0, i].item()
        results.append((idx, label, prob))
    return results


def print_top_k(header: str, predictions: list[tuple[int, str, float]]) -> None:
    """Pretty-print top-k predictions with a header."""
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")
    for rank, (idx, label, prob) in enumerate(predictions, 1):
        print(f"  {rank}. [{idx:>3}] {label:<40s} {prob:>8.4%}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AST (Audio Spectrogram Transformer) TTNN Bring-up Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python demo.py                          # PCC mode, dummy audio\n"
            "  python demo.py --audio_path clip.wav     # PCC mode, real audio\n"
            "  python demo.py --mode generate           # Demo mode with top-k\n"
            "  python demo.py --device_id 1             # Use device 1\n"
        ),
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to an audio file (.wav/.flac/.mp3). If omitted, 1s of random noise is used.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pcc", "generate"],
        default="pcc",
        help="'pcc' = validate PCC against CPU golden (default); 'generate' = run and show predictions.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Tenstorrent device ID (default: 0).",
    )
    return parser.parse_args()


def run_cpu_reference(
    model: ASTForAudioClassification,
    input_values: torch.Tensor,
) -> torch.Tensor:
    """Run the HuggingFace AST model on CPU and return logits.

    Args:
        model: Pre-trained ``ASTForAudioClassification`` in eval mode.
        input_values: Spectrogram tensor of shape ``[1, 1024, 128]``.

    Returns:
        Golden logits tensor of shape ``[1, 527]``.
    """
    print("\n[STEP] Running PyTorch CPU reference inference …")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_values)
    elapsed = time.perf_counter() - t0
    logits = outputs.logits
    print(f"  → logits shape: {list(logits.shape)}")
    print(f"  → CPU inference time: {elapsed:.3f}s")
    assert logits.shape == (1, NUM_CLASSES), (
        f"Expected logits shape [1, {NUM_CLASSES}], got {list(logits.shape)}"
    )
    return logits


def run_ttnn_inference(
    model_params: dict,
    patches: torch.Tensor,
    device: ttnn.Device,
) -> torch.Tensor:
    """Run the TTNN AST model and return logits as a CPU torch tensor.

    Args:
        model_params: State dict / parameter bundle returned by
            ``create_ast_model``.
        patches: Unfolded patch tensor of shape ``[1, 1212, 256]``.
        device: Open ``ttnn.Device``.

    Returns:
        TTNN logits converted back to a ``torch.Tensor`` of shape
        ``[1, 527]``.
    """
    print("\n[STEP] Running TTNN inference …")

    # Convert patches to ttnn tensor on device (bfloat16, TILE_LAYOUT)
    ttnn_input = ttnn.from_torch(
        patches,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    t0 = time.perf_counter()
    ttnn_logits = model_params["forward"](ttnn_input)
    elapsed = time.perf_counter() - t0

    # Move back to host and convert to torch
    ttnn_logits_host = ttnn.to_torch(ttnn_logits)
    ttnn.deallocate(ttnn_logits)
    ttnn.deallocate(ttnn_input)

    # The TTNN output may be padded to tile-aligned width; slice to [1, 527]
    ttnn_logits_host = ttnn_logits_host[:, :NUM_CLASSES].float()
    print(f"  → TTNN logits shape: {list(ttnn_logits_host.shape)}")
    print(f"  → TTNN inference time: {elapsed:.3f}s")
    return ttnn_logits_host


def validate_results(
    golden_logits: torch.Tensor,
    ttnn_logits: torch.Tensor,
    id2label: dict,
) -> bool:
    """Compare golden and TTNN logits; return True if all checks pass.

    Checks performed:
        1. PCC ≥ ``PCC_THRESHOLD``.
        2. Top-1 predicted class matches.
        3. At least ``TOP_K_OVERLAP_MIN`` of top-5 classes overlap.

    Args:
        golden_logits: CPU reference logits ``[1, 527]``.
        ttnn_logits: TTNN logits ``[1, 527]``.
        id2label: Class-index-to-label mapping from model config.

    Returns:
        ``True`` if **all** checks pass, ``False`` otherwise.
    """
    all_pass = True

    # --- PCC ---
    pcc = compute_pcc(golden_logits, ttnn_logits)
    pcc_pass = pcc >= PCC_THRESHOLD
    status = "PASS ✅" if pcc_pass else "FAIL ❌"
    print(f"\n[CHECK] PCC = {pcc:.6f}  (threshold ≥ {PCC_THRESHOLD})  [{status}]")
    if not pcc_pass:
        all_pass = False

    # --- Top-K predictions ---
    golden_top = format_top_k(golden_logits, id2label, TOP_K)
    ttnn_top = format_top_k(ttnn_logits, id2label, TOP_K)

    print_top_k("PyTorch CPU (Golden) Top-5", golden_top)
    print_top_k("TTNN Device Top-5", ttnn_top)

    # Top-1 match
    top1_match = golden_top[0][0] == ttnn_top[0][0]
    status = "PASS ✅" if top1_match else "FAIL ❌"
    print(f"\n[CHECK] Top-1 match: "
          f"golden='{golden_top[0][1]}' vs ttnn='{ttnn_top[0][1]}'  [{status}]")
    if not top1_match:
        all_pass = False

    # Top-5 overlap (at least TOP_K_OVERLAP_MIN of TOP_K must match)
    golden_set = {t[0] for t in golden_top}
    ttnn_set = {t[0] for t in ttnn_top}
    overlap = golden_set & ttnn_set
    overlap_count = len(overlap)
    overlap_pass = overlap_count >= TOP_K_OVERLAP_MIN
    status = "PASS ✅" if overlap_pass else "FAIL ❌"
    print(f"[CHECK] Top-5 overlap: {overlap_count}/{TOP_K} "
          f"(need ≥ {TOP_K_OVERLAP_MIN})  [{status}]")
    if not overlap_pass:
        all_pass = False

    return all_pass


def main() -> None:
    """Entry point for the AST TTNN bring-up demo."""
    args = parse_args()

    print("=" * 60)
    print("  Audio Spectrogram Transformer (AST) — TTNN Bring-up Demo")
    print(f"  Model : {MODEL_ID}")
    print(f"  Mode  : {args.mode}")
    print(f"  Device: {args.device_id}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load HuggingFace model & feature extractor
    # ------------------------------------------------------------------
    print("\n[STEP] Loading HuggingFace model and feature extractor …")
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    hf_model = ASTForAudioClassification.from_pretrained(MODEL_ID)
    hf_model.eval()
    id2label = hf_model.config.id2label
    print(f"  → Loaded {MODEL_ID}  ({NUM_CLASSES} classes)")

    # ------------------------------------------------------------------
    # 2. Prepare audio input
    # ------------------------------------------------------------------
    print("\n[STEP] Preparing audio input …")
    waveform = load_audio(args.audio_path)
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = inputs.input_values
    print(f"  → Spectrogram shape: {list(input_values.shape)}")
    assert input_values.shape == (1, SPEC_TIME_FRAMES, SPEC_FREQ_BINS), (
        f"Expected input shape [1, {SPEC_TIME_FRAMES}, {SPEC_FREQ_BINS}], "
        f"got {list(input_values.shape)}"
    )

    # ------------------------------------------------------------------
    # 3. CPU reference inference (golden)
    # ------------------------------------------------------------------
    golden_logits = run_cpu_reference(hf_model, input_values)

    # ------------------------------------------------------------------
    # 4. Unfold spectrogram into patches (CPU pre-processing)
    # ------------------------------------------------------------------
    print("\n[STEP] Unfolding spectrogram into patches …")
    patches = unfold_spectrogram(input_values)
    print(f"  → Patches shape: {list(patches.shape)}")
    print(f"  → freq_patches={FREQ_PATCHES}, time_patches={TIME_PATCHES}, "
          f"total_patches={TOTAL_PATCHES}, patch_dim={PATCH_DIM}")

    # ------------------------------------------------------------------
    # 5. Open TTNN device and load model
    # ------------------------------------------------------------------
    print(f"\n[STEP] Opening TTNN device {args.device_id} …")
    device = ttnn.open_device(device_id=args.device_id)
    ttnn.enable_program_cache(device)

    try:
        print("\n[STEP] Loading TTNN AST model …")
        model_params = create_ast_model(
            hf_model,
            device=device,
        )

        # ------------------------------------------------------------------
        # 6. TTNN inference
        # ------------------------------------------------------------------
        ttnn_logits = run_ttnn_inference(model_params, patches, device)

        # ------------------------------------------------------------------
        # 7. Validation / demo output
        # ------------------------------------------------------------------
        if args.mode == "pcc":
            all_pass = validate_results(golden_logits, ttnn_logits, id2label)

            print("\n" + "=" * 60)
            if all_pass:
                print("  🎉  ALL CHECKS PASSED  🎉")
            else:
                print("  ⚠️   SOME CHECKS FAILED  ⚠️")
            print("=" * 60 + "\n")

            # Hard assert in PCC mode so CI pipelines catch regressions
            pcc = compute_pcc(golden_logits, ttnn_logits)
            assert pcc >= PCC_THRESHOLD, (
                f"PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"
            )

        elif args.mode == "generate":
            # In generate mode, just show predictions from both backends
            golden_top = format_top_k(golden_logits, id2label, TOP_K)
            ttnn_top = format_top_k(ttnn_logits, id2label, TOP_K)
            print_top_k("PyTorch CPU Predictions", golden_top)
            print_top_k("TTNN Device Predictions", ttnn_top)

            pcc = compute_pcc(golden_logits, ttnn_logits)
            print(f"\n[INFO] PCC between CPU and TTNN: {pcc:.6f}")

    finally:
        # ------------------------------------------------------------------
        # 8. Cleanup
        # ------------------------------------------------------------------
        print("\n[STEP] Closing TTNN device …")
        ttnn.close_device(device)

    print("[DONE] Demo complete.\n")


if __name__ == "__main__":
    main()
