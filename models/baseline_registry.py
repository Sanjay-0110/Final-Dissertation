"""
baseline_registry.py
=====================
Stage 1 output, structured for Stage 2 consumption.

Rather than a static literature list, this is a queryable registry so Stage 2
("pick 3-5 baselines spanning design philosophies, with available code") can
be done programmatically and reproducibly -- filter by category, code
availability, and param budget instead of eyeballing a paper list.

Fields:
  name           - model name as used in its paper
  year           - publication year
  category       - design-philosophy bucket (see CATEGORIES below)
  venue          - publication venue
  params_m       - approx. parameters in millions (None if unknown/unverified)
  has_public_code- bool, verified as of this review. Re-check before relying
                    on it -- repos disappear/get archived.
  code_url       - GitHub/project URL if has_public_code else None
  relevance      - why it matters for THIS project specifically, not generic
                    praise. Every entry ties back to domain-robustness or
                    compute-efficiency, per your instruction.
  cross_domain_tested - bool, whether the ORIGINAL paper reports external
                    (unseen-dataset) evaluation, which tells you whether you
                    can sanity-check your reproduction against a published
                    cross-domain number.
"""

from dataclasses import dataclass, field
from typing import Optional, List

CATEGORIES = {
    "classic_unet": "Classic U-shaped encoder-decoder",
    "attention_decoder": "PraNet-style parallel decoder + reverse/boundary attention (CNN)",
    "transformer": "Pyramid-vision-transformer or hybrid CNN-transformer encoder",
    "ultra_lightweight": "Sub-1M-parameter, edge/CPU-targeted",
    "domain_generalization": "Explicit domain-generalization mechanism (not just efficiency)",
}


@dataclass
class PaperEntry:
    name: str
    year: int
    category: str
    venue: str
    params_m: Optional[float]
    has_public_code: bool
    code_url: Optional[str]
    relevance: str
    cross_domain_tested: bool
    notes: str = ""


REGISTRY: List[PaperEntry] = [
    PaperEntry(
        name="UNet++",
        year=2018,
        category="classic_unet",
        venue="DLMIA workshop @ MICCAI",
        params_m=9.0,
        has_public_code=True,
        code_url="https://github.com/MrGiovanni/UNetPlusPlus",
        relevance="Standard efficiency/accuracy floor. Every polyp-seg paper "
                  "benchmarks against it -- needed so your numbers are "
                  "externally comparable, not just internally consistent.",
        cross_domain_tested=False,
        notes="No color-normalization or compression mechanism -- pure "
              "architecture baseline, uniform capacity throughout.",
    ),
    PaperEntry(
        name="PraNet",
        year=2020,
        category="attention_decoder",
        venue="MICCAI",
        params_m=32.5,
        has_public_code=True,
        code_url="https://github.com/DengPingFan/PraNet",
        relevance="The most-cited reproducible CNN baseline in this field. "
                  "Any claim of 'competitive with PraNet at a fraction of "
                  "the compute' is immediately legible to reviewers.",
        cross_domain_tested=True,
        notes="Original paper already reports Kvasir-SEG / CVC-ClinicDB / "
              "ETIS-LaribPolypDB numbers -- directly reusable as a published "
              "sanity check for your reproduction.",
    ),
    PaperEntry(
        name="SANet",
        year=2021,
        category="attention_decoder",
        venue="MICCAI",
        params_m=23.9,
        has_public_code=True,
        code_url="https://github.com/weijun88/SANet",
        relevance="Explicitly addresses color-related overfitting: uses a "
                  "color-exchange augmentation strategy to reduce reliance "
                  "on color cues. Directly competing philosophy to your "
                  "color-normalization frontend, worth citing as contrast "
                  "(augmentation vs. normalization).",
        cross_domain_tested=True,
    ),
    PaperEntry(
        name="Polyp-PVT",
        year=2021,
        category="transformer",
        venue="arXiv / CAAI TRIT",
        params_m=25.1,
        has_public_code=True,
        code_url="https://github.com/DengPingFan/Polyp-PVT",
        relevance="Your required transformer baseline. Pyramid Vision "
                  "Transformer encoder gives strong cross-domain results in "
                  "its own paper via global context -- good contrast case: "
                  "does global attention substitute for explicit color "
                  "normalization, or are they complementary.",
        cross_domain_tested=True,
    ),
    PaperEntry(
        name="ColonSegNet",
        year=2021,
        category="classic_unet",
        venue="IEEE Access",
        params_m=5.0,
        has_public_code=True,
        code_url="https://github.com/DebeshJha/ColonSegNet",
        relevance="Positioned explicitly as a real-time/efficient model, "
                  "smaller than PraNet/UNet++. Useful middle ground between "
                  "the classic baseline and the ultra-lightweight tier.",
        cross_domain_tested=False,
        notes="~5M params, moderate not extreme compression.",
    ),
    PaperEntry(
        name="UNeXt",
        year=2022,
        category="ultra_lightweight",
        venue="MICCAI",
        params_m=1.47,
        has_public_code=True,
        code_url="https://github.com/jeya-maria-jose/UNeXt-pytorch",
        relevance="MLP-based, not attention-based -- a genuinely different "
                  "lightweight design philosophy from asymmetric CNN "
                  "compression. Good diversity pick if you want a 5th "
                  "baseline beyond CNN/transformer/ultra-lightweight-CNN.",
        cross_domain_tested=False,
        notes="Originally validated on skin-lesion/ISIC + BUSI, not polyp-"
              "native -- would need retraining on Kvasir-SEG, which you're "
              "doing anyway for fairness, so this is not a blocker.",
    ),
    PaperEntry(
        name="UACANet",
        year=2021,
        category="attention_decoder",
        venue="ACM MM",
        params_m=69.16,
        has_public_code=True,
        code_url="https://github.com/plemeri/UACANet",
        relevance="Uncertainty-augmented context attention -- heavier, "
                  "strong-accuracy CNN reference point if you want an "
                  "'upper bound' comparison alongside the lightweight tier.",
        cross_domain_tested=True,
        notes="Large param count (69M) -- useful as an accuracy ceiling but "
              "not a fair 'lightweight' comparison; label it as such in "
              "your results table.",
    ),
    PaperEntry(
        name="InvNorm",
        year=2022,
        category="domain_generalization",
        venue="arXiv",
        params_m=None,
        has_public_code=False,
        code_url=None,
        relevance="Closest conceptual competitor: a light, learned, "
                  "invertible normalization block for GI domain "
                  "generalization. NOT reproducible without code -- cite as "
                  "related work and contrast (yours is non-learned/fixed, "
                  "theirs is learned/invertible), do not attempt to "
                  "reproduce.",
        cross_domain_tested=True,
        notes="VERIFY code availability again before Stage 2 lock-in; "
              "arXiv-only papers' repos appear/disappear unpredictably.",
    ),
]


def filter_registry(
    category: Optional[str] = None,
    has_public_code: Optional[bool] = None,
    max_params_m: Optional[float] = None,
    cross_domain_tested: Optional[bool] = None,
) -> List[PaperEntry]:
    """Query helper for Stage 2 baseline selection."""
    results = REGISTRY
    if category is not None:
        results = [r for r in results if r.category == category]
    if has_public_code is not None:
        results = [r for r in results if r.has_public_code == has_public_code]
    if max_params_m is not None:
        results = [r for r in results if r.params_m is not None and r.params_m <= max_params_m]
    if cross_domain_tested is not None:
        results = [r for r in results if r.cross_domain_tested == cross_domain_tested]
    return results


def print_summary():
    print(f"{'Name':<14}{'Cat':<20}{'Params(M)':<11}{'Code':<6}{'XDom':<6}")
    for r in REGISTRY:
        p = f"{r.params_m:.2f}" if r.params_m is not None else "?"
        print(f"{r.name:<14}{r.category:<20}{p:<11}{str(r.has_public_code):<6}{str(r.cross_domain_tested):<6}")


if __name__ == "__main__":
    print("Reproducible candidates (has code):")
    print_summary()
    print()
    print("Candidates with public code, spanning categories (Stage-2-ready shortlist):")
    for r in filter_registry(has_public_code=True):
        print(f"  - {r.name} ({r.category}, {r.params_m}M params)")