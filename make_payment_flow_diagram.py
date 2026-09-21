"""Generate a payment-flow diagram PNG for the credit card case study appendix.

The diagram mirrors §3.2 / §3.3 of the case study: card data flows from the
customer's browser into the payment gateway, the gateway returns an opaque
token to the merchant, and authorisation / 3-D Secure / SCA happens between
the gateway, the card network, and the issuing bank.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "payment_flow_diagram.png"

# Palette tuned for printability and AAA contrast.
COL_BOX = "#1F3A5F"
COL_TEXT = "white"
COL_CARD = "#0F7B4F"   # card data path
COL_TOKEN = "#A04800"  # token / auth path
COL_BG = "white"
COL_EDGE = "#0E1B2C"


def box(ax, x, y, w, h, label, sub=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        linewidth=1.2, edgecolor=COL_EDGE, facecolor=COL_BOX,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2 + (0.05 if sub else 0), label,
            ha="center", va="center", color=COL_TEXT, fontsize=11, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18, sub,
                ha="center", va="center", color=COL_TEXT, fontsize=8.5, style="italic")


def arrow(ax, x1, y1, x2, y2, *, color, label, offset=0.06, label_offset=(0, 0.15)):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color=color,
    )
    ax.add_patch(a)
    ax.text((x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label, ha="center", va="center",
            fontsize=8.5, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COL_BG)

    # Top row: customer -> merchant checkout -> payment gateway
    box(ax, 0.2, 4.8, 2.2, 1.0, "Customer browser", sub="(TLS 1.3, HSTS)")
    box(ax, 3.6, 4.8, 2.6, 1.0, "Merchant checkout page", sub="(hosted iframes)")
    box(ax, 7.3, 4.8, 2.4, 1.0, "Payment gateway", sub="(PCI DSS Level 1)")

    # Bottom row: card network -> issuing bank -> merchant server
    box(ax, 7.3, 1.6, 2.4, 1.0, "Card network", sub="(Visa / Mastercard / ...)")
    box(ax, 4.0, 0.2, 2.6, 1.0, "Issuing bank", sub="(performs SCA / 3DS)")
    box(ax, 0.6, 0.2, 2.6, 1.0, "Merchant server", sub="(no card data)")

    # Card data path: customer -> merchant -> gateway
    arrow(ax, 2.4, 5.7, 3.6, 5.7, color=COL_CARD, label="card data (never touches merchant JS)")
    arrow(ax, 6.2, 5.7, 7.3, 5.7, color=COL_CARD, label="card data (PAN)")

    # Token path: gateway -> merchant checkout
    arrow(ax, 7.3, 4.4, 4.9, 4.4, color=COL_TOKEN, label="opaque token", label_offset=(0, 0.25))

    # Merchant server uses the token (token path continues into merchant server)
    arrow(ax, 4.0, 4.0, 2.4, 1.7, color=COL_TOKEN, label="token used for auth", label_offset=(-0.2, 0.4))

    # Auth path (down then up via card network): merchant server -> issuing bank
    arrow(ax, 3.2, 0.7, 4.0, 0.7, color=COL_TOKEN, label="auth request")
    arrow(ax, 6.6, 0.7, 7.3, 1.7, color=COL_TOKEN, label="forwarded", label_offset=(0.25, 0.2))

    # Gateway -> Card network: auth request down the right edge
    arrow(ax, 8.85, 4.8, 8.85, 2.6, color=COL_TOKEN, label="auth request", label_offset=(0.6, 0))

    # Card network -> Issuing bank: response coming back left
    arrow(ax, 7.3, 1.2, 6.6, 1.2, color=COL_CARD, label="result", label_offset=(0, -0.2))
    arrow(ax, 4.0, 1.2, 3.2, 1.2, color=COL_CARD, label="result", label_offset=(0, -0.2))

    # Legend
    ax.plot([0.2, 0.9], [6.6, 6.6], color=COL_CARD, linewidth=2.0)
    ax.text(1.0, 6.6, "card data (PAN) flow", va="center", fontsize=9)
    ax.plot([0.2, 0.9], [6.25, 6.25], color=COL_TOKEN, linewidth=2.0)
    ax.text(1.0, 6.25, "token / auth flow", va="center", fontsize=9)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=COL_BG)
    print(f"Wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()