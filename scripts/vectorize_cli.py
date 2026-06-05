#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import click
import cv2

from vectorizer.config import VectorizerConfig
from vectorizer.pipeline import run as pipeline_run


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default=None, help="Output SVG path (default: <input>.svg)")
@click.option("--dpi", default=96, show_default=True, help="Output DPI for coordinate scaling")
@click.option("--no-deskew", is_flag=True, help="Disable automatic deskew")
@click.option("--binarize", default="otsu", show_default=True,
              type=click.Choice(["otsu", "sauvola", "adaptive"]),
              help="Binarisation method")
@click.option("--fidelity", default=0.5, show_default=True,
              help="Fidelity level 0 (clean) – 1 (faithful)")
def cli(input_path: str, output: str | None, dpi: int,
        no_deskew: bool, binarize: str, fidelity: float) -> None:
    """Vectorize a raster architectural drawing to SVG."""
    src = Path(input_path)
    dst = Path(output) if output else src.with_suffix(".svg")

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        click.echo(f"Error: cannot read image '{src}'", err=True)
        sys.exit(1)

    config = VectorizerConfig(
        output_dpi=dpi,
        deskew_enabled=not no_deskew,
        binarize_method=binarize,
        fidelity_level=fidelity,
    )

    click.echo(f"Vectorizing {src} …")
    result = pipeline_run(image, config)

    dst.write_text(result.svg_string, encoding="utf-8")
    click.echo(f"Saved → {dst}")


if __name__ == "__main__":
    cli()
