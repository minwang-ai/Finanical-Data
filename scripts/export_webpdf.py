#!/usr/bin/env python3
"""Convert notebooks to WebPDF with configurable paper size."""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import nbformat
from nbconvert.exporters.webpdf import IS_WINDOWS, WebPDFExporter


class SizedWebPDFExporter(WebPDFExporter):
    """WebPDF exporter that forces a specific paper size."""

    def __init__(self, page_format: str = "A3", **kwargs):
        super().__init__(**kwargs)
        self._page_format = page_format

    def run_playwright(self, html: str) -> bytes:  # noqa: C901 - inherited complexity
        """Run playwright with custom page format."""

        async def main(temp_file: tempfile.NamedTemporaryFile):
            args = ["--no-sandbox"] if self.disable_sandbox else []
            try:
                from playwright.async_api import async_playwright  # type: ignore[import]
            except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
                msg = (
                    "Playwright is not installed to support Web PDF conversion. "
                    "Please install `nbconvert[webpdf]` to enable."
                )
                raise RuntimeError(msg) from exc

            if self.allow_chromium_download:
                cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
                subprocess.check_call(cmd)  # noqa: S603, S607

            playwright = await async_playwright().start()
            chromium = playwright.chromium

            try:
                browser = await chromium.launch(
                    handle_sigint=False,
                    handle_sigterm=False,
                    handle_sighup=False,
                    args=args,
                )
            except Exception as exc:  # pragma: no cover - environment guard
                msg = (
                    "No suitable chromium executable found on the system. "
                    "Please use '--allow-chromium-download' to allow downloading one, "
                    "or install it using `playwright install chromium`."
                )
                await playwright.stop()
                raise RuntimeError(msg) from exc

            page = await browser.new_page()
            await page.emulate_media(media="print")
            await page.wait_for_timeout(100)
            await page.goto(f"file://{temp_file.name}", wait_until="networkidle")
            await page.wait_for_timeout(100)

            pdf_params: dict[str, object] = {"print_background": True, "format": self._page_format}
            if not self.paginate:
                dimensions = await page.evaluate(
                    """() => {
                    const rect = document.body.getBoundingClientRect();
                    return {
                        width: Math.ceil(rect.width) + 1,
                        height: Math.ceil(rect.height) + 1,
                    }
                }"""
                )
                width = dimensions["width"]
                height = dimensions["height"]
                pdf_params.update({
                    "width": min(width, 200 * 72),
                    "height": min(height, 200 * 72),
                })

            pdf_data = await page.pdf(**pdf_params)

            await browser.close()
            await playwright.stop()
            return pdf_data

        pool = concurrent.futures.ThreadPoolExecutor()
        temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        with temp_file:
            temp_file.write(html.encode("utf-8"))
        try:
            def run_coroutine(coro):
                loop = (
                    asyncio.ProactorEventLoop() if IS_WINDOWS else asyncio.new_event_loop()
                )
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)

            pdf_data = pool.submit(run_coroutine, main(temp_file)).result()
        finally:
            os.unlink(temp_file.name)
        return pdf_data


def convert_notebook(notebook_path: Path, output_path: Path, page_format: str, allow_chromium_download: bool) -> None:
    exporter = SizedWebPDFExporter(page_format=page_format)
    exporter.allow_chromium_download = allow_chromium_download
    exporter.disable_sandbox = True

    with notebook_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    pdf_data, _ = exporter.from_notebook_node(nb)
    output_path.write_bytes(pdf_data)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Jupyter notebooks to PDF with a custom page size.")
    parser.add_argument("notebooks", nargs="+", help="Notebook files to export.")
    parser.add_argument(
        "--page-format",
        default="A3",
        choices=["A2", "A3", "A4", "Letter", "Legal"],
        help="Paper size to pass to Chromium printing (default: A3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write PDFs to. Defaults to the notebook's directory.",
    )
    parser.add_argument(
        "--allow-chromium-download",
        action="store_true",
        help="Allow Playwright to download a compatible Chromium build if needed.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    notebooks = [Path(p).resolve() for p in args.notebooks]

    for nb_path in notebooks:
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")
        target_dir = args.output_dir.resolve() if args.output_dir else nb_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / (nb_path.stem + ".pdf")
        convert_notebook(nb_path, output_path, args.page_format, args.allow_chromium_download)
        print(f"Wrote {output_path.relative_to(Path.cwd())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
