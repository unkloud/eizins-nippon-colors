#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass, field
from functools import cache
from itertools import batched
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


@dataclass(unsafe_hash=True)
class NipponColor:
    kanji_name: str = field(repr=True, hash=False)
    english_name: str = field(repr=False, hash=False)
    hex_rgb: str = field(repr=False, default="", hash=True)
    cmyk: tuple[int, int, int, int] = field(repr=False, default=None, hash=False)

    def __post_init__(self):
        if not self.cmyk and not self.hex_rgb:
            resp = httpx.post(
                "https://nipponcolors.com/php/io.php", data={"color": self.english_name}
            ).json()
            rgb_str, cmyk_str = resp["rgb"], resp["cmyk"]
            self.hex_rgb = rgb_str
            self.cmyk = tuple([int("".join(s)) for s in batched(cmyk_str, 3)])  # noqa

    @classmethod
    def refresh(cls, source_page: str = "https://nipponcolors.com"):
        page = httpx.get(source_page).content
        soup = BeautifulSoup(page, "html5lib")
        colors = soup.find(id="colors")
        all_colors = []
        for color in colors.find_all("li"):
            kanji_name, english_name, *_ = color.text.split(",")
            all_colors.append(cls(kanji_name.strip(), english_name.strip()))
        return all_colors

    @classmethod
    def load(cls, input_file_path: Path):
        colors = json.loads(input_file_path.read_text())
        return [NipponColor(**val) for val in colors]

    def as_markdown(self, element="span") -> str:
        return f"<{element} style='background-color: #{self.hex_rgb}'>{self.kanji_name}, {self.english_name}</{element}>"

    @property
    @cache
    def rgbi(self):
        """Return the color in integer RGB format."""
        return (
            int(self.hex_rgb[0:2], 16),
            int(self.hex_rgb[2:4], 16),
            int(self.hex_rgb[4:6], 16),
        )


def main():
    parser = argparse.ArgumentParser(description="Manage Nippon Colors data.")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )
    refresh_parser = subparsers.add_parser(
        "refresh", help="Refresh data and save to an output file."
    )
    refresh_parser.add_argument(
        "--output",
        "-o",
        default="nippon_colors.json",
        help="Path to the output file (default: nippon_colors.json)",
    )
    refresh_parser.add_argument(
        "--source",
        "-s",
        default="https://nipponcolors.com",
        help="URL of the source page to fetch color data from (default: https://nipponcolors.com)",
    )
    # Define the 'markdown' command.
    markdown_parser = subparsers.add_parser(
        "markdown", help="Generate markdown output from a file."
    )
    markdown_parser.add_argument(
        "file_name",
        help="Path to the input file for markdown generation.",
    )
    markdown_parser.add_argument(
        "--element", default="span", help="HTML element to use for markdown output."
    )
    args = parser.parse_args()
    if args.command == "refresh":
        colors = NipponColor.refresh(source_page=args.source)
        if colors:
            output_data = [color.to_dict() for color in colors]
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
            except IOError as e:
                print(f"Error saving data to file {args.output}: {e}")
        else:
            print("No colors were extracted. Nothing to save.")
    elif args.command == "markdown":
        colors = NipponColor.load(Path(args.file_name))
        for color in colors:
            print(color.as_markdown(args.element))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
