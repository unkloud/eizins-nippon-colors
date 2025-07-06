#!/usr/bin/env python3
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from eizins_nippon_colors import EizinsNipponColors

template_path = Path("./").resolve().expanduser()
env = Environment(loader=FileSystemLoader(template_path))
enzins_colors = sorted(
    EizinsNipponColors.enzin_nippon_colors(Path("eizins_nippon_colors.jsonl")),
    key=lambda x: x.piece,
)
template = env.get_template("eizins_nippon_colors.jinja2")
for enzins_color in enzins_colors:
    html_output = template.render(enzins_colors=enzins_colors)
    with open("eizins_nippon_colors.html", "w", encoding="utf-8") as f:
        print(html_output, file=f)
