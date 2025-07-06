#!/usr/bin/env python3

import json
import sys
from base64 import b64encode
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class EizinWork:
    title: str
    desc: str = field(repr=False)
    base64_encoded_img: Optional[str] = field(default=None, repr=False)

    def __lt__(self, other):
        return self.title < other.title

    @classmethod
    def read_profolio(cls, pth: Path, img_root_path: Path):
        soup = BeautifulSoup(pth.read_text(), "html5lib")
        content = soup.find(id="content")
        title = content.find("header").text.strip()
        article = content.find("article")
        desc = article.text.strip()
        img = article.find("img")
        if img:
            img_path = img_root_path / img["src"]
            return cls(title, desc, b64encode(img_path.read_bytes()).decode("utf-8"))
        else:
            return cls(title, desc, None)

    @classmethod
    def generate(cls, file_name="eizin_profolio.json") -> Path:
        site_root_path = Path(__file__).parent.resolve().expanduser()
        catalog = sorted(
            [
                EizinWork.read_profolio(
                    Path(pth).resolve(),
                    site_root_path / "gallery" / "gallery.eizin.co.jp",
                )
                for pth in sys.argv[1:]
            ]
        )
        profolio_path = Path(file_name)
        with profolio_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in catalog], f, indent=4, ensure_ascii=False)

    @classmethod
    def load_profolio(cls, file_name="eizin_profolio.json"):
        profolio_path = Path(file_name)
        with profolio_path.open("r", encoding="utf-8") as f:
            return [EizinWork(**entry) for entry in json.load(f)]
