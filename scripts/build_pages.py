"""Validate and stage the public Pages site without uploading the repository."""
import argparse
from collections import deque
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"index.html", "README.md", "LICENSE"}
PUBLIC_ASSETS = {"site.css", "theme.js", "site.js"}
DOWNLOADS = {"reduced_vgg_retrained.pth", "reduced_vgg_fp32.pth", "export_manifest.json"}
DOC_EXTENSIONS = {".md", ".json", ".csv", ".txt", ".png", ".svg", ".npz"}


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            identifier = attributes["id"]
            if identifier in self.ids:
                raise ValueError(f"Duplicate HTML id: {identifier}")
            self.ids.add(identifier)
        self.links.extend(value for key, value in attrs if key in {"href", "src"} and value)


def public_file(path):
    parts = path.parts
    return (path.as_posix() in ROOT_FILES or
            len(parts) == 2 and parts[0] == "assets" and parts[1] in PUBLIC_ASSETS or
            len(parts) == 2 and parts[0] == "visualizations" and path.suffix == ".html" or
            len(parts) == 2 and parts[0] == "docs" and path.suffix in DOC_EXTENSIONS or
            len(parts) == 2 and parts[0] == "weights" and parts[1] in DOWNLOADS)


def referenced_links(path):
    if path.suffix == ".html":
        parser = References()
        parser.feed(path.read_text(encoding="utf-8"))
        return parser.links
    if path.suffix == ".md":
        source = path.read_text(encoding="utf-8")
        source = re.sub(r"```.*?```", "", source, flags=re.S)
        return re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", source)
    return []


def collect_public_files(root):
    root = Path(root).resolve()
    queue = deque([root / "index.html", root / "LICENSE"])
    selected, external = set(), set()
    while queue:
        current = queue.popleft()
        if not current.resolve().is_relative_to(root):
            raise ValueError(f"Public file resolves outside the repository: {current.name}")
        if current in selected:
            continue
        relative = current.relative_to(root)
        if not public_file(relative):
            raise ValueError(f"Reference is outside the public allowlist: {relative}")
        if not current.is_file():
            raise ValueError(f"Missing public file: {relative}")
        selected.add(current)
        for link in referenced_links(current):
            target = urlsplit(link)
            if target.scheme or target.netloc:
                external.add(link)
                continue
            if not target.path:
                continue
            if target.path.startswith("/") or "\\" in target.path:
                raise ValueError(f"Link must work beneath a project subpath: {relative}: {link}")
            destination = (current.parent / unquote(target.path)).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Link escapes the repository: {relative}: {link}")
            queue.append(destination)
    return sorted(selected), sorted(external)


def build_site(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root.is_relative_to(output):
        raise ValueError("Site output must not replace the source directory or its parent")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Site output must be a new or empty directory")
    files, external = collect_public_files(root)
    manifest = {"source_revision": os.environ.get("GITHUB_SHA", "local-working-tree"),
                "files": [], "external_dependencies": external}
    output.mkdir(parents=True, exist_ok=True)
    for source in files:
        relative = source.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["files"].append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (output / "site-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staged, _ = collect_public_files(output)
    if len(staged) != len(files):
        raise ValueError("Staged public file graph differs from the source")
    total_bytes = sum(item["bytes"] for item in manifest["files"])
    print(f"Pages build complete: {len(files)} public files, {total_bytes:,} bytes")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build_site(parser.parse_args().output)
