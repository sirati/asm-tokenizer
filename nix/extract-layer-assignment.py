#!/usr/bin/env python3
"""Extract layer-to-store-path assignment from a built docker-archive.

Output format matches what `flatten_references_graph` emits (and
what `semantic-layering.nix::buildPipeline`'s `previousAssignment`
arg expects): a JSON list of lists of /nix/store paths.

Usage:
    nix build .#dockerImage --print-out-paths \\
      | xargs python nix/extract-layer-assignment.py \\
      > .docker-layer-cache.json

Then on the next build:
    NIX_DOCKER_LAYER_CACHE=$PWD/.docker-layer-cache.json \\
      nix build .#dockerImage --impure

The pipeline preserves each previously-existing layer's content
across rebuilds (modulo paths no longer in the closure).

The customisation layer (the topmost layer added by
stream_layered_image, containing config-referenced symlinks and
extraCommands output) is skipped automatically: it has no
/nix/store/ entries by construction, so it produces an empty
layer in the output and is filtered out.
"""

from __future__ import annotations

import json
import sys
import tarfile


def extract_layer_assignment(image_archive_path: str) -> list[list[str]]:
    with tarfile.open(image_archive_path, "r:*") as outer:
        manifest_member = outer.getmember("manifest.json")
        manifest = json.load(outer.extractfile(manifest_member))
        layer_paths = manifest[0]["Layers"]

        layers: list[list[str]] = []
        for layer_member_name in layer_paths:
            store_paths: set[str] = set()
            layer_member = outer.getmember(layer_member_name)
            layer_stream = outer.extractfile(layer_member)
            if layer_stream is None:
                continue
            with tarfile.open(fileobj=layer_stream, mode="r|*") as inner:
                for entry in inner:
                    name = entry.name.lstrip("./")
                    if not name.startswith("nix/store/"):
                        continue
                    parts = name.split("/", 3)
                    if len(parts) >= 3:
                        store_paths.add("/nix/store/" + parts[2])
            if store_paths:
                layers.append(sorted(store_paths))
            # Empty layers (e.g., the customisation layer) are
            # dropped — they contain only host-path entries
            # (/opt/, /app/, /bin/...), no /nix/store/ paths to
            # cache for next build.

    return layers


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: extract-layer-assignment.py <docker-archive.tar.gz>",
              file=sys.stderr)
        return 1
    layers = extract_layer_assignment(sys.argv[1])
    json.dump(layers, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
