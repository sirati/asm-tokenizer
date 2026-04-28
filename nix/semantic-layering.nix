/*
  Generalized "semantic layering" pipeline builder for
  `dockerTools.buildLayeredImage`'s `layeringPipeline` argument.

  Most projects building docker images via nixpkgs
  `buildLayeredImage` rely on the default popularity-contest
  algorithm to assign store paths to layers. That works fine until
  you want to:

    1. Guarantee that specific things end up in their own layer
       (for caching: rust wheel, project source, ghidra+jdk, etc.).
    2. Avoid re-uploading huge layers when only a small part
       changes — see the layered_transfer.py blob-cache scheme
       in this repo's dynamic_batch/packaging/.
    3. Survive nixpkgs bumps without random layer reshuffling
       (which invalidates the gateway's blob cache).

  This module turns those concerns into a declarative API:

    layers = semanticLayering.buildPipeline {
      units = [
        { name = "project-code";  roots = [ projectFiles ];      isolate = true; }
        { name = "rust-wheel";    roots = [ wheel ];             isolate = true; }
        { name = "ghidra";        roots = [ ghidra openjdk21 ]; }
        { name = "python-pkgs";   roots = [ angr numpy pandas sympy ... ]; }
        # whatever's left becomes the "basics" tier
      ];
      maxLayers = 120;
    };

    image = pkgs.dockerTools.buildLayeredImage {
      name = "my-app";
      contents = [...];
      config = {...};
      layeringPipeline = layers;
    };

  Order matters: list units MOST-SPECIFIC FIRST, foundational
  things LAST. Each unit's split_paths peel removes its closure
  from the remaining graph that subsequent peels see; whatever
  remains after the last unit becomes the "basics" tier.

  The unit-spec fields:

    name      String. Used for documentation only — doesn't affect
              layer content.
    roots     List of derivations or store paths to peel into this
              unit's layer(s).
    isolate   Optional bool, default false.

              true: split the unit's *explicit roots* into ONE
                    layer, separate from their exclusive deps which
                    get popularity-contested into individual
                    layers. Use for: a single leaf you want alone
                    in one layer (rust wheel, project source).

              false: popularity-contest the entire unit. Each path
                     in the unit's exclusive closure ends up in its
                     own layer. Use for: a category containing
                     multiple independent big things you want
                     individually layered (ghidra+openjdk; many
                     python packages).

  Returns: a list of pipeline ops suitable to pass directly as the
  `layeringPipeline` argument to `dockerTools.buildLayeredImage` (or
  `streamLayeredImage`).

  ─── Partial builds ─────────────────────────────────────────────

  Pass `previousAssignment` (a list of lists of store paths from a
  prior build's pipeline output) to stabilise the BASICS tier
  across rebuilds. Without this, a small input change can shift
  popularity ordering for the basics tier, causing many layers to
  re-hash and invalidating the upload-side cache. With it, each
  previously-existing layer is reconstructed by an explicit
  split_paths so its content stays stable; only NEW paths land in
  popularity-contested leftover layers.

  Reading the previous assignment requires `--impure` (nix needs
  to read a non-store file). Use `extract-layer-assignment.py` in
  this directory to dump a built image's layer assignment, then
  feed it back via env var:

    nix build .#dockerImage --print-out-paths \
      | xargs python ${./extract-layer-assignment.py} \
      > .docker-layer-cache.json

    NIX_DOCKER_LAYER_CACHE=$PWD/.docker-layer-cache.json \
      nix build .#dockerImage --impure

  Inside `flake.nix`:

    previousAssignment =
      let p = builtins.getEnv "NIX_DOCKER_LAYER_CACHE";
      in if p != "" && builtins.pathExists p
         then builtins.fromJSON (builtins.readFile p)
         else null;
*/
{ lib }:
let
  toRoots = roots: map (p: "${p}") roots;

  /*
    Op applied to a unit's "main" subgraph (the unit's
    exclusive content after split_paths peel).

    `isolate=true` → subcomponent_in puts JUST the explicit roots
    into one layer; remaining exclusive deps get
    popularity-contested into individual layers.

    `isolate=false` → popularity_contest splits the entire main
    subgraph (roots + exclusive deps) into one layer per path.
  */
  unitMainOp = unit:
    if unit.isolate or false then
      [
        "pipe"
        [
          [
            "subcomponent_in"
            (toRoots unit.roots)
          ]
          [
            "over"
            "rest"
            [ "popularity_contest" ]
          ]
        ]
      ]
    else
      [ "popularity_contest" ];

  /*
    Recursively chain the basics tier into a sequence of explicit
    `split_paths` operations, one per previously-existing layer.
    This keeps each old layer's exact content together (modulo
    paths that no longer exist in the current closure), so the
    upload-side blob cache hits across rebuilds.

    Termination: the chain ends with the LAST layer's split_paths
    alone — no `over rest` after it. Whatever ends up in that
    final `rest` (i.e., paths added to the closure since the
    previous build) becomes ONE merged layer when `flatten` walks
    the result. We can't append `over rest [popularity_contest]`
    at the bottom: split_paths might consume all remaining graph,
    omitting `rest` from the result dict, which would crash the
    next stage with KeyError. So we accept a coarser layer for
    newly-added closure paths — usually rare and small if you're
    rebuilding the same project.

    To get fresh popularity_contest layering for new paths, just
    run one build without NIX_DOCKER_LAYER_CACHE set, then save
    the resulting assignment for subsequent rebuilds.
  */
  preserveLayers =
    layers:
    let
      nonEmpty = builtins.filter (paths: paths != [ ]) layers;
    in
    if nonEmpty == [ ] then
      [ "popularity_contest" ]
    else
      preserveLayersChain nonEmpty;

  preserveLayersChain =
    layers:
    if builtins.length layers == 1 then
      [
        "split_paths"
        (builtins.head layers)
      ]
    else
      let
        paths = builtins.head layers;
        rest = builtins.tail layers;
      in
      [
        "pipe"
        [
          [
            "split_paths"
            paths
          ]
          [
            "over"
            "rest"
            (preserveLayersChain rest)
          ]
        ]
      ];

  /*
    Recursively peel remaining units from the "rest" subgraph of
    the previously-peeled split. Base case applies the basics-tier
    op (popularity_contest, or a previousAssignment-preserving
    chain).
  */
  chainUnits = basicsOp: remainingUnits:
    if remainingUnits == [ ] then
      basicsOp
    else
      let
        unit = builtins.head remainingUnits;
        rest = builtins.tail remainingUnits;
      in
      [
        "pipe"
        [
          [
            "split_paths"
            (toRoots unit.roots)
          ]
          [
            "over"
            "main"
            (unitMainOp unit)
          ]
          [
            "over"
            "rest"
            (chainUnits basicsOp rest)
          ]
        ]
      ];
in
{
  /*
    Build a `layeringPipeline` for `dockerTools.buildLayeredImage`
    that places each semantic unit in its own layer(s).

    Args:
      units              ordered list of unit specs.
      maxLayers          int, default 120. Caps total layer count.
                         Anything past this gets merged into one
                         tail layer; keep this above your unit
                         count + basics-tier path count to avoid
                         cache-unfriendly merging.
      previousAssignment optional. List of lists of store paths
                         from a previous build's pipeline output.
                         If provided, the basics tier preserves
                         these layer groupings instead of running
                         a fresh popularity_contest. Useful for
                         partial-build cache stability.
  */
  buildPipeline =
    {
      units,
      maxLayers ? 120,
      previousAssignment ? null,
    }:
    let
      basicsOp =
        if previousAssignment != null then
          preserveLayers previousAssignment
        else
          [ "popularity_contest" ];
    in
    if units == [ ] then
      [
        basicsOp
        [ "flatten" ]
        [
          "limit_layers"
          maxLayers
        ]
      ]
    else
      let
        firstUnit = builtins.head units;
        restUnits = builtins.tail units;
      in
      [
        [
          "split_paths"
          (toRoots firstUnit.roots)
        ]
        [
          "over"
          "main"
          (unitMainOp firstUnit)
        ]
        [
          "over"
          "rest"
          (chainUnits basicsOp restUnits)
        ]
        [ "flatten" ]
        [
          "limit_layers"
          maxLayers
        ]
      ];

  /*
    Convenience: read a previous layer assignment from a JSON file
    path provided via the NIX_DOCKER_LAYER_CACHE env var. Returns
    null if the env var is unset or the file doesn't exist.

    Use with `--impure` (the env var read is impure). The file
    format is a JSON list of lists of /nix/store paths — see
    `extract-layer-assignment.py` for how to produce it.
  */
  readAssignmentFromEnv =
    envVarName:
    let
      path = builtins.getEnv envVarName;
    in
    if path == "" then
      null
    else if builtins.pathExists path then
      builtins.fromJSON (builtins.readFile path)
    else
      null;
}
