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
    No-op pipeline step (identity). `pipe([])` curried with data
    returns data unchanged.
  */
  noop = [
    "pipe"
    [ ]
  ];

  /*
    Each unit becomes ONE layer (non-isolate) or TWO layers
    (isolate). We use `subcomponent_out` to claim the unit's full
    closure (no `common` output to deal with — unlike split_paths,
    subcomponent_out gives just `{main, rest}`). Order matters:
    a unit's `subcomponent_out` claims paths from its closure that
    earlier units didn't already peel. So peel foundational units
    FIRST (they appear earlier in the units list).

    Layout:
      isolate=false → one layer for the unit's full closure
      isolate=true  → one layer for the explicit roots, one layer
                      for everything else in their closure

    Compared to the previous popularity-contest design, we
    typically end up with one tenth as many layers — easier on
    docker's manifest ceiling, fewer cache misses to track in
    layered_transfer.py, simpler to reason about for partial
    rebuilds. The trade-off: a single change to any path inside a
    layer invalidates the whole layer's bytes, so a 200 MB
    "tokenizer-python-other" layer with one new wheel costs 200 MB
    on the wire. Pick units accordingly: if a sub-package is
    likely to update independently, give it its own unit (see
    `angr` in our flake.nix).
  */

  /*
    Build the per-unit ops emitted into the pipeline:
      - subcomponent_out roots → {main: closure, rest: bulk}
      - if isolate, also: over main [subcomponent_in roots]
        → main becomes {main: just_roots, rest: deps}
  */
  peelOps =
    unit:
    [
      [
        "subcomponent_out"
        (toRoots unit.roots)
      ]
    ]
    ++ lib.optional (unit.isolate or false) [
      "over"
      "main"
      [
        "subcomponent_in"
        (toRoots unit.roots)
      ]
    ];

  /*
    Recursively chain the units. Each unit's peel emits one or
    two layers, then `over rest` recurses on the remaining graph.
    Last unit terminates the chain: applies its peel + (optional)
    isolate transform, then `over rest [basicsOp]` if basicsOp is
    non-trivial, else just stops.

    "Stops" means: whatever's in `rest` of the last peel becomes
    one layer when `flatten` walks the result dict. That's the
    basics tier when basicsOp is noop.
  */
  chainUnits =
    basicsOp: remainingUnits:
    if remainingUnits == [ ] then
      basicsOp
    else
      let
        unit = builtins.head remainingUnits;
        rest = builtins.tail remainingUnits;
        recurseStep = [
          "over"
          "rest"
          (chainUnits basicsOp rest)
        ];
      in
      [
        "pipe"
        ((peelOps unit) ++ [ recurseStep ])
      ];

  /*
    Optional partial-build helper. Replaces the basics tier's
    "everything in one layer" default with a chain of split_paths
    that preserves each previously-existing layer's content
    grouping. Useful when the basics tier is itself heterogeneous
    enough that you want layer-by-layer cache stability across
    input changes.

    Reads the previous assignment from a JSON file via
    `readAssignmentFromEnv`. Requires `--impure`.
  */
  preserveLayers =
    layers:
    let
      nonEmpty = builtins.filter (paths: paths != [ ]) layers;
    in
    if nonEmpty == [ ] then noop else preserveLayersChain nonEmpty;

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
      basicsOp = if previousAssignment != null then preserveLayers previousAssignment else noop;
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
        recurseStep = [
          "over"
          "rest"
          (chainUnits basicsOp restUnits)
        ];
      in
      ((peelOps firstUnit) ++ [
        recurseStep
        [ "flatten" ]
        [
          "limit_layers"
          maxLayers
        ]
      ]);

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
