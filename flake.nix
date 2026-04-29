{
  description = "Python 3.14 development environment for asm-tokenizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/e4bae1bd10c9c57b2cf517953ab70060a828ee6f";
    # External runner: provides `python3Packages.dynamic-runner` via its
    # overlay (replaces the previous in-tree `dynamic-batch-rs` path-flake).
    dynamic-runner.url = "github:sirati/dynamic-runner/b1b97431d0679ed5c8ba7df9c801471e697d61b5";
    # Generic semantic-layering helpers + extract-layer-assignment tool
    # (replaces the in-tree `nix/semantic-layering.nix` import).
    nix-docker-layered-image.url = "github:sirati/nix-docker-layered-image/v0.1.0";
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      dynamic-runner,
      nix-docker-layered-image,
      gitignore,
    }:
    let
      # Support multiple systems
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Add pyghidra (not yet in this nixpkgs pin)
      pyghidraOverlay = final: prev: {
        python314 = prev.python314.override {
          packageOverrides = pyFinal: pyPrev: {
            pyghidra = pyFinal.buildPythonPackage {
              pname = "pyghidra";
              version = "3.0.2";
              pyproject = true;
              src = prev.fetchPypi {
                pname = "pyghidra";
                version = "3.0.2";
                hash = "sha256-ea1P1XHjLzQ88/zb2E/G4zPvGiZHWjqPcrYpqfPIedo=";
              };
              pythonRelaxDeps = [ "jpype1" ];
              build-system = [ pyFinal.setuptools ];
              dependencies = [
                pyFinal.jpype1
                pyFinal.packaging
              ];
              pythonImportsCheck = [ "pyghidra" ];
              doCheck = false;
              meta = {
                description = "Native CPython for Ghidra";
                homepage = "https://pypi.org/project/pyghidra";
                license = prev.lib.licenses.asl20;
              };
            };
          };
        };
      };

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [
            pyghidraOverlay
            # Injects `dynamic-runner` into every Python package set,
            # so `pkgs.python314.pkgs.dynamic-runner` is in scope.
            dynamic-runner.overlays.default
          ];
        };

      # Package definitions
      deploymentPythonPackages =
        python-pkgs: with python-pkgs; [
          # Core binary analysis and disassembly
          angr
          capstone
          lief
          pyelftools
          pyghidra

          intervaltree
          numpy
          pandas
          tqdm
          portalocker
          aioquic
          websockets
        ];

      devPythonPackages =
        python-pkgs: with python-pkgs; [
          pip
          ruff
          pytest
        ];

      deploymentPackages =
        pkgs: with pkgs; [
          ghidra
          openjdk21
          openssl
        ];

      dockerOnlyPackages =
        pkgs: with pkgs; [
          bash
          coreutils
        ];

      devPackages =
        pkgs: with pkgs; [
          basedpyright
          nil
          nixd
          vscode-json-languageserver
          bash-language-server
          package-version-server
        ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                (python314.withPackages (
                  python-pkgs:
                  (deploymentPythonPackages python-pkgs)
                  ++ (devPythonPackages python-pkgs)
                  ++ [ python-pkgs.dynamic-runner ]
                ))
                pkg-config
              ]
              ++ (deploymentPackages pkgs)
              ++ (devPackages pkgs);

            shellHook = ''
              echo "╔════════════════════════════════════════════════════════════╗"
              echo "║  Python 3.14 development environment (via nixpkgs unstable)║"
              echo "╚════════════════════════════════════════════════════════════╝"
              echo ""
              echo "Python version: $(python --version)"
              echo "Ready to run your scripts!"
              export bin_python=$(which python)
              export bin_python3=$(which python3)
              export GHIDRA_INSTALL_DIR="${pkgs.ghidra}/lib/ghidra"
            '';
          };
        }
      );

      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          # Pre-built rust+python wheel from the external dynamic-runner
          # flake (consumed via its overlay; see `pkgsFor`).
          runnerWheel = pkgs.python314.pkgs.dynamic-runner;
          inherit (gitignore.lib) gitignoreSource;
          semanticLayering = nix-docker-layered-image.lib.${system}.semanticLayering;

          # Python WITHOUT the rust wheel — that goes into its own
          # explicit layer (see `dockerImage` below). The bulk python
          # wrapper carries everything else (angr, ghidra/pyghidra,
          # numpy/pandas/sympy, etc.) so updating the rust wheel
          # invalidates only the wheel layer, not this 1+ GB closure.
          bulkPython = pkgs.python314.withPackages (
            python-pkgs: deploymentPythonPackages python-pkgs
          );

          # Restrict app payload to selected source directories and root Python files only
          projectSource = pkgs.lib.cleanSourceWith {
            src = gitignoreSource ./.;
            filter =
              path: type:
              let
                relPath = pkgs.lib.removePrefix (toString ./. + "/") (toString path);
                pathParts = pkgs.lib.splitString "/" relPath;
                topLevel = if pathParts == [ ] then "" else builtins.head pathParts;
                isRootPyFile = builtins.match "[^/]+\\.py" relPath != null;
                allowedTopLevelDirs = [
                  "dynamic_runner_tokenizer"
                  "dynamic_runner_disasm"
                  "preproc"
                  "shared"
                  "tokenizer"
                ];
              in
              if relPath == "" then
                true
              else if type == "directory" then
                builtins.elem topLevel allowedTopLevelDirs
              else
                (builtins.elem topLevel allowedTopLevelDirs) || isRootPyFile;
          };

          # Project files are placed at /app inside the image so the
          # legacy SLURM job script's `--workdir /app` keeps working.
          projectFiles = pkgs.runCommand "asm-tokenizer-source" { } ''
            mkdir -p $out/app
            cp -r ${projectSource}/. $out/app/
            chmod -R +w $out/app
          '';

          # Wheel placed under a fixed path so PYTHONPATH can find
          # it without adding it to bulkPython's wrapper.
          rustWheelTree = pkgs.runCommand "rust-wheel-tree" { } ''
            mkdir -p $out/opt/runner-wheel
            ln -s ${runnerWheel}/lib $out/opt/runner-wheel/lib
          '';

          # ── Semantic layer plan ───────────────────────────────────
          #
          # Each unit becomes ONE or TWO layers (per the external
          # `nix-docker-layered-image` flake's subcomponent_out approach):
          #   isolate=false → 1 layer (unit's full closure)
          #   isolate=true  → 2 layers (roots alone + deps)
          # The "rest" after all unit peels becomes one basics
          # layer (or, with --impure + NIX_DOCKER_LAYER_CACHE, a
          # chain that preserves the previous build's basics-tier
          # layer grouping).
          #
          # Order matters: peel FOUNDATIONAL units first (they end
          # up at the bottom of the docker manifest, and earlier
          # peels claim shared closure paths so later units don't
          # redundantly include e.g. python3+libc).
          #
          # Layer-by-layer outcome (bottom-up in docker manifest):
          #
          #   base-python (1 layer)        - python3 interpreter +
          #                                  its closure: glibc,
          #                                  openssl, libc++, etc.
          #   ghidra (1 layer)             - ghidra + jdk21 + their
          #                                  exclusive transitive
          #                                  closure (gtk/glib/etc.
          #                                  if not already in base)
          #   tokenizer-python (1 layer)   - numpy, pandas, sympy,
          #                                  capstone, lief, etc.
          #                                  AND their exclusive
          #                                  closure under the
          #                                  reduced graph
          #   angr (1 layer)               - angr + angr-only deps
          #                                  (z3, pyvex, etc.)
          #   rust-wheel (2 layers)        - the wheel alone, then
          #                                  its rust-only closure
          #                                  (usually just the wheel
          #                                  since python3 is
          #                                  claimed by base-python)
          #   project-code (1-2 layers)    - projectFiles alone, no
          #                                  closure → 1 layer
          #   basics (1 layer)             - whatever's left:
          #                                  bash, coreutils, etc.
          #                                  (or per-prev-layer
          #                                  groups via --impure)
          #
          # Total: ~7 explicit layers + customisation = 8 layers.
          # Well under docker's ~127 ceiling, well under any
          # podman limit. Each layer's content is deterministic in
          # its unit's nix-store inputs, so partial rebuilds via
          # content-addressed blob cache (see
          # dynamic_runner.packaging.layered_transfer) flow
          # naturally.

          py = pkgs.python314.pkgs;

          # angr is split out from the rest of the python pkgs
          # because it's the largest single python package
          # (~85 MB) and updates independently of the others.
          tokenizerPyOtherRoots = with py; [
            capstone
            lief
            pyelftools
            pyghidra
            intervaltree
            numpy
            pandas
            tqdm
            portalocker
            aioquic
            websockets
          ];

          previousAssignment = semanticLayering.readAssignmentFromEnv "NIX_DOCKER_LAYER_CACHE";

          dockerLayeringPipeline = semanticLayering.buildPipeline {
            units = [
              {
                name = "base-python";
                roots = [ pkgs.python314 ];
              }
              {
                name = "ghidra";
                roots = [
                  pkgs.ghidra
                  pkgs.openjdk21
                ];
              }
              {
                name = "tokenizer-python";
                roots = tokenizerPyOtherRoots;
              }
              {
                name = "angr";
                roots = [ py.angr ];
              }
              {
                name = "rust-wheel";
                roots = [ runnerWheel ];
                isolate = true;
              }
              {
                name = "project-code";
                roots = [ projectFiles ];
                isolate = true;
              }
            ];
            maxLayers = 120;
            inherit previousAssignment;
          };

        in
        {
          # Single image. The base/app split is retired; layered
          # transfer (dynamic_runner.packaging.layered_transfer)
          # provides per-blob upload deduplication on the gateway,
          # so the historical reason for the split (avoid
          # re-uploading a 2.7 GB static base) is now redundant.
          dockerImage = pkgs.dockerTools.buildLayeredImage {
            name = "asm-tokenizer";
            tag = "latest";
            layeringPipeline = pkgs.writeText "asm-tokenizer-pipeline.json" (
              builtins.toJSON dockerLayeringPipeline
            );
            contents =
              [
                bulkPython
                rustWheelTree
                projectFiles
              ]
              ++ (deploymentPackages pkgs)
              ++ (dockerOnlyPackages pkgs);
            config = {
              Entrypoint = [
                "${bulkPython}/bin/python"
                "-m"
              ];
              # The rust wheel lives at /opt/runner-wheel (not in
              # bulkPython's site-packages, on purpose — that's what
              # gives it its own layer). PYTHONPATH adds it to
              # python's import path at startup.
              Env = [
                "PYTHONPATH=/opt/runner-wheel/lib/python3.14/site-packages"
              ];
              WorkingDir = "/app";
            };
          };

          default = self.packages.${system}.dockerImage;
        }
      );
    };
}
