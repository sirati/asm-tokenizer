{
  description = "Python 3.14 development environment for asm-tokenizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/e4bae1bd10c9c57b2cf517953ab70060a828ee6f";
    dynamic-batch-rs = {
      url = "path:./rust/dynamic_batch";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      dynamic-batch-rs,
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
          overlays = [ pyghidraOverlay ];
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
          dbrs = dynamic-batch-rs.packages.${system};
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                (python314.withPackages (
                  python-pkgs:
                  (deploymentPythonPackages python-pkgs) ++ (devPythonPackages python-pkgs) ++ [ dbrs.python-package ]
                ))
                dbrs.rust-toolchain
                maturin
                rustfmt
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
          dbrs = dynamic-batch-rs.packages.${system};
          inherit (gitignore.lib) gitignoreSource;

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
                  "dynamic_batch"
                  "dynamic_batch_tokenizer"
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
            ln -s ${dbrs.python-package}/lib $out/opt/runner-wheel/lib
          '';

          # Concrete root paths for each semantic category. By
          # listing these as roots for `split_paths`, each peel-off
          # yields a dict `{main, common, rest}` where:
          #   - `main` = closure(roots) - paths reachable from rest
          #             (the category's exclusive content)
          #   - `common` = closure(roots) ∩ closure(rest)
          #                (deps shared with what's not yet peeled)
          #   - `rest`   = paths not reachable from roots
          #
          # Iterating peels in specific-first order gives us
          # non-overlapping layers, lowest peel = topmost layer.
          # See nixpkgs/pkgs/by-name/fl/flatten-references-graph/
          # for the algorithm.

          ghidraRoots = [
            "${pkgs.ghidra}"
            "${pkgs.openjdk21}"
          ];

          # The python packages used by the tokenizer worker (and
          # only those — the wheel and project source live in their
          # own layers above). split_paths on these collects the
          # python-pkgs.<*> derivations + each one's exclusive
          # closure into a "main" subgraph.
          tokenizerPyPkgRoots = map (p: "${p}") (
            deploymentPythonPackages pkgs.python314.pkgs
          );

          # ── Explicit semantic layering ────────────────────────────
          #
          # Pipeline is read top-down as "peel off the most-specific
          # categories first; whatever doesn't fit becomes basics".
          # The order in the pipeline output (after `flatten`) ends
          # up bottom-up in the docker manifest, so the topmost
          # (most-volatile) layer in the image is `project code`.
          #
          # Layer-by-layer breakdown after `flatten` (bottom-up
          # in the image, left-to-right in the pipeline output):
          #
          #   project_code               (projectFiles only)
          #   rust_wheel                 (dbrs.python-package only)
          #   rust_wheel_exclusive_deps  (popularity-contested if non-empty)
          #   rust_wheel_shared          (deps shared with rest of bulk)
          #   ghidra+jdk_main            (popularity-contested)
          #   ghidra_shared              (deps shared with rest of bulk)
          #   tokenizer_python_pkgs      (popularity-contested)
          #   tokenizer_python_shared    (deps shared with basics)
          #   basics                     (popularity-contested:
          #                              python interpreter, glibc,
          #                              gcc-lib, openssl, bash,
          #                              coreutils, ...)
          #
          # Each "split_paths" peel emits {main, common, rest} in
          # dict-insertion order, which is what flatten walks. The
          # `over <key> <pipe>` pattern keeps recursing into "rest"
          # to layer-up the next category.
          #
          # Partial rebuild: when one of these categories changes
          # (e.g. you edit rust source and rebuild the wheel), only
          # the layers whose `main` content changed get a new
          # sha256 of layer.tar. layered_transfer.py keys its
          # gateway-side blob cache by exactly that sha, so the
          # untouched categories upload zero bytes. Determinism
          # comes from nix; the partial-rebuild win comes from the
          # cache.

          dockerLayeringPipeline = [
            # 1. Peel: project code
            [ "split_paths" [ "${projectFiles}" ] ]
            [
              "over"
              "rest"
              [
                "pipe"
                [
                  # 2. Peel: rust wheel
                  [ "split_paths" [ "${dbrs.python-package}" ] ]
                  [
                    "over"
                    "main"
                    [
                      "pipe"
                      [
                        # Separate the wheel from its exclusive deps
                        # — wheel alone in `main`, deps in `rest`.
                        [ "subcomponent_in" [ "${dbrs.python-package}" ] ]
                        [
                          "over"
                          "rest"
                          [ "popularity_contest" ]
                        ]
                      ]
                    ]
                  ]
                  [
                    "over"
                    "rest"
                    [
                      "pipe"
                      [
                        # 3. Peel: ghidra + openjdk
                        [ "split_paths" ghidraRoots ]
                        [
                          "over"
                          "main"
                          [ "popularity_contest" ]
                        ]
                        [
                          "over"
                          "rest"
                          [
                            "pipe"
                            [
                              # 4. Peel: tokenizer python packages
                              [ "split_paths" tokenizerPyPkgRoots ]
                              [
                                "over"
                                "main"
                                [ "popularity_contest" ]
                              ]
                              # 5. Whatever's left = basics
                              # (python interpreter, glibc, openssl,
                              # bash, coreutils, gcc-lib, ...).
                              [
                                "over"
                                "rest"
                                [ "popularity_contest" ]
                              ]
                            ]
                          ]
                        ]
                      ]
                    ]
                  ]
                ]
              ]
            ]
            [ "flatten" ]
            # Cap at 120 to stay below docker's manifest layer
            # ceiling (~127). The semantic peels above produce ~10
            # named layers; the rest is popularity_contest output
            # (one path per layer). 120 leaves headroom for
            # ~110 popularity-only basics layers — plenty.
            [
              "limit_layers"
              120
            ]
          ];

        in
        {
          # Single image. The base/app split is retired; layered
          # transfer (dynamic_batch/packaging/layered_transfer.py)
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
