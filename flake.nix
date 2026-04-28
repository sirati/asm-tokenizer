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
          semanticLayering = import ./nix/semantic-layering.nix { inherit (pkgs) lib; };

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

          # ── Semantic layer plan ───────────────────────────────────
          #
          # Define each user-facing concern as a "unit" — an ordered
          # list (most-specific first → foundational last). The
          # generic `semantic-layering.nix` helper turns this into
          # a chain of split_paths peels for buildLayeredImage's
          # layeringPipeline arg.
          #
          # Layer-by-layer outcome (bottom-up in the docker manifest):
          #
          #   project_code               (projectFiles only)
          #   rust_wheel                 (dbrs.python-package only)
          #   rust_wheel_exclusive_deps  (popularity-contested)
          #   ghidra + openjdk21         (each in own layer)
          #   tokenizer_python_pkgs      (each in own layer)
          #   basics                     (popularity-contested:
          #                              python interpreter, glibc,
          #                              gcc-lib, openssl, bash,
          #                              coreutils, gtk libs, ...)
          #
          # Partial rebuild: any unit's content is deterministic in
          # its inputs, so unchanged categories produce identical
          # layer.tar bytes → identical sha256 → upload-side blob
          # cache hits in dynamic_batch/packaging/layered_transfer.py.
          # Across input-changing rebuilds, set
          # NIX_DOCKER_LAYER_CACHE=<path-to-prev-assignment.json>
          # and pass `--impure` to stabilise basics-tier popularity
          # ordering — see nix/semantic-layering.nix docstring.

          tokenizerPyPkgRoots = deploymentPythonPackages pkgs.python314.pkgs;

          # Optional impure read of a previous build's layer
          # assignment, captured via
          # `nix/extract-layer-assignment.py`. When unset we use
          # the standard popularity_contest for the basics tier.
          previousAssignment = semanticLayering.readAssignmentFromEnv "NIX_DOCKER_LAYER_CACHE";

          dockerLayeringPipeline = semanticLayering.buildPipeline {
            units = [
              {
                name = "project-code";
                roots = [ projectFiles ];
                isolate = true;
              }
              {
                name = "rust-wheel";
                roots = [ dbrs.python-package ];
                isolate = true;
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
                roots = tokenizerPyPkgRoots;
              }
            ];
            # Cap at 120 to stay under docker's ~127 manifest
            # ceiling. Our closure has ~80 paths; popularity_contest
            # gives one path per layer below the cap.
            maxLayers = 120;
            inherit previousAssignment;
          };

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
