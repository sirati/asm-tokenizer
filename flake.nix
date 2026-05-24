{
  description = "Python 3.14 development environment for asm-tokenizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/e4bae1bd10c9c57b2cf517953ab70060a828ee6f";
    # External runner: provides `python3Packages.dynamic-runner` via its
    # overlay (replaces the previous in-tree `dynamic-batch-rs` path-flake).
    dynamic-runner.url = "github:sirati/dynamic-runner";
    # Generic semantic-layering helpers + extract-layer-assignment tool
    # (replaces the in-tree `nix/semantic-layering.nix` import).
    nix-docker-layered-image.url = "github:sirati/nix-docker-layered-image/v0.1.0";
    # In-tree pyo3 extension providing the `u64 -> u32` primary dedup
    # map used by `tokenizer.memmap_builder`. Lives in its own
    # subfolder + flake (maturin is scoped there, not in the outer
    # tokenizer dev shell). The overlay below injects
    # `pkgs.python3Packages.dedup-hashmap` (and every other Python
    # version's pkgs) so `python314.pkgs.dedup-hashmap` resolves.
    dedup-hashmap.url = "path:./dedup_hashmap";
    dedup-hashmap.inputs.nixpkgs.follows = "nixpkgs";
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
      dedup-hashmap,
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

      # Patch Ghidra 12.0's broken riscv.opinion (every variant constraint
      # references a deprecated 'RV(32|64)*' name that was retired from
      # riscv.ldefs when 64- and 32-bit subvariants were consolidated to a
      # single variant="default" entry; the result is "No load spec found"
      # for every riscv ELF). Upstream commit 6208df2 ("GP-1 Corrected
      # RISCV import opinion file") rolled the fix into Ghidra 12.0.1.
      # Nixpkgs master is on 12.0.4, but bumping our nixpkgs pin cascades
      # to angr-9.2.193 which requires setuptools-rust we don't provide
      # here; surgical patch is cheaper.
      ghidraOverlay = final: prev: {
        ghidra = prev.ghidra.overrideAttrs (old: {
          postPatch = (old.postPatch or "") + ''
            sed -i -E 's/variant="RV(32|64)[A-Z]+"/variant="default"/g' \
              Ghidra/Processors/RISCV/data/languages/riscv.opinion
          '';
        });
      };

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [
            pyghidraOverlay
            ghidraOverlay
            # Injects `dynamic-runner` into every Python package set,
            # so `pkgs.python314.pkgs.dynamic-runner` is in scope.
            dynamic-runner.overlays.default
            # Injects `dedup-hashmap` (in-tree subflake) into every
            # Python package set, so `pkgs.python314.pkgs.dedup-hashmap`
            # is in scope.
            dedup-hashmap.overlays.default
          ];
        };

      # Package definitions
      deploymentPythonPackages =
        python-pkgs:
        (with python-pkgs; [
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
          xxhash
        ])
        # `dedup-hashmap` has a dash in its attr name; `with python-pkgs;`
        # would parse the bare identifier as subtraction, so it ships via
        # explicit attribute access alongside the `with`-block.
        ++ [ python-pkgs.dedup-hashmap ];

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
              # Point dynamic_runner's PodmanPackaging at the upstream
              # nix-docker-layered-image extractor. The framework's
              # legacy fallback path `<root>/nix/extract-layer-assignment.py`
              # has been stale since the 2026-04-29 extractor split;
              # this env-var lookup wins over that fallback. Fixed
              # framework-side at dynamic-runner 0d1b6b7.
              export DYNRUNNER_LAYER_EXTRACTOR_SCRIPT=${nix-docker-layered-image.packages.${system}.extract-layer-assignment}/bin/extract-layer-assignment
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
          # In-tree pyo3 wheel for the memmap-builder primary dedup map.
          dedupHashmapWheel = pkgs.python314.pkgs.dedup-hashmap;
          inherit (gitignore.lib) gitignoreSource;
          semanticLayering = nix-docker-layered-image.lib.${system}.semanticLayering;

          # Python wrapper carries everything the container needs:
          # angr, ghidra/pyghidra, numpy/pandas/sympy, AND
          # dynamic-runner. Including dynamic-runner in withPackages
          # means the wrapper resolves it via its own site-packages
          # (no PYTHONPATH injection at the OCI layer). The
          # layering pipeline below isolates `runnerWheel` into its
          # own layer, so wheel updates only re-upload the wheel
          # blob plus the (tiny) wrapper script — not the 1+ GB
          # numpy/pandas/angr closure.
          bulkPython = pkgs.python314.withPackages (
            python-pkgs:
            (deploymentPythonPackages python-pkgs)
            ++ [ python-pkgs.dynamic-runner ]
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
                  "dynrunner"
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

          # `mkShell` derivation that captures the runtime env we want
          # the container to launch into. We pass its drvAttrs through
          # `unstructuredDerivationInputEnv` to extract the env vars
          # nix-shell would set, then bake them into the image's OCI
          # Env. At container start, the entrypoint sources
          # `$stdenv/setup` which processes each input package's
          # setupHook (so e.g. `openjdk21`'s setupHook sets
          # `JAVA_HOME=$out/lib/openjdk` automatically — without us
          # hardcoding the `/lib/openjdk` subpath), then runs our
          # shellHook for GHIDRA_INSTALL_DIR. This is the
          # dockerTools.buildNixShellImage approach minus its custom
          # base image, so we keep the layered-transfer optimization.
          containerShell = pkgs.mkShell {
            packages = (deploymentPackages pkgs) ++ (dockerOnlyPackages pkgs);
            shellHook = ''
              export GHIDRA_INSTALL_DIR="${pkgs.ghidra}/lib/ghidra"
            '';
          };
          containerShellEnv =
            pkgs.devShellTools.unstructuredDerivationInputEnv {
              inherit (containerShell) drvAttrs;
            }
            // pkgs.devShellTools.derivationOutputEnv {
              outputList = containerShell.outputs;
              outputMap = containerShell;
            };
          # Activation rcfile: replicates the slice of `$stdenv/setup`
          # that processes setupHooks for our nativeBuildInputs, without
          # pulling stdenv's gcc/binutils/glibc closure (~400 MB) into
          # the runtime image. The OCI Env (extracted from
          # `containerShell.drvAttrs` via
          # `devShellTools.unstructuredDerivationInputEnv`) supplies
          # `nativeBuildInputs`, `buildInputs`, `shellHook`, etc.
          #
          # For each input package we walk `nix-support/`'s propagated
          # transitive closure, source `setup-hook` if present, and
          # prepend `bin`/`sbin` to PATH. Setup-hooks that register
          # callbacks via stdenv's `addEnvHooks` (e.g.
          # `set-java-classpath-hook`) become inert — those callbacks
          # only fire during nix builds. Setup-hooks that directly
          # export env vars (e.g. openjdk21's JAVA_HOME export) work
          # unchanged. Then we eval the user `shellHook` (which sets
          # GHIDRA_INSTALL_DIR) and exec the python entrypoint.
          containerEntrypointRc = pkgs.writeText "asm-tokenizer-rc.sh" ''
            unset PATH
            addEnvHooks() { :; }
            addToSearchPath() {
              local var="$1" path="$2"
              [ -d "$path" ] && export "$var=$path''${!var:+:''${!var}}"
            }
            __processed=" "
            __processPkg() {
              local pkg="$1"
              case "$__processed" in *" $pkg "*) return;; esac
              __processed="$__processed$pkg "
              local prop_file
              for prop_file in propagated-native-build-inputs propagated-build-inputs; do
                if [ -e "$pkg/nix-support/$prop_file" ]; then
                  local prop __contents
                  __contents=$(< "$pkg/nix-support/$prop_file")
                  for prop in $__contents; do
                    __processPkg "$prop"
                  done
                fi
              done
              if [ -e "$pkg/nix-support/setup-hook" ]; then
                source "$pkg/nix-support/setup-hook"
              fi
              local sub
              for sub in bin sbin; do
                [ -d "$pkg/$sub" ] && PATH="$pkg/$sub''${PATH:+:}$PATH"
              done
            }
            for pkg in $nativeBuildInputs $buildInputs; do
              __processPkg "$pkg"
            done
            export PATH
            eval "$shellHook"
            exec ${bulkPython}/bin/python -m "$@"
          '';

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
                name = "dedup-hashmap-wheel";
                roots = [ dedupHashmapWheel ];
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
            # fakeNss provides /etc/passwd + /etc/group with a `root`
            # entry, so getpwuid(0) succeeds inside the container.
            # Without it Java's `System.getProperty("user.home")`
            # returns "?" and LaunchSupport exits 1 (`User home
            # directory does not exist: ?`). `extraCommands` creates
            # `/root` (HOME) since fakeNss only ships passwd entries.
            contents =
              [
                bulkPython
                projectFiles
                pkgs.dockerTools.fakeNss
              ]
              ++ (deploymentPackages pkgs)
              ++ (dockerOnlyPackages pkgs);
            extraCommands = ''
              mkdir -p root tmp
              chmod 1777 tmp
            '';
            config = {
              Entrypoint = [
                "${pkgs.bash}/bin/bash"
                "${containerEntrypointRc}"
              ];
              Env =
                (pkgs.lib.mapAttrsToList (n: v: "${n}=${toString v}") containerShellEnv)
                ++ [ "HOME=/root" ];
              WorkingDir = "/app";
            };
          };

          default = self.packages.${system}.dockerImage;
        }
      );

      apps = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          # Dedicated Python env for the TUI inspector. Carries
          # `textual` ON TOP OF the deployment package set so the
          # inspector can `import textual` while the default
          # `nix develop` shell remains textual-free.
          tuiPython = pkgs.python314.withPackages (
            python-pkgs: (deploymentPythonPackages python-pkgs) ++ [ python-pkgs.textual ]
          );
        in
        {
          tui-inspector = {
            type = "app";
            program = toString (
              pkgs.writeShellScript "tui-inspector" ''
                exec ${tuiPython}/bin/python -m tokenizer.inspector "$@"
              ''
            );
          };
        }
      );
    };
}
