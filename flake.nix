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
          python = pkgs.python314.withPackages (
            python-pkgs: (deploymentPythonPackages python-pkgs) ++ [ dbrs.python-package ]
          );
          inherit (gitignore.lib) gitignoreSource;

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

          # Create a derivation that contains the project files
          projectFiles = pkgs.runCommand "asm-tokenizer-source" { } ''
            mkdir -p $out/app
            cp -r ${projectSource}/. $out/app/
            chmod -R +w $out/app
          '';
        in
        {
          dockerImageBase = pkgs.dockerTools.buildLayeredImage {
            name = "asm-tokenizer-base";
            tag = "latest";
            maxLayers = 3;

            contents = [ python ] ++ (deploymentPackages pkgs) ++ (dockerOnlyPackages pkgs);

            config = {
              Entrypoint = [
                "${python}/bin/python"
                "-m"
              ];
              WorkingDir = "/app";
            };
          };

          dockerImageApp = pkgs.dockerTools.buildLayeredImage {
            name = "asm-tokenizer";
            tag = "latest";
            maxLayers = 6;

            fromImage = self.packages.${system}.dockerImageBase;

            contents = [ projectFiles ];

            config = {
              Entrypoint = [
                "${python}/bin/python"
                "-m"
              ];
              WorkingDir = "/app";
            };
          };

          dockerImage = self.packages.${system}.dockerImageApp;
          default = self.packages.${system}.dockerImageApp;
        }
      );
    };
}
