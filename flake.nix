{
  description = "Python 3.14 development environment for asm-tokenizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Support multiple systems
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              (python314.withPackages (python-pkgs: with python-pkgs; [
                # Core binary analysis and disassembly
                angr
                capstone
                lief
                pyelftools

                intervaltree
                numpy
                pandas
                tqdm
                portalocker

                # Development tools

                pip
                # language servers
                ruff
              ]))

              # normal nix packages
              basedpyright # a language server
              nil
              nixd
            ];

            shellHook = ''
              echo "╔════════════════════════════════════════════════════════════╗"
              echo "║  Python 3.14 development environment (via nixpkgs unstable)║"
              echo "╚════════════════════════════════════════════════════════════╝"
              echo ""
              echo "Python version: $(python --version)"
              echo "Ready to run your scripts!"
              echo ""
            '';
          };
        });
    };
}
