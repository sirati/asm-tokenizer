{
  description = "dedup_hashmap - u64 -> u32 and u32 -> u32 hashmaps for asm-tokenizer dedup + writer state";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    # System-independent outputs (the overlay consumers apply).
    {
      overlays.default = import ./nix/overlay.nix;
    }
    //
      # Per-system outputs (packages, devShells).
      flake-utils.lib.eachDefaultSystem (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          dedup-hashmap = pkgs.python3Packages.callPackage ./nix/wheel.nix { };
        in
        {
          packages = {
            inherit dedup-hashmap;
            default = dedup-hashmap;
          };

          devShells.default = pkgs.mkShell {
            name = "dedup-hashmap-dev";

            nativeBuildInputs = with pkgs; [
              rustc
              cargo
              maturin
              pkg-config
            ];

            buildInputs = [ pkgs.python3 ];
          };
        }
      );
}
