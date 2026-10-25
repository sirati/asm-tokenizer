{
  description = "tokenizer-async-ready-pool - generic Rust/PyO3 asynchronous preparation pool";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          name = "tokenizer-async-ready-pool-dev";
          nativeBuildInputs = with pkgs; [ rustc cargo rustfmt maturin pkg-config ];
          buildInputs = [ pkgs.python3 ];
        };
      });
}
