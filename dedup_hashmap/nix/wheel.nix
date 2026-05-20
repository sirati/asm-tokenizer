{
  lib,
  buildPythonPackage,
  rustPlatform,
  pkg-config,
}:

# Wheel/Python-package derivation for dedup_hashmap.
#
# The pyo3 extension is built via maturin. The resulting Python module
# is `dedup_hashmap` (configured via `[tool.maturin] module-name` in the
# subfolder's pyproject.toml).
#
# `cargoDeps.hash` is the SRI of the vendored Cargo deps; any Cargo.lock
# change invalidates it. To recalibrate: set `hash = lib.fakeHash;`,
# run `nix build .#dedup-hashmap`, copy the `got: sha256-...` line from
# the failure into this field, and commit.

buildPythonPackage {
  pname = "dedup-hashmap";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ./..;

  cargoDeps = rustPlatform.fetchCargoVendor {
    src = lib.cleanSource ./..;
    hash = "sha256-9AOzi/OUiySETZ8xIJ5Ljj7WSflLpjg3UxpsYHLBPdc=";
  };

  nativeBuildInputs = [
    rustPlatform.cargoSetupHook
    rustPlatform.maturinBuildHook
    pkg-config
  ];

  doCheck = false;

  meta = with lib; {
    description = "u64 -> u32 hashmap exposed to Python for asm-tokenizer content-addressed dedup.";
    license = licenses.asl20;
    platforms = platforms.unix;
  };
}
