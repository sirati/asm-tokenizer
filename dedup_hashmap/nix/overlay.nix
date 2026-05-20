final: prev: {
  # Inject `dedup-hashmap` into every Python package set on the consuming
  # nixpkgs instance. Consumers can then reference it as
  # `final.python3Packages.dedup-hashmap` (or any other pythonXYPackages
  # set, e.g. `final.python314.pkgs.dedup-hashmap`), exactly like an
  # upstream nixpkgs Python package.
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (pyFinal: pyPrev: {
      dedup-hashmap = pyFinal.callPackage ./wheel.nix { };
    })
  ];
}
