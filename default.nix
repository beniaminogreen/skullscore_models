{ pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = true;
  };
} }:
  pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      cudaPackages_12.cudatoolkit
      stdenv.cc.cc.lib
      python311Packages.numpy
      glib.out
      libGL
    ];

    shellHook = ''
      echo "You are now using a NIX environment"
      export CUDA_PATH=${pkgs.cudatoolkit}
      export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}:${pkgs.glib.out}/lib:${pkgs.libGL}/lib/
    '';
}
