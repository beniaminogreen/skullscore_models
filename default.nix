{ pkgs ? import <nixpkgs> {} }: with pkgs; mkShell { buildInputs = [
    stdenv.cc.cc.lib
    python311Packages.numpy
    libGL
    glib.out
    xorg.libxcb.dev
];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.glib.out}/lib
  '';
}
