{ pkgs, ... }:
let
  infernal = pkgs.stdenv.mkDerivation (finalAttrs: {
    pname = "infernal";
    version = "1.1.5";

    src = pkgs.fetchurl {
      url = "http://eddylab.org/software/infernal/infernal-${finalAttrs.version}.tar.gz";
      hash = "sha256-rU3a4C+STKfIW8jEp5yfh1r435autyZwL6mFy+dSSX8=";
    };

    buildInputs = [ pkgs.zlib ];

    doCheck = false;

    meta = {
      description = "Search and align RNA sequences to covariance models";
      homepage = "http://eddylab.org/infernal";
      license = pkgs.lib.licenses.bsd3;
      mainProgram = "cmsearch";
      platforms = pkgs.lib.platforms.unix;
    };
  });
in
{
  languages.python = {
    enable = true;
    venv.enable = true;
    uv.enable = true;
    uv.sync = {
      enable = true;
      arguments = [ "--locked" ];
      groups = [ "docs" ];
    };
  };
  packages = [
    pkgs.graphviz
    pkgs.highs
    pkgs.zlib
    pkgs.jupyter
    infernal
  ];
}
