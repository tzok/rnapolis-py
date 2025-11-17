{ pkgs, ... }:
{
  languages.python = {
    enable = true;
    poetry.enable = true;
  };
  packages = [
    pkgs.graphviz
    pkgs.highs
    pkgs.zlib
  ];
  enterShell = ''
    export PYTHONPATH=src/
  '';
}
