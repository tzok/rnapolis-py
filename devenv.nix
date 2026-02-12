{ pkgs, ... }:
{
  languages.python = {
    enable = true;
    poetry.enable = true;
    poetry.install.groups = [ "docs" ];
  };
  packages = [
    pkgs.graphviz
    pkgs.highs
    pkgs.zlib
    pkgs.jupyter
  ];
  enterShell = ''
    export PYTHONPATH=src/
  '';
}
