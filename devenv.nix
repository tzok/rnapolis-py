{ pkgs, ... }:
{
  languages.python = {
    enable = true;
    poetry.enable = true;
    poetry.install.groups = [ "docs" ];
    jupyter.enable = true;
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
