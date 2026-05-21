{ pkgs, ... }:
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
  ];
}
