{ pkgs, ... }:
{
  languages.python = {
    enable = true;
    venv = {
      enable = true;
      requirements = ./requirements.txt;
    };
  };
  packages = [
    pkgs.highs
    pkgs.zlib
  ];
  enterShell = ''
    export PYTHONPATH=src/
  '';
}
