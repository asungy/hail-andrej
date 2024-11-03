{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs.python312Packages; [
    graphviz
    jupyterlab
    matplotlib
    numpy
    torch
  ];
  languages.python.enable = true;
}
