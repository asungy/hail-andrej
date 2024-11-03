{ pkgs, lib, config, inputs, ... }:

{
  packages = (with pkgs.python312Packages; [
    graphviz
    jupyterlab
    matplotlib
    numpy
    torch
  ]) ++ (with pkgs; [
    bashInteractive
  ]);
  languages.python.enable = true;

  enterShell = ''
    export SHELL=/run/current-system/sw/bin/bash
  '';
}
