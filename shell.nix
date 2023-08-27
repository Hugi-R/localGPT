{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.python311
    pkgs.poetry
    pkgs.python311Packages.torch
    pkgs.python311Packages.numpy
  ];

  CMAKE_ARGS = "-DLLAMA_CUBLAS=on";
}