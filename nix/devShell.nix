{ mkShell
, jdk8
, python311
}:

let
  python-env = python311.withPackages (pp: with pp;
    [
      graphviz
      jupyter
      matplotlib
      numpy
      torch
    ]
  );
in
mkShell {
  buildInputs =
    []
    ++ [ python-env ];

  NIX_PYTHONPATH = "${python-env}/${python-env.sitePackages}";

  shellHook = ''
    if [[ ! -d .venv ]]; then
      echo "No virtual env found at ./.venv, creating a new virtual env linked to the Python site defined with Nix"
      ${python-env}/bin/python -m venv .venv
      cp ${builtins.toString ./sitecustomize.py} .venv/lib/python*/site-packages/
    fi
    source .venv/bin/activate
    echo "Nix development shell loaded."
  '';
}
