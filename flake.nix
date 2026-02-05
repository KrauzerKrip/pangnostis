{
  description = "uv + Python on NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.uv
          pkgs.python313 # or python311/python313 as you like
        ];
        shellHook = ''
          export UV_PYTHON="$(which python3.13)"
          echo "UV_PYTHON -> $UV_PYTHON"
        '';
      };
    };
}

