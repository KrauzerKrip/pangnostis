{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.0.tar.gz"; # unstable Nixpkgs

  outputs =
    { self, ... }@inputs:

    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs { inherit system; };
          }
        );

      version = "3.13";
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        let
          concatMajorMinor =
            v:
            pkgs.lib.pipe v [
              pkgs.lib.versions.splitVersion
              (pkgs.lib.sublist 0 2)
              pkgs.lib.concatStrings
            ];
          python = pkgs."python${concatMajorMinor version}";
        in
        {
          default = pkgs.mkShell {
            venvDir = ".venv";
            
            # Ensure libstdc++ is available in LD_LIBRARY_PATH
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

            packages = [
              python
              pkgs.uv
              python.pkgs.numpy
              # We don't add python packages here directly as we manage them with uv
              # but we ensure the environment is ready for compiling/linking if needed
            ];
            
            shellHook = ''
              export UV_PYTHON="${python}/bin/python"
              echo "UV_PYTHON -> $UV_PYTHON"
              echo "LD_LIBRARY_PATH -> $LD_LIBRARY_PATH"
            '';
          };
        }
      );
    };
}