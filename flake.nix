{
  description = "Rust Rover environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          # config.cudaSupport = true;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override {
          extensions = [ "rust-src" "clippy" "rustfmt" "rust-analyzer" ];
        });
      in {
        devShell = pkgs.mkShell rec {
          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            rustToolchain
            jetbrains.rust-rover
            linuxPackages_latest.perf
            vulkan-tools
          ];

          buildInputs = with pkgs; [
            SDL2
            SDL2_gfx
            vulkan-loader
            libGL
            libxkbcommon
            wayland

            # cudaPackages.cudatoolkit
            # cudaPackages.cuda_cudart
            # cudaPackages.cuda_cupti
            # cudaPackages.cuda_nvrtc
            # cudaPackages.cuda_nvtx
            # cudaPackages.cudnn
            # cudaPackages.libcublas
            # cudaPackages.libcufft
            # cudaPackages.libcurand
            # cudaPackages.libcusolver
            # cudaPackages.libcusparse
            # cudaPackages.libnvjitlink
            # cudaPackages.nccl
            # cudaPackages.nsight_systems
          ];

          shellHook = ''
            mkdir -p ~/.rust-rover/toolchain

            ln -sfn ${rustToolchain}/lib ~/.rust-rover/toolchain
            ln -sfn ${rustToolchain}/bin ~/.rust-rover/toolchain

            export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export RUST_SRC_PATH="$HOME/.rust-rover/toolchain/lib/rustlib/src/rust/library"
            export RUST_BACKTRACE=full
            export EZ_LOG=trace
            zsh
          '';
        };
      }
    );
}
