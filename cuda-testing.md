# CUDA Testing Notes

## WSL GPU Visibility

In WSL, CUDA device access may require the WSL driver libraries to be first on
`LD_LIBRARY_PATH`. The normal Linux checks may be misleading:

- `/dev/nvidia*` may not exist.
- `/dev/dxg` is the WSL GPU device path.
- `nvidia-smi` may live at `/usr/lib/wsl/lib/nvidia-smi`.

Useful checks:

```bash
ls -l /dev/dxg
/usr/lib/wsl/lib/nvidia-smi
```

## Sandbox Access

When tests are run from a sandboxed agent/session, WSL GPU access may be hidden
even though the host has a CUDA-capable GPU. In that case, symptoms can include:

- `/dev/dxg` missing inside the sandbox.
- `/usr/lib/wsl/lib/nvidia-smi` reporting GPU access blocked.
- CUDA tests skipping with `no CUDA-capable device is detected`.
- CUDA tests skipping with `CUDA driver version is insufficient for CUDA runtime version`.

Run the CUDA checks and test binary with sandbox bypass / escalated execution so
the process can access `/dev/dxg`. In this environment, the successful run used
sandbox bypass plus the WSL CUDA library path:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./build-cuda/test/cuda_device_tests
```

## Build CUDA Tests

Use a CUDA-enabled build directory:

```bash
cmake -S . -B build-cuda -DSANDY_ENABLE_CUDA=ON
cmake --build build-cuda --target cuda_device_tests -j2
```

## Run CUDA Tests In WSL

Run the CUDA test binary with the WSL CUDA libraries on the library path:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./build-cuda/test/cuda_device_tests
```

Expected successful output ends with:

```text
[==========] 11 tests from 1 test suite ran.
[  PASSED  ] 11 tests.
```

If the binary reports `CUDA driver version is insufficient for CUDA runtime version`
or `no CUDA-capable device is detected`, first confirm `/dev/dxg` is visible and
that `/usr/lib/wsl/lib` is present in `LD_LIBRARY_PATH` for the test process.
