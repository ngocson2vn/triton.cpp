# triton.cpp
This project is for studying trion compiler C++ code https://github.com/triton-lang/triton/tree/984b694dc2916ee4f8cd18d3a28d1d8da14e076d/lib
<br/>

Main programs:<br/>
**1. triton compiler**: [bin/triton_compiler.cpp](bin/triton_compiler.cpp)<br/>
This is for compiling a triton ir file to a cubin file.<br/>
Pipeline: triton ir -> triton gpu ir -> llvm dialect -> llvm ir -> ptx -> cubin<br/>

**2. runner**: [bin/run_add_kernel.cpp](bin/run_add_kernel.cpp)<br/>
This is for running the cubin file generated from the triton ir file [add_kernel.ttir](./add_kernel.ttir).

## Sync
**1. Sync https://github.com/triton-lang/triton**
  ```Bash
  ./scripts/sync.sh
  ```
**2. Update `bin/RegisterTritonDialects.h`**
  ```C++
  // - Comment out all AMD headers
  // - Comment out all proton headers
  // - Comment out `namespace test`
  // - Comment out all AMD, proton, and test calls
  ```

**3. Update llvm-project**
```Bash
cd llvm-project/
COMMIT_HASH=$(cat ./cmake/llvm-hash.txt)
git fetch origin ${COMMIT_HASH}
```

## Build
### Prerequisites
```Bash
# clang 17
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo apt install -y lsb-release wget software-properties-common gnupg
sudo ./llvm.sh 17
cd /usr/bin/
sudo ln -sf ../lib/llvm-17/bin/clang .
sudo ln -sf ../lib/llvm-17/bin/clang++ .
sudo ln -sf ../lib/llvm-17/bin/ld.lld .
sudo ln -sf ../lib/llvm-17/bin/llvm-dwarfdump .
sudo ln -sf ../lib/llvm-17/bin/lldb .
sudo ln -sf ../lib/llvm-17/bin/lldb-vscode ./lldb-dap

# CUDA
/usr/local/cuda-12.4
Minimum Driver Version: 535.183.06

# Update llvm-project according to `cmake/llvm-hash.txt`
git submodule update --init --recursive
COMMIT_HASH=$(cat ./cmake/llvm-hash.txt)
echo "LLVM COMMIT_HASH = ${COMMIT_HASH}"
cd llvm-project/
git fetch origin ${COMMIT_HASH}
```

### Run build script
```
export CUDA_VERSION=xxx
./build.sh
```

## Run
Compile a sample triton ir file [add_kernel.ttir](./add_kernel.ttir):
```Bash
./compile.sh

# Output files
./output.ptx
./output.cubin
```
<br/>

Run the generated `./output.cubin`:
```Bash
./run.sh
```

## Debug
\- Using VSCode <br/>
\- Install LLDB DAP extension <br/>
\- Launch configs: [.vscode/launch.json](./.vscode/launch.json)
