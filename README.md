# Prerequisites
```Bash
# gcc
gcc --version
gcc (GCC) 12.5.0

# sync triton code
./scripts/sync.sh

# llvm-project
./scripts/llvm.sh

# lldb-dap
cd /usr/bin/
sudo ln -sf ../lib/llvm-17/bin/lldb-vscode ./lldb-dap
```

# Build
```Bash
./build.sh
```
