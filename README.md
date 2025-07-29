# Prerequisites
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

# lldb-dap
sudo ln -sf ../lib/llvm-17/bin/lldb-vscode ./lldb-dap
```

# Build
```Bash
./build.sh
```
