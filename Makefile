default:
	./build.sh

clean:
	rm -fv ./.build_llvm.done && rm -rf llvm-project/build && rm -rf build/
