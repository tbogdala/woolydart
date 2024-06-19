# Woolydart

A Dart wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides enough functionality to be versatile, but also exposes the raw llama.cpp C callable functions
for further lower level access if desired.

At present, it is in pre-alpha development and highly unstable.


## Build notes

* `llama.cpp` is now a submodule in `./src` and will need to be pulled and updated with `--recurse-submodules`.

* Before building the upstream project, run the patch to include the custom bindings:

```bash
cd src/llama.cpp
git apply ../llamacpp_patch.patch
```

* Built MacOs with the following commands:

```bash
cd src/llama.cpp
cmake -B build -DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=On -DLLAMA_BUILD_TESTS=Off -DLLAMA_BUILD_EXAMPLES=Off -DLLAMA_METAL_EMBED_LIBRARY=On
cd build
make build_info
cmake --build . --config Release
```

* Once the upstream `llama.cpp` libraries have been built, the Dart wrappers should function. You can run the
  tests by using the following command:

  ```bash
  dart test
  ```


### Developer Notes

* MacOS needed `brew install llvm` to run this, I believe.

* FFIGEN invoked as `dart run ffigen`, but that shouldn't need to be done by consumers of the library unless you're
  updating the `llama.cpp` bindings yourself.

