# Woolydart

A Dart wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides enough functionality to be versatile, but also exposes the raw llama.cpp C callable functions
for further lower level access if desired.

At present, it is in pre-alpha development and the API is unstable. 

Upstream llama.cpp is pinned to commit [ABD894A](https://github.com/ggerganov/llama.cpp/commit/abd894ad96a242043b8e197ec130d8649eead22e)


## Features

* Simple high-level Dart class to use for text generation (`LlamaModel`); low-level llama.cpp functions are exposed for those that need more.
* Basic samplers of llama.cpp, including: temp, top-k, top-p, min-p, tail free sampling, locally typical sampling, mirostat.
* Support for llama.cpp's BNF-like grammar rules fro sampling.
* Ability to cache the processed prompt data in memory so that it can be reused to speed up regeneration using the exact same prompt.


## Build notes

`llama.cpp` is now a submodule in `./src` and will need to be pulled and updated with `--recurse-submodules`.

Before building the upstream project, run the patch to include the custom bindings:

```bash
cd src/llama.cpp
git apply ../llamacpp_patch.patch
```

Built MacOs with the following commands:

```bash
cd src/llama.cpp
cmake -B build -DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=On -DLLAMA_BUILD_TESTS=Off -DLLAMA_BUILD_EXAMPLES=Off -DLLAMA_METAL_EMBED_LIBRARY=On
cd build
make build_info
cmake --build . --config Release
```

Once the upstream `llama.cpp` libraries have been built, the Dart wrappers should function. You can run the
tests by using the following command:

```bash
dart test
```


## Examples

The basic example can be run with the following command (use `--help` to see all command-line arguments):

```bash
dart examples/basic_example.dart -m <GGUF file path>
```

Make sure to actually specify a GGUF file path so it can load the model for testing.


### Developer Notes

* MacOS needed `brew install llvm` to run this, I believe.

* FFIGEN invoked as `dart run ffigen`, but that shouldn't need to be done by consumers of the library unless you're
  updating the `llama.cpp` bindings yourself.

* Callbacks are probably not re-entrant and have not been tested for that use case.


### TODO

* Reenable some advanced sampling features again like logit biases.
* Missing calls to just tokenize text and to pull embeddings out from text.
* Maybe a dynamic LoRA layer, trained every time enough tokens fill up the context space, approx.