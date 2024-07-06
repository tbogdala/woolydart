# Woolydart

A Dart wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides enough functionality to be versatile, but also exposes the raw llama.cpp C callable functions
for further lower level access if desired.

At present, it is in pre-alpha development and the API is unstable. 

Upstream llama.cpp is pinned to commit [d7fd29f](https://github.com/ggerganov/llama.cpp/commit/d7fd29fff16456ce9c3a23fd2d09a66256b05aff)
from July 04, 2024.

Supported Operating Systems: MacOS, iOS, Android (more to come!)

## License

MIT licensed, like the core upstream `llama.cpp` it wraps. See `LICENSE` for details.


## Features

* Simple high-level Dart class to use for text generation (`LlamaModel`); low-level llama.cpp functions are exposed for those that need more.
* Basic samplers of llama.cpp, including: temp, top-k, top-p, min-p, tail free sampling, locally typical sampling, mirostat.
* Support for llama.cpp's BNF-like grammar rules for sampling.
* Ability to cache the processed prompt data in memory so that it can be reused to speed up regeneration using the exact same prompt.


## Build notes

To build the version of upstream llama.cpp that has woolydart's binding code in it, use the following commands.

```bash
cd src
cmake -B build
cmake --build build --config Release
```

For the Apple crowd, if you want Metal support with an embedded shader library for ease of distribution, you'll need to
add the appropriate flags:

```bash
cmake -B build -DLLAMA_METAL=On -DLLAMA_METAL_EMBED_LIBRARY=On
```

Once the custom library with `llama.cpp` code and the custom bindings code has been built, the Dart wrappers should function. You can run the
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

* `llama.cpp` is now a submodule in `./src` and will need to be pulled and updated with `--recurse-submodules` if upgrading the 
pinned version.


### TODO

* Reenable some advanced sampling features again like logit biases.
* Missing calls to just tokenize text and to pull embeddings out from text.
* Maybe a dynamic LoRA layer, trained every time enough tokens fill up the context space, approx.