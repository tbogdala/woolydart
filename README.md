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
cmake -B build -DGGML_METAL=On -DGGML_METAL_EMBED_LIBRARY=On 
```

Once the custom library with `llama.cpp` code and the custom bindings code has been built, the Dart wrappers should function. You can run the
tests by using the following command:

```bash
dart test
```


## Examples

### Basic Text Generation

The basic example can be run with the following command (use `--help` to see all command-line arguments):

```bash
dart examples/basic_example.dart -m <GGUF file path>
```

Make sure to actually specify a GGUF file path so it can load the model for testing.

### Simple RAG Summarization

This is a simple sample that will load the content from the provided URL on the command line, run it
through an optional *readability* pass, and then pass it to the LLM for summarization.

Note: There's a git submodule in `./src` called `libreadability` which wraps a Rust library called
[readability.rs](https://github.com/readable-app/readability.rs). This library cleans out some 
unneeded portions of the retrieved HTML so that we can pass less text to the LLM. To build this,
you will need a Rust toolchain installed and then issue the following command from this project's root
folder: `cd src/libreadability;cargo build --release` ... this sample will still run if you don't build
it, but you'll get a warning message and more text will get passed to the LLM.

Currently there is no chunking strategy to this example so you will have to fit the whole HTML document
(reduced with `libreadability`) into the context. You can specify the context size with the `-c` parameter.
 On my MacBook Air M3 with 24 GB of memory, I can run the following sample command:

```bash
dart example/rag_summarize.dart -m ~/.cache/lm-studio/models/bartowski/Phi-3.1-mini-128k-instruct-GGUF/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf --url "https://en.wikipedia.org/wiki/William_Perry_(American_football)" -c 28000
```

This generates the following output:

```
 William Anthony "The Refrigerator" Perry, born December 16, 1962 in Aiken, South Carolina, 
 was a renowned NFL defensive tackle who played for the Chicago Bears and Clemson Tigers. 
 He earned the nickname "Refrigerator" due to his massive size during college football at 
 Clemson University, where he also received the ACC Player of the Year award. Drafted in 
 1985 by the Bears, Perry gained fame as part of their first Super Bowl-winning team, 
 setting a record for the heaviest player to score a touchdown in the Super Bowl and 
 possessing the largest Super Bowl ring. Despite facing challenges with his weight 
 throughout his professional career, he played 10 seasons and became an iconic figure among
 Bears fans. Perry's off-field ventures included music collaborations with Walter Payton 
 and media appearances, including a notable boxing match against Bob Sapp. After retiring 
 from football in 1994, he faced personal struggles, losing over one hundred pounds before 
 regaining much of his weight. His health issues continued to affect him later on, leading 
 to hospitalization for diabetes treatment and hearing loss. Perry's life post-football has 
 been marked by various ups and downs, including a brief comeback attempt in the World 
 League of American Football and an induction into the WWE Hall of Fame. He also faced 
 legal disputes over his Super Bowl ring, which was later auctioned for $200,000. Perry's 
 legacy is remembered not just for his athletic achievements but also for his 
 larger-than-life personality off the field.

Performance data: 358 tokens (1529 characters) total in 371565.63 ms (0.96 T/s) ; 
26962 prompt tokens in 289057.38 ms (93.28 T/s)
```

For a fun time on MacOS, you can configure `Accessibility > Spoken Content` in the `System Settings` app to have the 
system voice you want, and then re-run the above example but quiet `llama.cpp`'s output and then pipe the generated
text to `say`.

```bash
dart example/rag_summarize.dart -m ~/.cache/lm-studio/models/bartowski/Phi-3.1-mini-128k-instruct-GGUF/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf --url "https://en.wikipedia.org/wiki/William_Perry_(American_football)" -c 28000 -q | say
```


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