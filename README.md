# Woolydart

A Dart wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides enough functionality to be versatile and useful. The basic, higher-level C functions that this
library builds upon are are provided by the [woolycore](https://github.com/tbogdala/woolycore) library.

Supported Operating Systems: Windows, MacOS, Linux, iOS, Android 

Note: Android support is non-GPU accelerated.


## License

MIT licensed, like the core upstream `llama.cpp` it wraps. See `LICENSE` for details.


## Features

* Simple high-level Dart class to use for text generation (`LlamaModel`).
* Basic samplers of llama.cpp, including: temp, top-k, top-p, min-p, tail free sampling, locally typical sampling, mirostat.
* Support for llama.cpp's BNF-like grammar rules for sampling.
* Ability to cache the processed prompt data in memory so that it can be reused to speed up regeneration using the exact same prompt.
  Additionally, the processed prompt and predicted tokens can cache the model state after prediction as well so that it may
  be resumed quickly.


## Build notes

To use these bindings, the upstream `llama.cpp` support needs to be compiled. This is provided through the 
[woolycore](https://github.com/tbogdala/woolycore) library. Further information can be found in that project's
README file, but the basic build can be executed with the following commands:

```bash
cd src
cmake -B build -DWOOLY_TESTS=Off woolycore
cmake --build build --config Release
```

This will generate the library files required so that the Dart wrapper can load them. Additionally,
this sets an additional CMake flag to stop woolycore from building its unit tests. 

This will automatically use metal on MacOS, but for CUDA platforms you'll need to enable 
it with a separate flag:

```bash
cd src
cmake -B build -DWOOLY_TESTS=Off -DGGML_CUDA=On woolycore
cmake --build build --config Release
```

Windows users have an extra level of pain to deal with and need additional steps to make all this work
(in this example, CUDA is enabled, but `-DGGML_CUDA=On` can be removed for a CPU only build):

```bash
cd src
cmake -B build -DWOOLY_TESTS=Off -DGGML_CUDA=On -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE woolycore
cmake --build build --config Release
cd ..

# basically we now need to make the compiled dll files findable while running the tests.
# the simple solution is to just copy them to the project root for test running.

cp src/build/bin/Release/ggml.dll .
cp src/build/bin/Release/llama.dll .
cp src/build//Release/woolycore.dll .
```



## Git updates

This project uses submodules for upstream projects so make sure to update with appropriate parameters:

```bash
git pull --recurse-submodules
```


## Tests

Once `wollycore` has been built, the Dart wrappers should function. The unit tests require an environment variable
(`WOOLY_TEST_MODEL_FILE`) to be set with the path to the GGUF file for the model to use during testing.

You can run the tests by using the following command:

```bash
export WOOLY_TEST_MODEL_FILE=models/example-llama-3-8b.gguf
dart test --concurrency=1
```

It's important to limit concurrency or else the tests will run in parallel and cause performance
and memory limit issues.


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
(reduced with `libreadability` from 195,963 characters to 64,413 characters) into the context. 
You can specify the context size with the `-c` parameter.

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

The same example with the same context size and URL, but using Meta-Llama-3.1-8B-Instruct-Q8_0.gguf, gives the 
following performance on a Windows 11 machine with a 4090 and an AMD R9 7950X: 

`Performance data: 249 tokens (1171 characters) total in 12499.50 ms (19.92 T/s) ; 19967 prompt tokens in 8117.99 ms (2459.60 T/s)`

For a fun time on MacOS, you can configure `Accessibility > Spoken Content` in the `System Settings` app to have the 
system voice you want, and then re-run the above example but quiet `llama.cpp`'s output and then pipe the generated
text to `say`.

```bash
dart example/rag_summarize.dart -m ~/.cache/lm-studio/models/bartowski/Phi-3.1-mini-128k-instruct-GGUF/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf --url "https://en.wikipedia.org/wiki/William_Perry_(American_football)" -c 28000 -q | say
```


### Developer Notes

* MacOS needed `brew install llvm` to run this, I believe.

* FFIGEN invoked as `dart run ffigen`, but that shouldn't need to be done by consumers of the library unless you're
  updating the `woolycore` bindings yourself.

* Callbacks are probably not re-entrant and have not been tested for that use case.
