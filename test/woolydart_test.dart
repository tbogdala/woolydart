import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:format/format.dart';
import 'package:woolydart/src/llama_cpp_bindings.dart';
import 'package:test/test.dart';

void main() {
  group('Raw binding tests', () {
    const dylibPath = "src/llama.cpp/build/libllama.dylib";
    final lib = woolydart(DynamicLibrary.open(dylibPath));

    var modelPath =
        "/Users/timothy/.cache/lm-studio/models/SanctumAI/Phi-3-mini-4k-instruct-GGUF/phi-3-mini-4k-instruct.Q8_0.gguf"
            .toNativeUtf8();
    var emptyString = "".toNativeUtf8();

    var loadedModel = lib.wooly_load_model(
        modelPath as Pointer<Char>,
        2048, // ctx
        42, // seed
        false, // mlock
        true, // mmap
        false, // embeddings
        100, // gpu layers
        256, // batch
        0, // maingpu
        emptyString as Pointer<Char>, //tensorsplit
        0.0, // rope freq
        0.0); // rope scale

    test('Model load test', () {
      expect(loadedModel.model, isNotNull);
      expect(loadedModel.ctx, isNotNull);
    });

    var prompt =
        "Write the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.\n"
            .toNativeUtf8();
    var seed = 42;
    var threads = 4;
    var tokens = 100;
    var topK = 40;
    var topP = 1.0;
    var minP = 0.08;
    var temp = 1.1;
    var repeatPenalty = 1.1;
    var repeatLastN = 512;
    var ignoreEos = false;
    var nBatch = 128;
    var nKeep = 128;
    var antiprompt = "".toNativeUtf8();
    var antipromptCount = 0;
    var tfsZ = 1.0;
    var typicalP = 1.0;
    var frequencyPenalty = 0.0;
    var presencePenalty = 0.0;
    var mirostat = 0;
    var mirostatEta = 0.1;
    var mirostatTau = 5.0;
    var penalizeNl = false;
    var logitBias = "".toNativeUtf8();
    var sessionFile = "".toNativeUtf8();
    var promptCacheInMemory = false;
    var mlock = false;
    var mmap = true;
    var maingpu = 0;
    var tensorsplit = "".toNativeUtf8();
    var filePromptCacheRo = false;
    var ropeFreqBase = 0.0;
    var ropeFreqScale = 0.0;
    var grammar = "".toNativeUtf8();

    var params = lib.wooly_allocate_params(
        prompt as Pointer<Char>,
        seed,
        threads,
        tokens,
        topK,
        topP,
        minP,
        temp,
        repeatPenalty,
        repeatLastN,
        ignoreEos,
        nBatch,
        nKeep,
        antiprompt as Pointer<Pointer<Char>>,
        antipromptCount,
        tfsZ,
        typicalP,
        frequencyPenalty,
        presencePenalty,
        mirostat,
        mirostatEta,
        mirostatTau,
        penalizeNl,
        logitBias as Pointer<Char>,
        sessionFile as Pointer<Char>,
        promptCacheInMemory,
        mlock,
        mmap,
        maingpu,
        tensorsplit as Pointer<Char>,
        filePromptCacheRo,
        ropeFreqBase,
        ropeFreqScale,
        grammar as Pointer<Char>);

    test('Parameter creation test', () {
      expect(params, isNotNull);
    });

    // allocate the buffer for the predicted text.
    final outputText = calloc.allocate((tokens + 1) * 4) as Pointer<Char>;

    var predictResult = lib.wooly_predict(
        params, loadedModel.ctx, loadedModel.model, false, outputText, nullptr);

    test('Text Prediction', () {
      expect(predictResult.result, 0);
    });

    // convert the predicted text back to a Dart string.
    final outputString = (outputText as Pointer<Utf8>).toDartString();
    print(format('Generated text:\n{}', outputString));

    print(format(
        '\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n',
        predictResult.n_eval,
        (predictResult.t_end_ms - predictResult.t_start_ms),
        1e3 /
            (predictResult.t_end_ms - predictResult.t_start_ms) *
            predictResult.n_eval));

    // free the allocated text buffer
    calloc.free(outputText);

    lib.wooly_free_model(loadedModel.ctx, loadedModel.model);
  });
}
