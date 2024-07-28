import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:format/format.dart';
import 'package:test/test.dart';
import 'dart:developer';

import 'package:woolydart/woolydart.dart';

String getPlatformLibraryFilepath() {
  return (Platform.isMacOS) ? "src/build/libwoolycore.dylib" : "src/build/libwoolycore.so";
}

void main() {
  group('Raw binding tests', () {
    final dylibPath = getPlatformLibraryFilepath();
    final lib = woolydart(DynamicLibrary.open(dylibPath));

    final modelFilepath = Platform.environment['WOOLY_TEST_MODEL_FILE'];
    if (modelFilepath == null) {
      print(
          'Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing');
      return;
    }
    var modelPath = modelFilepath.toNativeUtf8();

    var modelParams = lib.wooly_get_default_llama_model_params();
    modelParams.n_gpu_layers = 100;

    var contextParams = lib.wooly_get_default_llama_context_params();
    contextParams.seed = 42;
    contextParams.n_ctx = 2048;

    var loadedModel = lib.wooly_load_model(
        modelPath as Pointer<Char>, modelParams, contextParams, true);

    test('Model load test', () {
      expect(loadedModel.model, isNotNull);
      expect(loadedModel.ctx, isNotNull);
    });

    if (loadedModel.model == nullptr || loadedModel.ctx == nullptr) {
      throw Exception(
          'Failed to load the model file for testing! Test aborted.');
    }

    final params = lib.wooly_new_gpt_params();
    params.prompt =
        "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n"
            .toNativeUtf8() as Pointer<Char>;
    params.seed = 42;
    params.n_threads = 4;
    params.n_predict = 100;
    params.temp = 0.1;
    params.top_k = 1;
    params.top_p = 1.0;
    params.min_p = 0.1;
    params.penalty_repeat = 1.1;
    params.penalty_last_n = 512;
    params.ignore_eos = true;
    params.flash_attn = true;
    params.n_batch = 128;
    params.prompt_cache_all = false;

    final antipromptStrings = [
      "<|end|>",
    ];

    // keep track of the native strings so we can deallocate them.
    List<Pointer<Char>> antipromptPointers = [];

    // okay now actually create the antiprompt native strings if we have them.
    if (antipromptStrings.isNotEmpty) {
      log("Making antiprompt strings native...");

      // allocate all the array of pointers.
      final Pointer<Pointer<Char>> antiPointers =
          calloc.allocate(antipromptStrings.length * sizeOf<Pointer<Char>>());

      // allocate each of the native strings
      for (int ai = 0; ai < antipromptStrings.length; ai++) {
        log("Allocating antipromtp #$ai");
        Pointer<Char> native =
            antipromptStrings[ai].toNativeUtf8() as Pointer<Char>;
        antiPointers[ai] = native;

        antipromptPointers.add(native);
      }

      params.antiprompts = antiPointers;
      params.antiprompt_count = antipromptPointers.length;
    }

    test('Parameter creation test', () {
      expect(params, isNotNull);
    });

    final contextSize = loadedModel.context_length;

    // allocate the buffer for the predicted text. by default we just use the worst
    // case scenario of a whole context size with four bytes per utf-8.
    final outputText = calloc.allocate(contextSize * 4) as Pointer<Char>;

    // setup the callback test by zeroing the count and creating the cb pointer
    globalCallbackCount = 0;
    wooly_token_update_callback tokenUpdate =
        Pointer.fromFunction(testCallback, false);

    var predictResult = lib.wooly_predict(params, loadedModel.ctx,
        loadedModel.model, false, outputText, nullptr, tokenUpdate);

    test('Text Prediction', () {
      expect(predictResult.result, 0);
      expect(globalCallbackCount, 100);
    });

    // convert the predicted text back to a Dart string.
    final outputString = (outputText as Pointer<Utf8>).toDartString();
    print(format('Generated text:\n{}', outputString));

    print(format(
        '\nTiming Data: {} tokens total in {:.2f} ms ({:.2f} T/s) ; {} prompt tokens in {:.2f} ms ({:.2f} T/s)\n\n',
        predictResult.n_eval,
        (predictResult.t_end_ms - predictResult.t_start_ms),
        1e3 /
            (predictResult.t_end_ms - predictResult.t_start_ms) *
            predictResult.n_eval,
        predictResult.n_p_eval,
        predictResult.t_p_eval_ms,
        1e3 / predictResult.t_p_eval_ms * predictResult.n_p_eval));

    tearDownAll(() {
      // free the allocated text buffer
      calloc.free(outputText);
      malloc.free(modelPath);
      malloc.free(params.prompt);

      if (antipromptPointers.isNotEmpty) {
        for (int ai = 0; ai < antipromptPointers.length; ai++) {
          malloc.free(antipromptPointers[ai]);
        }
        malloc.free(params.antiprompts);
      }

      lib.wooly_free_model(loadedModel.ctx, loadedModel.model);
    });
  });
}

// maybe a terrible test, but it's at least something
int globalCallbackCount = 0;

bool testCallback(Pointer<Char> tokenString) {
  //var dartToken = (tokenString as Pointer<Utf8>).toDartString();
  //stdout.write(dartToken);
  globalCallbackCount += 1;
  return true;
}