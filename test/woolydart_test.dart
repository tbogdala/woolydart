import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:format/format.dart';
import 'package:test/test.dart';
import 'dart:developer';

import 'package:woolydart/woolydart.dart';

void main() {
  group('Raw binding tests', () {
    const dylibPath = "src/build/libwoolydart.dylib";
    final lib = woolydart(DynamicLibrary.open(dylibPath));

    final modelFilepath = Platform.environment['WOOLY_TEST_MODEL_FILE'];
    if (modelFilepath == null) {
      print(
          'Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing');
      return;
    }
    var modelPath = modelFilepath.toNativeUtf8();

    var modelParams = lib.llama_model_default_params();
    modelParams.n_gpu_layers = 100;

    var contextParams = lib.llama_context_default_params();
    contextParams.seed = 42;
    contextParams.n_ctx = 0;

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

    final params = lib.wooly_new_params();
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
    params.ignore_eos = false;
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

    final contextSize = lib.llama_n_ctx(loadedModel.ctx);

    // allocate the buffer for the predicted text. by default we just use the worst
    // case scenario of a whole context size with four bytes per utf-8.
    final outputText = calloc.allocate(contextSize * 4) as Pointer<Char>;

    // setup the callback test by zeroing the count and creating the cb pointer
    globalCallbackCount = 0;
    token_update_callback tokenUpdate =
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

  /****************************************************************************/
  // Managed classes tests

  group('Fancy bindings tests', () {
    const libFilepath = "src/build/libwoolydart.dylib";
    var llamaModel = LlamaModel(libFilepath);

    final modelFilepath = Platform.environment['WOOLY_TEST_MODEL_FILE'];
    if (modelFilepath == null) {
      print(
          'Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing');
      return;
    }

    final modelParams = llamaModel.getDefaultModelParams();
    modelParams.n_gpu_layers = 100;
    final contextParams = llamaModel.getDefaultContextParams();
    contextParams.seed = 42;
    contextParams.n_ctx = 0;

    final loadedResult =
        llamaModel.loadModel(modelFilepath, modelParams, contextParams, true);

    final checkIsLoaded = llamaModel.isModelLoaded();

    test('Model load test', () {
      expect(loadedResult, true);
      expect(checkIsLoaded, true);
      expect(llamaModel.model, isNotNull);
      expect(llamaModel.ctx, isNotNull);
    });

    if (loadedResult == false || checkIsLoaded == false) {
      throw Exception('Failed to load the test model! Test aborted.');
    }

    final params = llamaModel.getTextGenParams();
    params.seed = 42;
    params.n_threads = 4;
    params.n_predict = 100;
    params.temp = 0.1;
    params.top_k = 1;
    params.top_p = 1.0;
    params.min_p = 0.1;
    params.penalty_repeat = 1.1;
    params.penalty_last_n = 512;
    params.ignore_eos = false;
    params.flash_attn = true;
    params.n_batch = 128;
    params.prompt_cache_all = true;
    params.setPrompt(
        "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n");
    params.setAntiprompts([
      "<|end|>",
    ]);

    test('Parameter creation test', () {
      expect(params, isNotNull);
      expect(params.prompt, isNotNull);
      expect(params.antiprompts, isNotNull);
      expect(params.antiprompt_count, equals(1));
    });

    var (predictResult, outputString) = llamaModel.predictText(params, nullptr);

    test('Text Prediction', () {
      expect(predictResult.result, 0);
    });

    // Print out the generated text for fun as well as some stats on timing.
    print(format('Generated text:\n{}', outputString ?? "<failed prediction>"));
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

    // change the seed and generate again with the same prompt. this should trigger
    // the use of the prompt cache and drop n_p_eval and t_p_eval_ms down to 0.
    params.seed = 1337;

    // do another callback test too with this run
    globalCallbackCount = 0;
    token_update_callback tokenUpdate =
        Pointer.fromFunction(testCallback, false);

    var (predictResult2, outputString2) =
        llamaModel.predictText(params, tokenUpdate);

    test('Text Prediction', () {
      expect(predictResult2.result, 0);
      expect(predictResult2.n_p_eval, 0);
      expect(predictResult2.t_p_eval_ms, 0);
      expect(globalCallbackCount, 100);
    });

    // Print out the generated text for fun as well as some stats on timing.
    print(format('\nUsing the same prompt but different seed:\n{}',
        outputString2 ?? "<failed prediction>"));
    print(
      format(
          '\nTiming Data: {} tokens total in {:.2f} ms ({:.2f} T/s) ; {} prompt tokens in {:.2f} ms ({:.2f} T/s)\n',
          predictResult2.n_eval,
          (predictResult2.t_end_ms - predictResult2.t_start_ms),
          1e3 /
              (predictResult2.t_end_ms - predictResult2.t_start_ms) *
              predictResult2.n_eval,
          predictResult2.n_p_eval,
          predictResult2.t_p_eval_ms,
          1e3 / predictResult2.t_p_eval_ms * predictResult2.n_p_eval),
    );

    // free the allocated memory
    tearDownAll(() {
      params.dispose();
      llamaModel.freeModel();
    });
  });

  /****************************************************************************/
  // Managed class grammar test

  group('Fancy bindings gramar test', () {
    const libFilepath = "src/build/libwoolydart.dylib";
    var llamaModel = LlamaModel(libFilepath);

    final modelFilepath = Platform.environment['WOOLY_TEST_MODEL_FILE'];
    if (modelFilepath == null) {
      print(
          'Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing');
      return;
    }

    final modelParams = llamaModel.getDefaultModelParams();
    modelParams.n_gpu_layers = 100;
    final contextParams = llamaModel.getDefaultContextParams();
    contextParams.seed = -1;
    contextParams.n_ctx = 0;

    final loadedResult =
        llamaModel.loadModel(modelFilepath, modelParams, contextParams, true);

    test('Model load test', () {
      expect(loadedResult, true);
      expect(llamaModel.model, isNotNull);
      expect(llamaModel.ctx, isNotNull);
    });

    if (loadedResult == false) {
      throw Exception('Failed to load the test model! Test aborted.');
    }

    final params = llamaModel.getTextGenParams();
    params.n_predict = -1;
    params.temp = 1.4;
    params.top_k = 40;
    params.top_p = 1.0;
    params.min_p = 0.05;
    params.penalty_repeat = 1.1;
    params.penalty_last_n = 512;
    params.flash_attn = true;
    params.n_batch = 128;
    params.setPrompt(
        "<|user|>\nReturn a JSON object that describes an object in a fictional Dark Souls game. The returned JSON object should have 'Title' and 'Description' fields that define the item in the game. Make sure to write the item lore in the style of Fromsoft and thier Dark Souls series of games: there should be over-the-top naming of fantastically gross monsters and tragic historical events from the world, all with a very nihilistic feel.<|end|>\n<|assistant|>\n");
    params.setAntiprompts([
      "<|end|>",
    ]);

    // now we load the grammar from the llama.cpp project
    File grammarFile = File('src/llama.cpp/grammars/json.gbnf');
    String grammarRules = grammarFile.readAsStringSync();
    params.setGrammar(grammarRules);

    test('Parameter creation test', () {
      expect(params, isNotNull);
      expect(params.prompt, isNotNull);
      expect(params.antiprompts, isNotNull);
      expect(params.antiprompt_count, equals(1));
    });

    var (predictResult, outputString) = llamaModel.predictText(params, nullptr);

    test('Text Prediction', () {
      expect(predictResult.result, 0);
    });

    // Print out the generated text for fun as well as some stats on timing.
    print(format('Generated text:\n{}', outputString ?? "<failed prediction>"));
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

    // free the allocated memory
    tearDownAll(() {
      params.dispose();
      llamaModel.freeModel();
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
