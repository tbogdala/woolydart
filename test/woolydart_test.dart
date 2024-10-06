import 'dart:ffi';
import 'dart:io';
import 'package:format/format.dart';
import 'package:test/test.dart';

import 'package:woolydart/woolydart.dart';

String getPlatformLibraryFilepath() {
  if (Platform.isMacOS) {
    return "src/build/libwoolycore.dylib";
  } else if (Platform.isWindows) {
    return "woolycore.dll";
  } else {
    return "src/build/libwoolycore.so";
  }
}

void main() {
  /****************************************************************************/
  // Managed classes tests

  group('Fancy bindings tests', () {
    final libFilepath = getPlatformLibraryFilepath();
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
    contextParams.n_ctx = 2048;

    final loadedResult =
        llamaModel.loadModel(modelFilepath, modelParams, contextParams, true);

    final checkIsLoaded = llamaModel.isModelLoaded();

    test('Model load test', () {
      expect(loadedResult, true);
      expect(checkIsLoaded, true);
    });

    if (loadedResult == false || checkIsLoaded == false) {
      throw Exception('Failed to load the test model! Test aborted.');
    }

    // check the length of a string in tokens; the second bool is set to true
    // so that 'special tokens' get parsed and if using a llama3 tokenizer, for
    // example, the following test string gets tokenized differently.
    final tokenLenTestString =
        "<|start_header_id|>assistant<|end_header_id|>\n\nI've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion.";
    final tokenCount =
        llamaModel.getTokenCount(tokenLenTestString, false, true);
    print(
        'We got a test token length of $tokenCount for "$tokenLenTestString"\n');

    // we'll set the test to have a pretty wide margin of acceptable values here
    // because the user may have any GGUF select to run the test with ... we just
    // want to make sure we're getting approximately a reasonable number.
    test('Model token count test', () {
      expect(tokenCount > 10, true);
      expect(tokenCount < 100, true);
    });

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
    wooly_token_update_callback tokenUpdate =
        Pointer.fromFunction(testCallback, false);

    var (predictResult2, outputString2) =
        llamaModel.predictText(params, tokenUpdate);

    test('Text Prediction', () {
      expect(predictResult2.result, 0);
      expect(predictResult2.n_p_eval, 1);
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
}

// maybe a terrible test, but it's at least something
int globalCallbackCount = 0;

bool testCallback(Pointer<Char> tokenString) {
  //var dartToken = (tokenString as Pointer<Utf8>).toDartString();
  //stdout.write(dartToken);
  globalCallbackCount += 1;
  return true;
}
