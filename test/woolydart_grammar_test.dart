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
  // Managed class grammar test

  group('Fancy bindings grammar test', () {
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
    final isLoaded = llamaModel.isModelLoaded();

    test('Model load test', () {
      expect(loadedResult, true);
      expect(isLoaded, true);
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

    // use the built in template for the model to determine the prompt formatting
    // by building a list of `ChatMessage` objects.
    var messages = [
      ChatMessage("system",
          "You are a skilled creative writer working on video game assets."),
      ChatMessage("user",
          "Return a JSON object that describes an object in a fictional Dark Souls game. The returned JSON object should have 'Title' and 'Description' fields that define the item in the game. Make sure to write the item lore in the style of Fromsoft and thier Dark Souls series of games: there should be over-the-top naming of fantastically gross monsters and tragic historical events from the world, all with a very nihilistic feel.")
    ];
    final promptResult = llamaModel.makePromptFromMessages(messages, null);
    //print(format("DEBUG:\n{}\n\n", promptResult.$1));
    params.setPrompt(promptResult.$1);

    // now we load the grammar from the llama.cpp project
    File grammarFile = File('src/woolycore/llama.cpp/grammars/json.gbnf');
    String grammarRules = grammarFile.readAsStringSync();
    params.setGrammar(grammarRules);

    test('Parameter creation test', () {
      expect(params, isNotNull);
      expect(params.prompt, isNotNull);
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
