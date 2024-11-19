import 'dart:ffi';
import 'dart:io';
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

    // load the test model up
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
    test('Model load test', () {
      expect(loadedResult, true);
    });
    if (loadedResult == false) {
      throw Exception('Failed to load the test model! Test aborted.');
    }

    final params = llamaModel.getTextGenParams();
    params.seed = 42;
    params.n_threads = -1;
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
    params.setPrompt(
        "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n");
    params.setAntiprompts([
      "<|end|>",
    ]);
    params.dry_multiplier = 0.8;
    params.dry_base = 1.75;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = -1;
    params.setDrySequenceBreakers(["\n", ":", "\"", "*"]);

    // do just the prompt processing and stop
    final (promptTokenCount, firstSampler) = llamaModel.processPrompt(params);
    final firstSamplerAlive = firstSampler.isAlive;
    test('prompt ingestion test', () {
      expect(promptTokenCount > 0, true);
      expect(firstSamplerAlive, true);
    });

    // we freeze the state after processing the prompt so that we can generate
    // a second block of text after the first one without having to reprocess
    // the prompt. not as big of a deal when your prompt is 31 tokens, but it
    // IS a bigger deal when your prompt is 31,000 tokens.
    final frozenPrompt = llamaModel.freezePrompt(params);
    final frozenPromptIsAlive = frozenPrompt.isAlive;
    test('frozen object alive test', () {
      expect(frozenPromptIsAlive, true);
    });

    TokenList predictions = [];
    while (predictions.length < params.n_predict) {
      // start by sampling the next token
      Token nextToken = llamaModel.sampleNextToken(firstSampler);

      // check to see if it should stop the text prediction process
      if (llamaModel.checkEogAndAntiprompt(params, firstSampler)) {
        print(
            "End of generation or antiprompt token encountered - halting prediction...");
        break;
      }

      // run the model to calculate the next logits for the next token prediction,
      // but only do this if it's not the last iteration of the loop.
      if (predictions.length < params.n_predict) {
        final success = llamaModel.processNextToken(nextToken);
        test('processNextToken test', () {
          expect(success, true);
        });
      }

      // we add the token to our list as the last operation specifically because
      // we don't want the length of the collection to offset the position
      // by 1 in `processNextToken()` above...
      predictions.add(nextToken);
    }

    // now that the prediction is finished, turn the tokens to text and print it
    var predictionText = llamaModel.detokenizeToText(predictions, false);
    print("Prediction: (tokens: ${predictions.length})\n\n$predictionText");
    test('predicted text detokenization test', () {
      expect(predictionText != null, true);
      expect(predictionText!.isNotEmpty, true);
    });

    final firstPredictionText = predictionText;
    final firstPredictionCount = predictions.length;

    // additionally, we're going to take another snapshot here and resume
    // our prediction after the second text generation segment...
    final frozenPrediction =
        llamaModel.freezePromptWithPrediction(params, predictions);
    final frozenPredictionIsAlive = frozenPrediction.isAlive;
    test('frozen object alive test', () {
      expect(frozenPredictionIsAlive, true);
    });

    // we're done with the first sampler now so we have to free it
    llamaModel.freeGptSampler(firstSampler);

    test('free object alive test', () {
      expect(firstSampler.isAlive, false);
    });

    print("\n~~~ ---- ~~~~\n\n");

    // take the state from the frozen prompt above and restore it. this *does*
    // make a new sampler and we have to use that in the next sampling loop.
    final (defrostedTokenCount, secondSampler) =
        llamaModel.defrostFrozenState(params, frozenPrompt);
    test('defrost prompt test', () {
      expect(defrostedTokenCount, promptTokenCount);
    });

    // inject a little more prompt in just to make it different and test
    // the ability to add more prompt to ingest. This should produce a
    // distinctly different result than the first prediction.
    final newPromptText =
        "Do you have a suggestion for genre?<|end|>\n<|user|>\nMake it like a Pixar movie script, but with those two authors!<|end|>\n<|assistant|>\n";
    final newPromptTokenCount =
        llamaModel.processAdditionalPrompt(secondSampler, newPromptText);
    test('additional prompt test', () {
      expect(newPromptTokenCount > 0, true);
    });

    // reset our prediction list and do another prediction cycle.
    predictions = [];
    while (predictions.length < params.n_predict) {
      // start by sampling the next token
      Token nextToken = llamaModel.sampleNextToken(secondSampler);

      // check to see if it should stop the text prediction process
      if (llamaModel.checkEogAndAntiprompt(params, secondSampler)) {
        print(
            "End of generation or antiprompt token encountered - halting prediction...");
        break;
      }

      // run the model to calculate the next logits for the next token prediction,
      // but only do this if it's not the last iteration of the loop.
      if (predictions.length < params.n_predict) {
        final success = llamaModel.processNextToken(nextToken);
        test('processNextToken test', () {
          expect(success, true);
        });
      }

      predictions.add(nextToken);
    }

    // now that the prediction is finished, turn the tokens to text and print it
    predictionText = llamaModel.detokenizeToText(predictions, false);
    print("Prediction: (tokens: ${predictions.length})\n\n$predictionText\n");
    test('predicted text detokenization test', () {
      expect(predictionText != null, true);
      expect(predictionText!.isNotEmpty, true);
    });

    // we're finished with the second sampler so let it roam free, like a bird
    llamaModel.freeGptSampler(secondSampler);

    // we're also done with the frozen prompt state now, so release that too
    llamaModel.freeFrozenState(frozenPrompt);

    test('free object alive test', () {
      expect(secondSampler.isAlive, false);
      expect(frozenPrompt.isAlive, false);
    });

    print("\n~~~ ---- ~~~~\n\n");

    // defrost the state we saved with our initial predictions.
    // this should mean that the additional prompt string we added above will no longer
    // be there and the prediction can continue as it would have initially.
    final (defrostedPredictionTokenCount, thirdSampler) =
        llamaModel.defrostFrozenState(params, frozenPrediction);
    test('defrost prompt with prediction test', () {
      expect(defrostedPredictionTokenCount,
          promptTokenCount + firstPredictionCount);
    });

    // continue our first prediction cycle with another run
    predictions = [];
    while (predictions.length < params.n_predict) {
      // start by sampling the next token
      Token nextToken = llamaModel.sampleNextToken(thirdSampler);

      // check to see if it should stop the text prediction process
      if (llamaModel.checkEogAndAntiprompt(params, thirdSampler)) {
        print(
            "End of generation or antiprompt token encountered - halting prediction...");
        break;
      }

      // Note: it's important to account for the new token count from the
      // frozen prediction state or else the continuation won't make sense.
      if (predictions.length < params.n_predict) {
        final success = llamaModel.processNextToken(nextToken);
        test('processNextToken test', () {
          expect(success, true);
        });
      }

      predictions.add(nextToken);
    }

    // print out our first prediction followed by this third prediction to see
    // the full continuation.
    predictionText = llamaModel.detokenizeToText(predictions, false);
    print(
        "Final Prediction: (new tokens: ${predictions.length})\n\n$firstPredictionText$predictionText\n\n");

    // free the remaining frozen state and sampler
    llamaModel.freeGptSampler(thirdSampler);
    llamaModel.freeFrozenState(frozenPrediction);

    test('free object alive test', () {
      expect(thirdSampler.isAlive, false);
      expect(frozenPrediction.isAlive, false);
    });

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
