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
  group('Fancy bindings embedding test', () {
    final libFilepath = getPlatformLibraryFilepath();
    var llamaModel = LlamaModel(libFilepath);

    final modelFilepath = Platform.environment['WOOLY_TEST_EMB_MODEL_FILE'];
    if (modelFilepath == null) {
      print(
          'Set WOOLY_TEST_EMB_MODEL_FILE environment variable to the gguf embedding model to use for testing');
      return;
    }

    final modelParams = llamaModel.getDefaultModelParams();
    modelParams.n_gpu_layers = 100;
    final contextParams = llamaModel.getDefaultContextParams();
    contextParams.n_ctx = 2048;

    // the test is designed for nomic-ai/nomic-embed-text-v1.5-GGUF which has 2048 context by default.
    contextParams.n_batch = 2048;

    // setup embedding specific behaviors
    contextParams.embeddings = true;
    contextParams.n_ubatch = contextParams.n_batch;
    contextParams.pooling_type = LlamaPoolingType.mean.value;

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

    // setup some test sentences to test for similarity against the first prompt
    final prompts = [
      "That is a happy person.",
      "That's a very happy person.",
      "She is not happy about the news.",
      "Is that a happy dog?",
      "Behold an individual brimming with boundless joy and contentment.",
      "The weather is beautiful today.",
      "The sun is shining brightly.",
      "I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion.",
    ];

    // tokenize the prompts
    List<TokenList> tokenizedPrompts = [];
    for (final p in prompts) {
      final tokens = llamaModel.tokenizeText(p, true, false);
      tokenizedPrompts.add(tokens);
    }

    // do a little additional check of the detokenizer
    print('\nDetokenized prompts:\n');
    for (final tp in tokenizedPrompts) {
      final roundTripped = llamaModel.detokenizeToText(tp, false);
      print('\t$roundTripped');
    }

    // build the embeddings for all of them
    final embeddings = llamaModel.makeEmbeddings(
        EmbeddingNormalization.euclidean, tokenizedPrompts);
    test('Create embedding test', () {
      expect(embeddings.length, 8);
      expect(embeddings.first.isNotEmpty, true);
    });
    print(
        "\nWe got back ${embeddings.length} embeddings, of size ${embeddings.first.length}");

    // calculate the sentence similarity scores and print them out
    print('\n\nTesting similarity to: "${prompts.first}"\n');
    for (int i = 0; i < prompts.length; ++i) {
      final score = similarityCos(embeddings[0], embeddings[i]);
      print('\t$score: ${prompts[i]}');
    }

    // free the allocated memory
    tearDownAll(() {
      llamaModel.freeModel();
    });
  });
}
