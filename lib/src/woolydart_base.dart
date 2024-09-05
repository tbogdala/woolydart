import 'dart:ffi';
import 'dart:math';

import 'package:ffi/ffi.dart';
import 'package:woolydart/src/llama_cpp_bindings.dart';

enum LlamaPoolingType {
  unspecified(-1),
  none(0),
  mean(1),
  cls(2),
  last(3);

  final int value;

  const LlamaPoolingType(this.value);
}

enum EmbeddingNormalization {
  none(-1),
  maxAbsoluteInt16(0),
  taxicab(1),
  euclidean(2),
  pNorm(3);

  final int value;

  const EmbeddingNormalization(this.value);
}

typedef Token = int;
typedef TokenList = List<Token>;
typedef Embedding = List<double>;

class LlamaModel {
  late woolydart lib;

  // internal handle to the context
  Pointer<Void> _ctx = nullptr;

  // internal handle to the loaded model
  Pointer<Void> _model = nullptr;

  // internal handle to prompt cache from last prediction
  Pointer<Void> _lastPromptCache = nullptr;

  // the size of the context used to load the model
  int _loadedContextLength = 0;

  // the model parameters used when loading the current model
  wooly_llama_model_params? _loadedModelParams = null;

  // the context parameters used when loading the current model
  wooly_llama_context_params? _loadedContextParams = null;

  // Construct a new LlamaModel wrapper for llama.cpp by giving it a filepath
  // to the compiled library. On Android, this might be 'libwoolydart.so'. On iOS
  // this might be empty, ''. If the libFilepath parameter is empty, then it will
  // attempt to just use the active process instead of another library file.
  // On desktop, this might be a full version to the compiled binary,
  // like 'src/llama.cpp/build/libllama.dylib'.
  LlamaModel(String libFilepath) {
    if (libFilepath.isNotEmpty) {
      lib = woolydart(DynamicLibrary.open(libFilepath));
    } else {
      lib = woolydart(DynamicLibrary.process());
    }
  }

  // Load a model at the filepath specified in modelFile. The model and context
  // parameters are used when loading the model and creating a new context to
  // operate from. The silenceLlamaCpp boolean will allow the client code to
  // disable all of the information that upstream llama.cpp writes to output streams.
  //
  // Should the process fail, false is returned.
  bool loadModel(String modelFile, wooly_llama_model_params modelParams,
      wooly_llama_context_params contextParams, bool silenceLlamaCpp) {
    // if we have a cached prompt or a loaded model, we free the memory now
    if (_lastPromptCache != nullptr || _model != nullptr || _ctx != nullptr) {
      freeModel();
    }

    var nativeModelPath = modelFile.toNativeUtf8();
    var loadedModel = lib.wooly_load_model(nativeModelPath as Pointer<Char>,
        modelParams, contextParams, silenceLlamaCpp);
    malloc.free(nativeModelPath);

    if (loadedModel.ctx == nullptr || loadedModel.model == nullptr) {
      return false;
    }

    _model = loadedModel.model;
    _ctx = loadedModel.ctx;
    _lastPromptCache = nullptr;
    _loadedContextLength = loadedModel.context_length;
    _loadedModelParams = modelParams;
    _loadedContextParams = contextParams;

    return true;
  }

  // Unloads the model completely.
  void freeModel() {
    lib.wooly_free_model(_ctx, _model);
    if (_lastPromptCache != nullptr) {
      lib.wooly_free_prompt_cache(_lastPromptCache);
      _lastPromptCache = nullptr;
    }
    _ctx = nullptr;
    _model = nullptr;
    _loadedContextLength = 0;
    _loadedModelParams = null;
    _loadedContextParams = null;
  }

  // Returns true if a model is currently loaded, false otherwise.
  bool isModelLoaded() {
    if (_model != nullptr && _ctx != nullptr) {
      return true;
    } else {
      return false;
    }
  }

  // Gets a default copy of the context parameters using default settings from
  // llama.cpp.
  wooly_llama_context_params getDefaultContextParams() {
    return lib.wooly_get_default_llama_context_params();
  }

  // Gets a default copy of the model parameters using default settings from
  // llama.cpp upstream.
  wooly_llama_model_params getDefaultModelParams() {
    return lib.wooly_get_default_llama_model_params();
  }

  // Gets a new copy of the parameters used to control text generation. Under
  // the hood, this is a simplified form of the `gpt_params` struct from
  // llama.cpp's 'common.h' header, which has a bunch of c++ members that
  // cannot be bound with ffigen. A default copy of `gpt_params` is created
  // to pull initial values from.
  wooly_gpt_params getTextGenParams() {
    return lib.wooly_new_gpt_params();
  }

  // Runs text inferrence on the loaded model to predict text based on the set
  // of parameters provided by `params`. An callback function can be supplied
  // for `onNewToken` to provide a function that returns a bool as to whether
  // or not prediction should continue at each new token being predicted; it
  // can be set to `nullptr` if this feature is unneeded.
  (wooly_predict_result, String?) predictText(
      wooly_gpt_params params, wooly_token_update_callback onNewToken) {
    // allocate the buffer for the predicted text. by default we just use the worst
    // case scenario of a whole context size with four bytes per utf-8 and ten
    // characters per token.
    final outputTextSize = _loadedContextLength * 4 * 10;
    final outputText = calloc.allocate(outputTextSize) as Pointer<Char>;

    var predictResult = lib.wooly_predict(params, _ctx, _model, false,
        outputText, outputTextSize, _lastPromptCache, onNewToken);

    String? outputString;

    // if we had a successful run, try to make the output string
    if (predictResult.result == 0) {
      outputString = (outputText as Pointer<Utf8>).toDartString();
      if (params.prompt_cache_all) {
        _lastPromptCache = predictResult.prompt_cache;
      } else {
        lib.wooly_free_prompt_cache(predictResult.prompt_cache);
      }
    }

    calloc.free(outputText);
    return (predictResult, outputString);
  }

  // returns the token count for the `textPrompt` when processed by the loaded
  // model's tokenizer. `addSpecial` controls whether or not to add special tokens
  // when encoding sequences, such as 'bos' or 'eos' tokens. `parseSpecial`
  // controls whether or not to parse additional 'special' tokens defined for
  // the model, such as '<|begin_of_text|>' for Llama-3.
  int getTokenCount(
    String textPrompt,
    bool addSpecial,
    bool parseSpecial,
  ) {
    final textPtr = textPrompt.toNativeUtf8() as Pointer<Char>;
    final tokenCount = lib.wooly_llama_tokenize(
        _model, textPtr, addSpecial, parseSpecial, nullptr, 0);

    malloc.free(textPtr);
    return tokenCount;
  }

  // returns a List of ints represnting the tokens generated by the loaded
  // model for a given `textPrompt`. If `addSpecial` is true, the special
  // tokens like 'bos' are added. If `parseSpecial is true, the tokenizer
  // will look for the additional special tokens configured for the model
  // and tokenize them accordingly.
  TokenList tokenizeText(
      String textPrompt, bool addSpecial, bool parseSpecial) {
    // allocate the native string for the prompt
    final textPtr = textPrompt.toNativeUtf8() as Pointer<Char>;
    // allocate the buffer for the output parameter, one token per character
    // as a worst case performance.
    final outTokensBufferLen = textPrompt.length;
    final Pointer<Int32> outTokensBuffer = malloc<Int32>(outTokensBufferLen);

    // get the tokens
    final tokenCount = lib.wooly_llama_tokenize(_model, textPtr, addSpecial,
        parseSpecial, outTokensBuffer, outTokensBufferLen);

    // convert it to a Dart compatible data type that is warm and inviting...
    final TokenList results = tokenCount > 0
        ? TokenList.generate(tokenCount, (int i) => outTokensBuffer[i])
        : [];

    malloc.free(textPtr);
    malloc.free(outTokensBuffer);
    return results;
  }

  String? detokenizeToText(TokenList tokens, bool renderSpecials) {
    // build an output buffer for the text based on a worse case scenario
    // of a full context, ten characters per token and four chars per utf-8.
    var outputTextSize = _loadedContextLength * 4 * 10;
    var outputText = calloc.allocate(outputTextSize) as Pointer<Char>;

    // build the buffer for the input tokens
    final Pointer<Int32> tokenListNative = malloc<Int32>(tokens.length);
    for (int j = 0; j < tokens.length; j++) {
      tokenListNative[j] = tokens[j];
    }

    // call into the library to detokenize.
    var detokenCount = lib.wooly_llama_detokenize(_ctx, renderSpecials,
        tokenListNative, tokens.length, outputText, outputTextSize);

    // if we didn't pass the right size buffer, free it and recreate
    // with the absolute value of the returned number and try again.
    String? returnVal = null;
    if (detokenCount < 0) {
      calloc.free(outputText);
      outputTextSize = detokenCount.abs() + 1;
      outputText = calloc.allocate(outputTextSize) as Pointer<Char>;
      detokenCount = lib.wooly_llama_detokenize(_ctx, renderSpecials,
          tokenListNative, tokens.length, outputText, outputTextSize);
    }

    // make our Dart string from the result if we got detokenized characters.
    if (detokenCount > 0) {
      returnVal = (outputText as Pointer<Utf8>).toDartString();
    }

    calloc.free(outputText);
    malloc.free(tokenListNative);
    return returnVal;
  }

  List<Embedding> makeEmbeddings(
      EmbeddingNormalization embdNormalize, List<TokenList> tokenizedPrompts) {
    // figure out how big of an output buffer we need
    int embdNeeded;
    if (_loadedContextParams != null &&
        _loadedContextParams!.pooling_type == LlamaPoolingType.none.value) {
      // no pooling means we get a full embedding vector for every token, so
      // go through all the prompts and figure out the total number of tokens
      // and change the needed float count accordingly
      embdNeeded = 0;
      for (final p in tokenizedPrompts) {
        embdNeeded += p.length;
      }
    } else {
      embdNeeded = tokenizedPrompts.length;
    }

    // find the size of the embedding vectors and scale the floats needed
    // by that size and make the output buffer.
    final nEmbd = lib.wooly_llama_n_embd(_model);
    int embdFloatsNeeded = embdNeeded * nEmbd;
    final Pointer<Float> outEmbeddingsBuffer = malloc<Float>(embdFloatsNeeded);

    // allocate the main arrays going to woolycore
    final Pointer<Pointer<Int32>> tokenListsNative =
        malloc<Pointer<Int32>>(tokenizedPrompts.length);
    final Pointer<Int64> tokenListSizesNative =
        malloc<Int64>(tokenizedPrompts.length);

    // now allocate individual arrays for each tokenized prompt and fill them up
    for (int i = 0; i < tokenizedPrompts.length; i++) {
      final TokenList tokenList = tokenizedPrompts[i];
      final Pointer<Int32> tokenListNative = malloc<Int32>(tokenList.length);

      for (int j = 0; j < tokenList.length; j++) {
        tokenListNative[j] = tokenList[j];
      }

      tokenListsNative[i] = tokenListNative;
      tokenListSizesNative[i] = tokenList.length;
    }

    // generate the tokens, should return 0 on success
    var ret = lib.wooly_llama_make_embeddings(
        _model,
        _ctx,
        _loadedContextLength,
        _loadedContextParams?.pooling_type ?? LlamaPoolingType.mean.value,
        embdNormalize.value,
        tokenizedPrompts.length,
        tokenListsNative,
        tokenListSizesNative,
        outEmbeddingsBuffer,
        embdFloatsNeeded);

    // free all of our input buffers
    for (int i = 0; i < tokenizedPrompts.length; i++) {
      malloc.free(tokenListsNative[i]);
    }
    malloc.free(tokenListsNative);
    malloc.free(tokenListSizesNative);

    // if we failed, return an empty list
    if (ret != 0) {
      malloc.free(outEmbeddingsBuffer);
      return List<Embedding>.empty();
    } else {
      List<Embedding> results = [];
      for (int i = 0; i < embdNeeded; ++i) {
        Embedding embeddingVector = [];
        final offsetArray = outEmbeddingsBuffer + (i * nEmbd);
        for (int j = 0; j < nEmbd; ++j) {
          embeddingVector.add(offsetArray[j]);
        }
        results.add(embeddingVector);
      }
      malloc.free(outEmbeddingsBuffer);
      return results;
    }
  }
}

extension GptParamsSimpleExtension on wooly_gpt_params {
  // Frees the native strings used by the parameters and must be called
  // when the client code is done with the object to avoid memory leaks.
  void dispose() {
    freePrompt();
    freeAntiprompts();
    freeGrammar();
  }

  // Frees the memory used by the prompt native string.
  void freePrompt() {
    if (prompt != nullptr) {
      malloc.free(prompt);
      prompt = nullptr;
    }
  }

  // Sets the prompt string for the parameters taking care of the conversion
  // to a C compatible character array.
  void setPrompt(String newPrompt) {
    // if we already had a native prompt string, free it.
    if (prompt != nullptr) {
      freePrompt();
    }
    prompt = newPrompt.toNativeUtf8() as Pointer<Char>;
  }

  // Frees the memory used by the grammar native string.
  void freeGrammar() {
    if (grammar != nullptr) {
      malloc.free(grammar);
      grammar = nullptr;
    }
  }

  // Sets the grammar string, using llama.cpp's BNF-like syntax to constrain output,
  // for the parameters taking care of the conversation to a C compatible
  // character array.
  void setGrammar(String newGrammar) {
    if (grammar != nullptr) {
      freeGrammar();
    }
    grammar = newGrammar.toNativeUtf8() as Pointer<Char>;
  }

  // Frees the memory used by the antiprompt native strings
  void freeAntiprompts() {
    if (antiprompts != nullptr) {
      for (int ai = 0; ai < antiprompt_count; ai++) {
        malloc.free(antiprompts[ai]);
      }
      malloc.free(antiprompts);
      antiprompts = nullptr;
      antiprompt_count = 0;
    }
  }

  // Sets the antiprompt strings for the parameters taking care of the
  // conversion to a C compatible set of character arrays.
  void setAntiprompts(List<String> newAntiprompts) {
    freeAntiprompts();

    if (newAntiprompts.isNotEmpty) {
      // allocate all the array of pointers.
      final Pointer<Pointer<Char>> antiPointers =
          calloc.allocate(newAntiprompts.length * sizeOf<Pointer<Char>>());

      // allocate each of the native strings
      for (int ai = 0; ai < newAntiprompts.length; ai++) {
        Pointer<Char> native =
            newAntiprompts[ai].toNativeUtf8() as Pointer<Char>;
        antiPointers[ai] = native;
      }

      antiprompts = antiPointers;
      antiprompt_count = newAntiprompts.length;
    }
  }
}

// calculates the similarity cosine for embedding vectors.
double similarityCos(Embedding embd1, Embedding embd2) {
  double sum = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;

  for (int i = 0; i < embd1.length; i++) {
    sum += embd1[i] * embd2[i];
    sum1 += embd1[i] * embd1[i];
    sum2 += embd2[i] * embd2[i];
  }

  // Handle the case where one or both vectors are zero vectors
  if (sum1 == 0.0 || sum2 == 0.0) {
    if (sum1 == 0.0 && sum2 == 0.0) {
      return 1.0; // two zero vectors are similar
    }
    return 0.0;
  }

  return sum / (sqrt(sum1) * sqrt(sum2));
}
