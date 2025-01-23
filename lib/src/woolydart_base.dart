import 'dart:ffi';
import 'dart:math';

import 'package:ffi/ffi.dart';
import 'package:woolydart/src/llama_cpp_bindings.dart';

class ChatMessage {
  String role;
  String content;

  ChatMessage(this.role, this.content);
}

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

// this is just a basic class wrapper around a void pointer just for better
// clarity in the API.
class FrozenState {
  final Pointer<wooly_prompt_cache_t> _cachedState;
  bool isAlive = true;

  FrozenState(this._cachedState);
}

// this is just a basic class wrapper around a void pointer just for better
// clarity in the API.
class GptSampler {
  final Pointer<wooly_sampler_t> _samplerPtr;
  bool isAlive = true;

  GptSampler(this._samplerPtr);
}

class LlamaModel {
  late woolydart lib;

  // internal handle to the context
  Pointer<wooly_llama_context_t> _ctx = nullptr;

  // internal handle to the loaded model
  Pointer<wooly_llama_model_t> _model = nullptr;

  // internal handle to prompt cache from last prediction
  Pointer<wooly_prompt_cache_t> _lastPromptCache = nullptr;

  // the size of the context used to load the model
  int _loadedContextLength = 0;

  // the context parameters used when loading the current model
  wooly_llama_context_params? _loadedContextParams;

  // the last used model filepath when calling `loadModel`
  String? loadedModelFilepath;

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
    _loadedContextParams = contextParams;
    loadedModelFilepath = modelFile;

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
    _loadedContextParams = null;
    loadedModelFilepath = null;
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

  // Constructs a prompt from a list of chat messages and applies a chat template
  // or uses the default one in the GGUF model if `templateOverride` is left `null`.
  // Optionally, if `addAssistant` is true, the start of an assistant block is
  // added to the end of the generated prompt, which is perfect for making a new
  // response.
  (String, int) makePromptFromMessages(
      List<ChatMessage> messages, bool addAssistant, String? templateOverride) {
    // handle empty lists
    if (messages.isEmpty) {
      return ("", 0);
    }

    var prompt = "";
    var numProcessed = 0;

    // this is the compatible message log we build out to send over FFI
    final messageLog = calloc<wooly_chat_message>(messages.length);

    // this is the template name override if supplied
    Pointer<Char> templateOverrideNative;

    if (templateOverride == null) {
      templateOverrideNative = nullptr;
    } else {
      templateOverrideNative = templateOverride.toNativeUtf8() as Pointer<Char>;
    }
    try {
      // Populate the allocated memory with data from your Dart list
      for (int i = 0; i < messages.length; i++) {
        final message = messages[i];
        messageLog[i].role = message.role.toNativeUtf8() as Pointer<Char>;
        messageLog[i].content = message.content.toNativeUtf8() as Pointer<Char>;
      }

      // build a prompt output buffer
      final outputTextSize = _loadedContextLength * 4 * 10;
      final outputText = calloc.allocate(outputTextSize) as Pointer<Char>;

      // try to apply the chat template to the message log
      numProcessed = lib.wooly_apply_chat_template(
          _model,
          templateOverrideNative,
          addAssistant,
          messageLog,
          messages.length,
          outputText,
          outputTextSize);

      prompt = (outputText as Pointer<Utf8>).toDartString();
    } finally {
      // Free the allocated memory
      for (int i = 0; i < messages.length; i++) {
        calloc.free(messageLog[i].role);
        calloc.free(messageLog[i].content);
      }
      if (templateOverrideNative != nullptr) {
        calloc.free(templateOverrideNative);
      }
      calloc.free(messageLog);
    }

    return (prompt, numProcessed);
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

  // Takes the `params` passed in and sets up a new sampler as well as runs
  // the prompt through the loaded model for processing. The returned GptSampler
  // can be passed to the low-level wrapper functions for text prediction. If the
  // returned int is a negative number there was an error, otherwise it's the number
  // of tokens processed for the prompt.
  (int, GptSampler) processPrompt(wooly_gpt_params params) {
    final results = lib.wooly_process_prompt(params, _ctx, _model);
    return (results.result, GptSampler(results.gpt_sampler));
  }

  // Takes the sampler that was returned from a previous `processPrompt()` call
  // and applies more prompt text, updating the sampler state. The returned value
  // is the number of tokens added or a negative number on error.
  int processAdditionalPrompt(GptSampler gptSampler, String additionalPrompt) {
    final textPtr = additionalPrompt.toNativeUtf8() as Pointer<Char>;
    final retValue = lib.wooly_process_additional_prompt(
        _ctx, _model, gptSampler._samplerPtr, textPtr);
    malloc.free(textPtr);

    return retValue;
  }

  // Takes the void pointer to the sampler returned with `processPrompt()` and
  // ONLY samples the next token, which is returned.
  Token sampleNextToken(GptSampler gptSampler) {
    return lib.wooly_sample_next(_ctx, gptSampler._samplerPtr);
  }

  // Checks to see if the last sample token was the end-of-generation token
  // defined by the model or one of the antiprompts strings in the supplied
  // `params` parameter. Returns `true` if it is an eog or antiprompt,
  // and returns false otherwise.
  bool checkEogAndAntiprompt(wooly_gpt_params params, GptSampler gptSampler) {
    final result = lib.wooly_check_eog_and_antiprompt(
        params, _ctx, _model, gptSampler._samplerPtr);
    if (result == 0) {
      return false;
    } else {
      return true;
    }
  }

  // This function takes the token sampled from `sampledNextToken` and runs
  // it through the loaded model to compute everything necessary to sample
  // another token following it. This is a compute heavy function call.
  //
  // The function returns `true` if it was successful and `false` on error.
  bool processNextToken(Token nextToken) {
    final result = lib.wooly_process_next_token(_ctx, nextToken);
    if (result == 0) {
      return true;
    } else {
      return false;
    }
  }

  // This function 'freezes' the state of the model, pulling the prompt tokens
  // from the `params` passed in. This should be called before any further text
  // prediction is done with the model. If the prediction state should be frozen
  // too, use `freezePromptWithPrediction()` instead. The returned FrozenState
  // object should be freed when finished with it by calling `freeFrozenState()`.
  FrozenState freezePrompt(wooly_gpt_params params) {
    final cachedPtr =
        lib.wooly_freeze_prediction_state(params, _ctx, _model, nullptr, 0);
    return FrozenState(cachedPtr);
  }

  // This function 'freezes' the state of the model, pulling the prompt tokens
  // from the `params passed in and storing a copy of the  predicted tokens.
  // The returned FrozenState object should be freed when finished with it
  // by calling `freeFrozenState()`.
  FrozenState freezePromptWithPrediction(
      wooly_gpt_params params, TokenList predictions) {
    // build the buffer for the input prediction tokens
    final Pointer<Int32> tokenListNative = malloc<Int32>(predictions.length);
    for (int j = 0; j < predictions.length; j++) {
      tokenListNative[j] = predictions[j];
    }

    final cachedPtr = lib.wooly_freeze_prediction_state(
        params, _ctx, _model, tokenListNative, predictions.length);

    malloc.free(tokenListNative);
    return FrozenState(cachedPtr);
  }

  // This function `defrosts` a frozen state of the model, which will restore
  // the internal state for prediction, from a frozen state made earlier. The
  // tuple returned is the number of tokens restored (useful for calculating the
  // position of future predicted tokens) and the new GptSampler to be used when
  // sampling tokens.
  (int, GptSampler) defrostFrozenState(
      wooly_gpt_params params, FrozenState frozenState) {
    final results = lib.wooly_defrost_prediction_state(
        params, _ctx, _model, frozenState._cachedState);
    return (results.result, GptSampler(results.gpt_sampler));
  }

  // Frees the memory associated with the cached frozen state.
  void freeFrozenState(FrozenState ice) {
    lib.wooly_free_prompt_cache(ice._cachedState);
    ice.isAlive = false;
  }

  // Frees the memory associated with the sample.
  void freeGptSampler(GptSampler sampler) {
    lib.wooly_free_sampler(sampler._samplerPtr);
    sampler.isAlive = false;
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
        _ctx, textPtr, addSpecial, parseSpecial, nullptr, 0);

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
    final tokenCount = lib.wooly_llama_tokenize(_ctx, textPtr, addSpecial,
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
    String? returnVal;
    if (detokenCount < 0) {
      calloc.free(outputText);
      outputTextSize = detokenCount.abs() + 1;
      outputText = calloc.allocate(outputTextSize) as Pointer<Char>;
      detokenCount = lib.wooly_llama_detokenize(_ctx, renderSpecials,
          tokenListNative, tokens.length, outputText, outputTextSize);
    }

    // make our Dart string from the result if we got detokenized characters.
    try {
      if (detokenCount > 0) {
        returnVal = (outputText as Pointer<Utf8>).toDartString();
      }
      return returnVal;
    } catch (e) {
      return null;
    } finally {
      calloc.free(outputText);
      malloc.free(tokenListNative);
    }
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
    freeDrySequenceBreakers();
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

  // Frees the memory used by the DRY sampler sequence breaker native strings
  void freeDrySequenceBreakers() {
    if (dry_sequence_breakers != nullptr) {
      for (int sb = 0; sb < dry_sequence_breakers_count; sb++) {
        malloc.free(dry_sequence_breakers[sb]);
      }
      malloc.free(dry_sequence_breakers);
      dry_sequence_breakers = nullptr;
      dry_sequence_breakers_count = 0;
    }
  }

  // Sets the DRY sampler sequence breaking strings for the parameters taking care of the
  // conversion to a C compatible set of character arrays.
  void setDrySequenceBreakers(List<String> newSequenceBreakers) {
    freeDrySequenceBreakers();

    if (newSequenceBreakers.isNotEmpty) {
      // allocate all the array of pointers.
      final Pointer<Pointer<Char>> strPointers =
          calloc.allocate(newSequenceBreakers.length * sizeOf<Pointer<Char>>());

      // allocate each of the native strings
      for (int sb = 0; sb < newSequenceBreakers.length; sb++) {
        Pointer<Char> native =
            newSequenceBreakers[sb].toNativeUtf8() as Pointer<Char>;
        strPointers[sb] = native;
      }

      dry_sequence_breakers = strPointers;
      dry_sequence_breakers_count = newSequenceBreakers.length;
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
