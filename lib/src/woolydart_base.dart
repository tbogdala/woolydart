import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:woolydart/src/llama_cpp_bindings.dart';

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
    // case scenario of a whole context size with four bytes per utf-8.
    final outputText =
        calloc.allocate(_loadedContextLength * 4) as Pointer<Char>;

    var predictResult = lib.wooly_predict(
        params, _ctx, _model, false, outputText, _lastPromptCache, onNewToken);

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
