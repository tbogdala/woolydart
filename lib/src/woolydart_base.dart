import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:woolydart/woolydart.dart';

class LlamaModel {
  late woolydart lib;

  Pointer<llama_context> ctx = nullptr; // internal handle to the context
  Pointer<llama_model> model = nullptr; // internal handle to the loaded model

  // the size of the context used to load the model
  int _loadedContextLength = 0;

  // Construct a new LlamaModel wrapper for llama.cpp by giving it a filepath
  // to the compiled library. On Android, this might be 'libllama.so'. On iOS
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
  // operate from. Should the process fail, false is returned.
  bool loadModel(String modelFile, llama_model_params modelParams,
      llama_context_params contextParams) {
    var nativeModelPath =
        "/Users/timothy/.cache/lm-studio/models/SanctumAI/Phi-3-mini-4k-instruct-GGUF/phi-3-mini-4k-instruct.Q8_0.gguf"
            .toNativeUtf8();

    var loadedModel = lib.wooly_load_model(
        nativeModelPath as Pointer<Char>, modelParams, contextParams);

    malloc.free(nativeModelPath);

    if (loadedModel.ctx == nullptr || loadedModel.model == nullptr) {
      return false;
    }

    model = loadedModel.model;
    ctx = loadedModel.ctx;
    _loadedContextLength = lib.llama_n_ctx(ctx);
    return true;
  }

  void freeModel() {
    lib.wooly_free_model(ctx, model);
    ctx = nullptr;
    model = nullptr;
    _loadedContextLength = 0;
  }

  // Gets a default copy of the context parameters using default settings from
  // llama.cpp.
  llama_context_params getDefaultContextParams() {
    var contextParams = lib.llama_context_default_params();
    return contextParams;
  }

  // Gets a default copy of the model parameters using default settings from
  // llama.cpp upstream.
  llama_model_params getDefaultModelParams() {
    return lib.llama_model_default_params();
  }

  // Gets a new copy of the parameters used to control text generation. Under
  // the hood, this is a simplified form of the `gpt_params` struct from
  // llama.cpp's 'common.h' header, which has a bunch of c++ members that
  // cannot be bound with ffigen. A default copy of `gpt_params` is created
  // to pull initial values from.
  gpt_params_simple getTextGenParams() {
    return lib.wooly_new_params();
  }

  (wooly_predict_result, String?) predictText(gpt_params_simple params) {
    // allocate the buffer for the predicted text. by default we just use the worst
    // case scenario of a whole context size with four bytes per utf-8.
    final outputText =
        calloc.allocate(_loadedContextLength * 4) as Pointer<Char>;

    var predictResult =
        lib.wooly_predict(params, ctx, model, false, outputText, nullptr);

    String? outputString;

    // if we had a successful run, try to make the output string
    if (predictResult.result == 0) {
      outputString = (outputText as Pointer<Utf8>).toDartString();
    }

    calloc.free(outputText);
    return (predictResult, outputString);
  }
}

extension GptParamsSimpleExtension on gpt_params_simple {
  // Frees the native strings used by the parameters and must be called
  // when the client code is done with the object to avoid memory leaks.
  void dispose() {
    freePrompt();
    freeAntiprompts();
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
      malloc.free(prompt);
      prompt = nullptr;
    }
    prompt = newPrompt.toNativeUtf8() as Pointer<Char>;
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
