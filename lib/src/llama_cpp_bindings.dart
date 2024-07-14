// AUTO GENERATED FILE, DO NOT EDIT.
//
// Generated by `package:ffigen`.
// ignore_for_file: type=lint
import 'dart:ffi' as ffi;

/// llama.cpp binding
class woolydart {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  woolydart(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  woolydart.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  wooly_load_model_result wooly_load_model(
    ffi.Pointer<ffi.Char> fname,
    wooly_llama_model_params model_params,
    wooly_llama_context_params context_params,
    bool silent_llama,
  ) {
    return _wooly_load_model(
      fname,
      model_params,
      context_params,
      silent_llama,
    );
  }

  late final _wooly_load_modelPtr = _lookup<
      ffi.NativeFunction<
          wooly_load_model_result Function(
              ffi.Pointer<ffi.Char>,
              wooly_llama_model_params,
              wooly_llama_context_params,
              ffi.Bool)>>('wooly_load_model');
  late final _wooly_load_model = _wooly_load_modelPtr.asFunction<
      wooly_load_model_result Function(ffi.Pointer<ffi.Char>,
          wooly_llama_model_params, wooly_llama_context_params, bool)>();

  void wooly_free_model(
    ffi.Pointer<ffi.Void> llama_context_ptr,
    ffi.Pointer<ffi.Void> llama_model_ptr,
  ) {
    return _wooly_free_model(
      llama_context_ptr,
      llama_model_ptr,
    );
  }

  late final _wooly_free_modelPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>)>>('wooly_free_model');
  late final _wooly_free_model = _wooly_free_modelPtr.asFunction<
      void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)>();

  wooly_gpt_params wooly_new_gpt_params() {
    return _wooly_new_gpt_params();
  }

  late final _wooly_new_gpt_paramsPtr =
      _lookup<ffi.NativeFunction<wooly_gpt_params Function()>>(
          'wooly_new_gpt_params');
  late final _wooly_new_gpt_params =
      _wooly_new_gpt_paramsPtr.asFunction<wooly_gpt_params Function()>();

  wooly_predict_result wooly_predict(
    wooly_gpt_params simple_params,
    ffi.Pointer<ffi.Void> llama_context_ptr,
    ffi.Pointer<ffi.Void> llama_model_ptr,
    bool include_specials,
    ffi.Pointer<ffi.Char> out_result,
    ffi.Pointer<ffi.Void> prompt_cache_ptr,
    wooly_token_update_callback token_cb,
  ) {
    return _wooly_predict(
      simple_params,
      llama_context_ptr,
      llama_model_ptr,
      include_specials,
      out_result,
      prompt_cache_ptr,
      token_cb,
    );
  }

  late final _wooly_predictPtr = _lookup<
      ffi.NativeFunction<
          wooly_predict_result Function(
              wooly_gpt_params,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
              ffi.Bool,
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Void>,
              wooly_token_update_callback)>>('wooly_predict');
  late final _wooly_predict = _wooly_predictPtr.asFunction<
      wooly_predict_result Function(
          wooly_gpt_params,
          ffi.Pointer<ffi.Void>,
          ffi.Pointer<ffi.Void>,
          bool,
          ffi.Pointer<ffi.Char>,
          ffi.Pointer<ffi.Void>,
          wooly_token_update_callback)>();

  void wooly_free_prompt_cache(
    ffi.Pointer<ffi.Void> prompt_cache_ptr,
  ) {
    return _wooly_free_prompt_cache(
      prompt_cache_ptr,
    );
  }

  late final _wooly_free_prompt_cachePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
          'wooly_free_prompt_cache');
  late final _wooly_free_prompt_cache = _wooly_free_prompt_cachePtr
      .asFunction<void Function(ffi.Pointer<ffi.Void>)>();

  wooly_llama_model_params wooly_get_default_llama_model_params() {
    return _wooly_get_default_llama_model_params();
  }

  late final _wooly_get_default_llama_model_paramsPtr =
      _lookup<ffi.NativeFunction<wooly_llama_model_params Function()>>(
          'wooly_get_default_llama_model_params');
  late final _wooly_get_default_llama_model_params =
      _wooly_get_default_llama_model_paramsPtr
          .asFunction<wooly_llama_model_params Function()>();

  wooly_llama_context_params wooly_get_default_llama_context_params() {
    return _wooly_get_default_llama_context_params();
  }

  late final _wooly_get_default_llama_context_paramsPtr =
      _lookup<ffi.NativeFunction<wooly_llama_context_params Function()>>(
          'wooly_get_default_llama_context_params');
  late final _wooly_get_default_llama_context_params =
      _wooly_get_default_llama_context_paramsPtr
          .asFunction<wooly_llama_context_params Function()>();
}

final class wooly_load_model_result extends ffi.Struct {
  external ffi.Pointer<ffi.Void> model;

  external ffi.Pointer<ffi.Void> ctx;

  @ffi.Uint32()
  external int context_length;
}

final class wooly_predict_result extends ffi.Struct {
  @ffi.Int()
  external int result;

  external ffi.Pointer<ffi.Void> prompt_cache;

  @ffi.Double()
  external double t_start_ms;

  @ffi.Double()
  external double t_end_ms;

  @ffi.Double()
  external double t_load_ms;

  @ffi.Double()
  external double t_sample_ms;

  @ffi.Double()
  external double t_p_eval_ms;

  @ffi.Double()
  external double t_eval_ms;

  @ffi.Int()
  external int n_sample;

  @ffi.Int()
  external int n_p_eval;

  @ffi.Int()
  external int n_eval;
}

final class wooly_llama_model_params extends ffi.Struct {
  @ffi.Int32()
  external int n_gpu_layers;

  @ffi.Int32()
  external int split_mode;

  @ffi.Int32()
  external int main_gpu;

  external ffi.Pointer<ffi.Float> tensor_split;

  @ffi.Bool()
  external bool vocab_only;

  @ffi.Bool()
  external bool use_mmap;

  @ffi.Bool()
  external bool use_mlock;

  @ffi.Bool()
  external bool check_tensors;
}

final class wooly_llama_context_params extends ffi.Struct {
  @ffi.Uint32()
  external int seed;

  @ffi.Uint32()
  external int n_ctx;

  @ffi.Uint32()
  external int n_batch;

  @ffi.Uint32()
  external int n_ubatch;

  @ffi.Uint32()
  external int n_seq_max;

  @ffi.Uint32()
  external int n_threads;

  @ffi.Uint32()
  external int n_threads_batch;

  @ffi.Int32()
  external int rope_scaling_type;

  @ffi.Int32()
  external int pooling_type;

  @ffi.Float()
  external double rope_freq_base;

  @ffi.Float()
  external double rope_freq_scale;

  @ffi.Float()
  external double yarn_ext_factor;

  @ffi.Float()
  external double yarn_attn_factor;

  @ffi.Float()
  external double yarn_beta_fast;

  @ffi.Float()
  external double yarn_beta_slow;

  @ffi.Uint32()
  external int yarn_orig_ctx;

  @ffi.Float()
  external double defrag_thold;

  @ffi.Bool()
  external bool logits_all;

  @ffi.Bool()
  external bool embeddings;

  @ffi.Bool()
  external bool offload_kqv;

  @ffi.Bool()
  external bool flash_attn;
}

final class wooly_gpt_params extends ffi.Struct {
  external ffi.Pointer<ffi.Char> prompt;

  external ffi.Pointer<ffi.Pointer<ffi.Char>> antiprompts;

  @ffi.Int()
  external int antiprompt_count;

  @ffi.Uint32()
  external int seed;

  @ffi.Int32()
  external int n_threads;

  @ffi.Int32()
  external int n_threads_batch;

  @ffi.Int32()
  external int n_predict;

  @ffi.Int32()
  external int n_ctx;

  @ffi.Int32()
  external int n_batch;

  @ffi.Int32()
  external int n_gpu_layers;

  @ffi.Uint32()
  external int split_mode;

  @ffi.Int32()
  external int main_gpu;

  @ffi.Array.multi([128])
  external ffi.Array<ffi.Float> tensor_split;

  @ffi.Int32()
  external int grp_attn_n;

  @ffi.Int32()
  external int grp_attn_w;

  @ffi.Float()
  external double rope_freq_base;

  @ffi.Float()
  external double rope_freq_scale;

  @ffi.Float()
  external double yarn_ext_factor;

  @ffi.Float()
  external double yarn_attn_factor;

  @ffi.Float()
  external double yarn_beta_fast;

  @ffi.Float()
  external double yarn_beta_slow;

  @ffi.Int32()
  external int yarn_orig_ctx;

  @ffi.Int()
  external int rope_scaling_type;

  @ffi.Bool()
  external bool prompt_cache_all;

  @ffi.Bool()
  external bool ignore_eos;

  @ffi.Bool()
  external bool flash_attn;

  @ffi.Int32()
  external int top_k;

  @ffi.Float()
  external double top_p;

  @ffi.Float()
  external double min_p;

  @ffi.Float()
  external double tfs_z;

  @ffi.Float()
  external double typical_p;

  @ffi.Float()
  external double temp;

  @ffi.Float()
  external double dynatemp_range;

  @ffi.Float()
  external double dynatemp_exponent;

  @ffi.Int32()
  external int penalty_last_n;

  @ffi.Float()
  external double penalty_repeat;

  @ffi.Float()
  external double penalty_freq;

  @ffi.Float()
  external double penalty_present;

  @ffi.Int32()
  external int mirostat;

  @ffi.Float()
  external double mirostat_tau;

  @ffi.Float()
  external double mirostat_eta;

  @ffi.Bool()
  external bool penalize_nl;

  external ffi.Pointer<ffi.Char> grammar;
}

typedef wooly_token_update_callback
    = ffi.Pointer<ffi.NativeFunction<wooly_token_update_callbackFunction>>;
typedef wooly_token_update_callbackFunction = ffi.Bool Function(
    ffi.Pointer<ffi.Char> token_str);
typedef Dartwooly_token_update_callbackFunction = bool Function(
    ffi.Pointer<ffi.Char> token_str);
