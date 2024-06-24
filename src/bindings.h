#ifdef __cplusplus
#include <vector>
#include <string>

extern "C"
{
#endif

#include "llama.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#include <stdbool.h>

typedef struct load_model_result {
    struct llama_model* model;
    struct llama_context* ctx;
} load_model_result;

typedef struct wooly_predict_result {
    // 0 == success; 1 >= failure
    int result;

    // a pointer to llama_predict_prompt_cache, which is opaque to the bindings.
    void* prompt_cache;
    
    // timing data
    double t_start_ms;
    double t_end_ms;
    double t_load_ms;
    double t_sample_ms;
    double t_p_eval_ms;
    double t_eval_ms;

    int n_sample;
    int n_p_eval;
    int n_eval;
} wooly_predict_result;

typedef struct gpt_params_simple {
    /* select members of gpt_params */
    const char* prompt;
    const char ** antiprompts;
    int antiprompt_count;

    uint32_t seed;              // RNG seed
    int32_t n_threads;
    int32_t n_threads_batch;    // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_predict;          // new tokens to predict
    int32_t n_ctx;              // context size
    int32_t n_batch;            // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_gpu_layers;       // number of layers to store in VRAM (-1 - use default)
    uint32_t split_mode;// how to split the model across GPUs
    int32_t main_gpu;           // the GPU that is used for scratch and small tensors
    float   tensor_split[128];  // how split tensors should be distributed across GPUs
    int32_t grp_attn_n;         // group-attention factor
    int32_t grp_attn_w;         // group-attention width
    float   rope_freq_base;     // RoPE base frequency
    float   rope_freq_scale;    // RoPE frequency scaling factor
    float   yarn_ext_factor;    // YaRN extrapolation mix factor
    float   yarn_attn_factor;   // YaRN magnitude scaling factor
    float   yarn_beta_fast;     // YaRN low correction dim
    float   yarn_beta_slow;     // YaRN high correction dim
    int32_t yarn_orig_ctx;      // YaRN original context length
    int rope_scaling_type;

    bool prompt_cache_all;      // save user input and generations to prompt cache
    bool ignore_eos;            // ignore generated EOS tokens
    bool flash_attn;            // flash attention

    /* incorporate llama_sampling_params members too*/

    int32_t     top_k;                  // <= 0 to use vocab size
    float       top_p;                  // 1.0 = disabled
    float       min_p;                  // 0.0 = disabled
    float       tfs_z;                  // 1.0 = disabled
    float       typical_p;              // 1.0 = disabled
    float       temp;                   // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range;         // 0.0 = disabled
    float       dynatemp_exponent;      // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n;         // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat;         // 1.0 = disabled
    float       penalty_freq;           // 0.0 = disabled
    float       penalty_present;        // 0.0 = disabled
    int32_t     mirostat;               // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau;           // target entropy
    float       mirostat_eta;           // learning rate
    bool        penalize_nl;            // consider newlines as a repeatable token

    const char* grammar;

} gpt_params_simple;

// the token update callback for wooly_redict should return a bool indicating if prediction should continue (true),
// or if the prediction should stop (false).
typedef bool (*token_update_callback)(const char *token_str);

LLAMA_API load_model_result wooly_load_model(
    const char *fname, 
    struct llama_model_params model_params, 
    struct llama_context_params context_params,
    bool silent_llama);
        
LLAMA_API void wooly_free_model(struct llama_context *ctx_ptr, struct llama_model *model_ptr);

LLAMA_API gpt_params_simple wooly_new_params();

LLAMA_API wooly_predict_result wooly_predict(
    gpt_params_simple simple_params, struct llama_context *ctx_ptr, struct llama_model *model_ptr, bool include_specials, char *out_result, 
    void* prompt_cache_ptr, token_update_callback token_cb);    

// free the pointer returned in wooly_predict_result from llama_predict().
// only needed if you're not intending to use the prompt cache feature
LLAMA_API void wooly_free_prompt_cache(void *prompt_cache_ptr);

#ifdef __cplusplus
}

#endif