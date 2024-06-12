#ifdef __cplusplus
#include <vector>
#include <string>
extern "C"
{
#endif


#include <stdbool.h>

typedef struct load_model_result {
    void* model;
    void* ctx;
} load_model_result;

typedef struct wooly_predict_result {
    // 0 == success; 1 == failure
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


LLAMA_API load_model_result wooly_load_model(
    const char *fname, int n_ctx, int n_seed, bool mlock, bool mmap, bool embeddings, int n_gpu_layers, 
    int n_batch, int maingpu, const char *tensorsplit, float rope_freq, float rope_scale);
    
LLAMA_API void wooly_free_model(void *ctx_ptr, void *model_ptr);


LLAMA_API void *wooly_allocate_params(
    const char *prompt, int seed, int threads, int tokens, int top_k, float top_p, float min_p, 
    float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, int n_batch, int n_keep, 
    const char **antiprompt, int antiprompt_count, float tfs_z, float typical_p, float frequency_penalty, 
    float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, 
    const char *logit_bias, const char *session_file, bool prompt_cache_in_memory, bool mlock, bool mmap, 
    int maingpu, const char *tensorsplit, bool file_prompt_cache_ro, float rope_freq_base, 
    float rope_freq_scale, const char *grammar);

LLAMA_API void wooly_free_params(void *params_ptr);


LLAMA_API wooly_predict_result wooly_predict(
    void *params_ptr, void *ctx_ptr, void *model_ptr, bool include_specials, char *out_result, 
    void* prompt_cache_ptr);    

// free the pointer returned in wooly_predict_result from llama_predict().
// only needed if you're not intending to use the prompt cache feature
LLAMA_API void wooly_free_prompt_cache(void *prompt_cache_ptr);


#ifdef __cplusplus
}

#endif