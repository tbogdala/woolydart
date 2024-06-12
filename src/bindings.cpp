#include "llama.h"
#include "common/common.h"

#include "bindings.h"

#include <regex>
#include <fstream>
#include <string>
#include <vector>

typedef struct llama_predict_prompt_cache {
    std::string last_prompt;
    std::vector<llama_token> processed_prompt_tokens;
    uint8_t * last_processed_prompt_state;
} llama_predict_prompt_cache;

static bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string &path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

static std::vector<std::string> create_vector(const char **strings, int count)
{
    std::vector<std::string> *vec = new std::vector<std::string>;
    for (int i = 0; i < count; i++)
    {
        vec->push_back(std::string(strings[i]));
    }
    return *vec;
}

static void delete_vector(std::vector<std::string> *vec)
{
    delete vec;
}

// this is a dupe of the `llama_token_to_piece` function from `common` but with a parameter
// to control the printing of special tokens.
static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token, bool include_specials) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), include_specials);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), include_specials);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}



load_model_result wooly_load_model(
    const char *fname, int n_ctx, int n_seed, bool mlock, bool mmap, bool embeddings, int n_gpu_layers, 
    int n_batch, int maingpu, const char *tensorsplit, float rope_freq, float rope_scale)
{
    // load the model
    auto lparams = llama_context_default_params();
    auto mparams = llama_model_default_params();

    lparams.n_ctx = n_ctx;
    lparams.n_batch = n_batch;
    lparams.seed = n_seed;
    lparams.embeddings = embeddings;
    lparams.rope_freq_base = rope_freq;
    lparams.rope_freq_scale = rope_scale;
    mparams.use_mlock = mlock;
    mparams.use_mmap = mmap;
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.main_gpu = maingpu;

    if (tensorsplit != NULL && strlen(tensorsplit) > 0)
    {
        std::string arg_next = tensorsplit;
        // split string by , and /
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};
        auto max_devices = llama_max_devices();
        GGML_ASSERT(split_arg.size() <= max_devices);

	    float *tsplit = (float*)malloc(sizeof(float) * max_devices);

        for (size_t i = 0; i < max_devices; ++i)
        {
            if (i < split_arg.size())
            {
                tsplit[i] = std::stof(split_arg[i]);
            }
            else
            {
                tsplit[i] = 0.0f;
            }
        }
	    mparams.tensor_split = tsplit;
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    llama_model *model = nullptr;
    llama_context * lctx = nullptr;
    load_model_result res;
    res.ctx = nullptr;
    res.model = nullptr;
    // TODO: implement lora adapters (e.g. llama_init_from_gpt_params())
    try
    {
        model = llama_load_model_from_file(fname, mparams);
	    lctx = llama_new_context_with_model(model, lparams);
    }
    catch (std::runtime_error &e)
    {
        LOG("failed %s", e.what());
        llama_free(lctx);
        llama_free_model(model);
        return res;
    }

    {
        LOG("warming up the model with an empty run\n");

        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) n_batch), 0, 0));
        llama_kv_cache_clear(lctx);
        llama_reset_timings(lctx);
    }

    res.ctx = lctx;
    res.model = model;
    return res;
}

void wooly_free_model(void *ctx_ptr, void *model_ptr)
{
    llama_context *ctx = (llama_context *)ctx_ptr;
    llama_model *model = (llama_model *)model_ptr;
    llama_free_model(model);
    llama_free(ctx);
}

wooly_predict_result wooly_predict(
    void *params_ptr, void *ctx_ptr, void *model_ptr, bool include_specials, char *out_result, 
    void* prompt_cache_ptr) 
{
    llama_context *ctx = (llama_context *)ctx_ptr;
    llama_model *model = (llama_model *)model_ptr;
    llama_context *ctx_guidance = nullptr;
    gpt_params *params_p = (gpt_params *)params_ptr;
    llama_sampling_params &sparams = params_p->sparams;
    llama_predict_prompt_cache *prompt_cache_data = (llama_predict_prompt_cache *) prompt_cache_ptr;
    wooly_predict_result return_value;
    return_value.n_eval = return_value.n_p_eval = return_value.n_sample = 0;
    
    llama_set_n_threads(ctx, params_p->n_threads, params_p->n_threads_batch);
    llama_kv_cache_clear(ctx);
    llama_reset_timings(ctx);
        
    // print system information
    {
        // TODO: Update the build.rs file to generate llama.cpp/common/build-info.cpp
        // LOG("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
        // LOG("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);
        LOG("\n");
        LOG("%s\n", gpt_params_get_system_info(*params_p).c_str());
    }

    if (params_p->ignore_eos) {
        LOG("%s: Ignoring EOS token by setting bias to -INFINITY\n", __func__);
        sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    if (params_p->rope_freq_base != 0.0) {
        LOG("%s: warning: changing RoPE frequency base to %g.\n", __func__, params_p->rope_freq_base);
    }

    if (params_p->rope_freq_scale != 0.0) {
        LOG("%s: warning: scaling RoPE frequency by %g.\n", __func__, params_p->rope_freq_scale);
    }

    if (sparams.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(*params_p);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    LOG("%s: input: %s\n", __func__, params_p->prompt.c_str());

    if (n_ctx > n_ctx_train) {
        LOG("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    bool resuse_last_prompt_data = false;
    if (prompt_cache_data != nullptr && params_p->prompt_cache_all) {
        // check to see if we're repeating the same prompt and reuse the stored prompt data if so.
        // if it's not a match, clear out the cached tokens.
        if (prompt_cache_data->last_prompt == params_p->prompt) {
            LOG("Prompt match detected. Going to attempt to use last processed prompt token data and state.\n");
            resuse_last_prompt_data = true;
            llama_set_state_data(ctx, prompt_cache_data->last_processed_prompt_state);
        } else {
            // new prompt detected, so free the memory of the cached state
            if (prompt_cache_data->last_processed_prompt_state != nullptr) {
                delete[] prompt_cache_data->last_processed_prompt_state;
                prompt_cache_data->last_processed_prompt_state = nullptr;
            }
            prompt_cache_data->processed_prompt_tokens.clear();
        }
    } else {
        // if we don't have a prompt cache object, create one
        prompt_cache_data = new llama_predict_prompt_cache;
        prompt_cache_data->last_processed_prompt_state = nullptr;
    }
    // also copy the pointer of the prompt_cache_data to the result here now that it's for sure allocated
    return_value.prompt_cache = prompt_cache_data;

    if (params_p->seed <= 0) {
        params_p->seed = time(NULL);
        LOG("%s: Seed == 0 so a new one was generated: %d\n", __func__, params_p->seed);    
    }

    std::string path_session = params_p->path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return_value.result = 1;
                return return_value;
            }
            session_tokens.resize(n_token_count_out);
            LOG("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_should_add_bos_token(model);
    GGML_ASSERT(llama_add_eos_token(model) != 1);
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;
    if (!resuse_last_prompt_data) {
        if (!params_p->prompt.empty() || session_tokens.empty()) {
            LOG("tokenize the prompt\n");
            embd_inp = ::llama_tokenize(ctx, params_p->prompt, add_bos, true);
        } else {
            LOG("use session tokens\n");
            embd_inp = session_tokens;
        }
    }

    LOG("prompt: \"%s\"\n", log_tostr(params_p->prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }

    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    if (ctx_guidance) {
        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

        guidance_inp = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, true, true);
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp).c_str());

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params_p->prompt, true, true);
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp).c_str());

        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
        LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return_value.result = 2;
        return return_value;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params_p->prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG("%s: using full prompt from session file\n", __func__);
        }
        else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG("%s: session file has exact match for prompt!\n", __func__);
        }
        else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    LOGLN(
        "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
        log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params_p->n_keep < 0 || params_p->n_keep > (int)embd_inp.size()) {
        params_p->n_keep = (int)embd_inp.size();
    } else {
        params_p->n_keep += add_bos; // always keep the BOS token
    }

    {
        LOG("\n");
        LOG("%s: prompt: '%s'\n", __func__, params_p->prompt.c_str());
        LOG("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (ctx_guidance) {
            LOG("\n");
            LOG("%s: negative prompt: '%s'\n", __func__, sparams.cfg_negative_prompt.c_str());
            LOG("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int) guidance_inp.size(); i++) {
                LOG("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
            }
        }

       if (params_p->n_keep > add_bos) {
            LOG("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params_p->n_keep; i++) {
                LOG("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG("'\n");
        }
        LOG("\n");
    }
    LOG("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params_p->n_batch, params_p->n_predict, params_p->n_keep);

    LOG("\n\n");

    bool is_antiprompt        = false;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params_p->n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve(params_p->antiprompt.size());
    for (const std::string & antiprompt : params_p->antiprompt) {
        antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
    }

    struct llama_sampling_context * ctx_sampling =  llama_sampling_init(sparams);

    // our result to send back to Rust
    std::string res = "";
    bool need_to_save_state = true;

    // if we're reusing the prompt, clear out any input tokens to be processed
    // and set the tracking counter to the length of the saved prompt
    if (resuse_last_prompt_data) {
        embd_inp.clear();
        n_past = prompt_cache_data->processed_prompt_tokens.size();
        LOG("%s: reusing prompt tokens; initializing n_consumed to %d\n",  __func__, n_consumed);
    }

    while (n_remain != 0 && !is_antiprompt) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);
                LOG("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int)session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int)session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            if (ctx_guidance) {
                int input_size = 0;
                llama_token * input_buf = NULL;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end()) {
                        embd_guidance.insert(
                            embd_guidance.end(),
                            embd.begin() + original_prompt_len,
                            embd.end()
                        );
                    }

                    input_buf  = embd_guidance.data();
                    input_size = embd_guidance.size();

                    LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance).c_str());
                } else {
                    input_buf  = embd.data();
                    input_size = embd.size();
                }

                for (int i = 0; i < input_size; i += params_p->n_batch) {
                    int n_eval = std::min(input_size - i, params_p->n_batch);
                    if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
                        LOG("%s : failed to eval\n", __func__);
                        return_value.result = 3;
                        return return_value;
                    }

                    n_past_guidance += n_eval;
                }
            }

            for (int i = 0; i < (int)embd.size(); i += params_p->n_batch) {
                int n_eval = (int)embd.size() - i;
                if (n_eval > params_p->n_batch) {
                    n_eval = params_p->n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    LOG("%s : failed to eval\n", __func__);
                    return_value.result = 4;
                    const llama_timings timings = llama_get_timings(ctx);
                    return_value.n_sample = timings.n_sample;
                    return_value.n_p_eval = timings.n_p_eval;
                    return_value.n_eval = timings.n_eval;
                    return return_value;
                }

                n_past += n_eval;
                LOG("n_past = %d\n", n_past);
                LOG("\nTokens consumed so far = %d / %d\n", n_past, n_ctx);
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            if (need_to_save_session == true) {
                need_to_save_session = false;

                // optionally save the session on first sample (for faster prompt loading next time)
                if (!path_session.empty() && need_to_save_session && !params_p->prompt_cache_ro) {
                    llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
                    LOG("saved session to %s\n", path_session.c_str());
                }
            } 

            if (params_p->prompt_cache_all == true && need_to_save_state == true && resuse_last_prompt_data == false) {
                LOG("saving last used prompt data.\n");
                need_to_save_state = false;
                if (prompt_cache_data->last_processed_prompt_state != nullptr) {
                    delete[] prompt_cache_data->last_processed_prompt_state;
                }
                const size_t state_size = llama_get_state_size(ctx);
                prompt_cache_data->last_processed_prompt_state = new uint8_t[state_size];
                llama_copy_state_data(ctx, prompt_cache_data->last_processed_prompt_state);
                prompt_cache_data->last_prompt = params_p->prompt;
                LOG("Adding to the processed_prompt_tokens vector %d tokens from embd_inp.\n", (int)embd_inp.size());
                prompt_cache_data->processed_prompt_tokens.insert(prompt_cache_data->processed_prompt_tokens.end(), embd_inp.begin(),embd_inp.end());
            }

            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);

            // FIXME: callback support
            // call the token callback on the Rust side
            // auto token_str = llama_token_to_str(ctx, id, include_specials);
            // if (!tokenCallback(ctx_ptr, token_str.c_str())) {
            //     break;
            // }

            for (auto id : embd) {
                res += llama_token_to_str(ctx, id, include_specials);
            }
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params_p->n_batch) {
                    break;
                }
            }
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params_p->antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params_p->antiprompt) {
                    size_t extra_padding = 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = llama_sampling_last(ctx_sampling);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }
        }

        // end of generation
        if (!embd.empty() && llama_token_is_eog(model, embd.back())) {
            LOG(" [end of text]\n");
            break;
        }
    }

    if (!path_session.empty() && !params_p->prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    // build up the result structure with the success code and all the timing data
    const llama_timings timings = llama_get_timings(ctx);
    return_value.result = 0;
    return_value.t_start_ms = timings.t_start_ms;
    return_value.t_end_ms = timings.t_end_ms;
    return_value.t_load_ms = timings.t_load_ms;
    return_value.t_sample_ms = timings.t_sample_ms;
    return_value.t_p_eval_ms = timings.t_p_eval_ms;
    return_value.t_eval_ms = timings.t_eval_ms;
    return_value.n_sample = timings.n_sample;
    return_value.n_p_eval = timings.n_p_eval;
    return_value.n_eval = timings.n_eval;

    strcpy(out_result, res.c_str());

    if (ctx_guidance) { llama_free(ctx_guidance); }
    llama_sampling_free(ctx_sampling);
    
    LOG("Log end\n");

    return return_value;
}

void wooly_free_prompt_cache(void *prompt_cache_ptr)
{
    if (prompt_cache_ptr != nullptr) {
        llama_predict_prompt_cache *prompt_cache_data = (llama_predict_prompt_cache *) prompt_cache_ptr;
        delete[] prompt_cache_data->last_processed_prompt_state;
        delete prompt_cache_data;
    }
}

void wooly_free_params(void *params_ptr)
{
    gpt_params *params = (gpt_params *)params_ptr;
    delete params;
}

void *wooly_allocate_params(
    const char *prompt, int seed, int threads, int tokens, int top_k, float top_p, float min_p, 
    float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, int n_batch, int n_keep, 
    const char **antiprompt, int antiprompt_count, float tfs_z, float typical_p, float frequency_penalty, 
    float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, 
    const char *logit_bias, const char *session_file, bool prompt_cache_in_memory, bool mlock, bool mmap, 
    int maingpu, const char *tensorsplit, bool file_prompt_cache_ro, float rope_freq_base, 
    float rope_freq_scale, const char *grammar)
{
    gpt_params *params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_threads_batch = threads;
    params->n_predict = tokens;
    params->sparams.penalty_last_n = repeat_last_n;
    params->sparams.top_k = top_k;
    params->sparams.top_p = top_p;
    params->sparams.min_p = min_p;
    params->sparams.temp = temp;
    params->use_mmap = mmap;
    params->use_mlock = mlock;
    params->sparams.penalty_repeat = repeat_penalty;
    params->n_batch = n_batch;
    params->n_keep = n_keep;
    params->rope_freq_base = rope_freq_base;
    params->rope_freq_scale = rope_freq_scale;
    params->ignore_eos = ignore_eos;
    params->main_gpu = maingpu;

    if (tensorsplit[0] != '\0')
    {
        std::string arg_next = tensorsplit;
        // split string by , and /
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};
        auto max_devices = llama_max_devices();
        GGML_ASSERT(split_arg.size() <= max_devices);

        for (size_t i = 0; i < max_devices; ++i)
        {
            if (i < split_arg.size())
            {
                params->tensor_split[i] = std::stof(split_arg[i]);
            }
            else
            {
                params->tensor_split[i] = 0.0f;
            }
        }
    }

    params->prompt_cache_all = prompt_cache_in_memory;
    params->path_prompt_cache = session_file;
    params->prompt_cache_ro = file_prompt_cache_ro;

    if (antiprompt_count > 0)
    {
        params->antiprompt = create_vector(antiprompt, antiprompt_count);
    }
    params->sparams.tfs_z = tfs_z;
    params->sparams.typical_p = typical_p;
    params->sparams.penalty_present = presence_penalty;
    params->sparams.mirostat = mirostat;
    params->sparams.mirostat_eta = mirostat_eta;
    params->sparams.mirostat_tau = mirostat_tau;
    params->sparams.penalize_nl = penalize_nl;
    params->sparams.grammar = grammar;
    std::stringstream ss(logit_bias);
    llama_token key;
    char sign;
    std::string value_str;
    if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-'))
    {
        params->sparams.logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
    }
    params->sparams.penalty_freq = frequency_penalty;
    params->prompt = prompt;

    return params;
}