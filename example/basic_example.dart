import 'dart:io';

import 'package:args/args.dart';
import 'package:format/format.dart';
import 'package:woolydart/woolydart.dart';

void main(List<String> args) {
  ArgResults parsedArgs = _parseArgs(args);

  // load the library up for ffi work; the actual filepath
  // depends on the operating system's perference for libraries.
  const libFilepath = "src/llama.cpp/build/libllama.dylib";
  var llamaModel = LlamaModel(libFilepath);

  // setup the model parameters which has options to control
  // how the model file itself is loaded. this is a llama.cpp
  // structure, and the defaults come from the upstream library.
  //
  // right now, we set the `n_gpu_layers` member to 100, allowing it
  // too 'offload' up to 100 layers to the GPU for faster processing.
  final modelParams = llamaModel.getDefaultModelParams();
  modelParams.n_gpu_layers = 100;

  // setup the context parameters which has options to control
  // how the model behaves under text inference. this is a llama.cpp
  // structure, and the defaults come from the upstream library.
  //
  // Particularly important here is setting the seed to -1 for random
  // or to a particular value if consistency is needed as well as
  // setting the size of the context. Setting `n_ctx`, the size of the context,
  // to 0 is shorthand to let llama.cpp set it to the maximum size supported
  // by the GGUF model.
  final contextParams = llamaModel.getDefaultContextParams()
    ..seed = 42
    ..n_ctx = 0;

  // get the model filepath to load for text inference from the command-line
  String modelFilepath = parsedArgs.option('model')!;

  // now actually try to load the model; the returned value will indicate
  // if the loading was successful or not.
  final bool loadedResult = llamaModel.loadModel(
      modelFilepath, modelParams, contextParams, parsedArgs.flag('quiet'));
  if (!loadedResult) {
    print('\nFailed to load the model successfully.');
    print(
        'Ensure you selected the right GGUF file and that you have the options set correctly.');
  }

  // now we build _another_ structure, this one a simplified form of gpt_params
  // from the upstream llama.cpp library. it will specify more parameters for
  // controlling how the text is generated and sampled.
  //
  // note that we attempt to predict 100 new tokens, set the system to use 4 threads,
  // hardcode our seed again for testing, use flash attention where possible,
  // use a batch size of 128 for prompt processing and then set the hyperparameters
  // for sampling.
  final params = llamaModel.getTextGenParams()
    ..seed = 42
    ..n_threads = 4
    ..n_predict = 100
    ..top_k = 1
    ..top_p = 1.0
    ..min_p = 0.1
    ..min_p = 0.1
    ..penalty_repeat = 1.1
    ..penalty_last_n = 512
    ..ignore_eos = false
    ..flash_attn = true
    ..n_batch = 128
    ..prompt_cache_all = false;

  // on the params object, we call two specilized functions to set the prompt and
  // antiprompt (phrases that, when detected, will stop text generation). this is
  // because the String values have to be processed before being used by the
  // llama.cpp library, and it is more convenient to hide that away. if client
  // code needed to, it could set `params.prompt`, `params.antiprompts` and
  // `params.antiprompt_count` manually.
  params.setPrompt(parsedArgs.option('prompt') ??
      "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n");
  params.setAntiprompts([
    "<|end|>",
  ]);

  // now we actually run the text prediction using the parameters defined above.
  // the function returns a structure that has the overall success indicator
  // as well as timing information, as well as returning the full predicted output.
  final (predictResult, outputString) = llamaModel.predictText(params);
  if (predictResult.result != 0) {
    print('Error: LlamaModel.predictText() returned ${predictResult.result}');
    exit(1);
  }

  // print out the predicted text
  print('\n${outputString?.trim()}\n');

  // print out some stats from diming data returned by the text generation func.
  print(format(
      '\nPerformance data: {} tokens total in {:.2f} ms ({:.2f} T/s) ; {} prompt tokens in {:.2f} ms ({:.2f} T/s)\n\n',
      predictResult.n_eval,
      (predictResult.t_end_ms - predictResult.t_start_ms),
      1e3 /
          (predictResult.t_end_ms - predictResult.t_start_ms) *
          predictResult.n_eval,
      predictResult.n_p_eval,
      predictResult.t_p_eval_ms,
      1e3 / predictResult.t_p_eval_ms * predictResult.n_p_eval));

  // we'll be a good citizen and clean up after ourselves. params needs to
  // be disposed to free the memory allocated by prompts and antiprompts.
  // freeModel() will unload the model and release the memory it holds.
  params.dispose();
  llamaModel.freeModel();
}

ArgParser _buildArgParser() {
  var parser = ArgParser();
  parser.addOption('model',
      abbr: 'm',
      mandatory: true,
      help: 'GGUF model file to user for text generation');
  parser.addOption('prompt',
      abbr: 'p', help: 'A custom prompt to use for text generation');
  parser.addFlag('quiet',
      defaultsTo: false,
      abbr: 'q',
      help: 'Silences the llama.cpp library output');
  parser.addFlag('help',
      negatable: false,
      help: 'Show the full list of supported command-line arguments');
  return parser;
}

ArgResults _parseArgs(List<String> args) {
  ArgResults argResults;
  ArgParser argParser = _buildArgParser();

  try {
    argResults = argParser.parse(args);

    if (!argResults.wasParsed('model')) {
      print(
          '\nError: user must supply the "--model" option on the command line.\n');
      print(argParser.usage);
      exit(1);
    }
  } catch (e) {
    print('\nError: $e\n\nSupported arguments:');
    print(argParser.usage);
    exit(1);
  }
  if (argResults.wasParsed('help')) {
    print('\nSupported arguments:');
    print(argParser.usage);
    exit(0);
  }

  return argResults;
}
