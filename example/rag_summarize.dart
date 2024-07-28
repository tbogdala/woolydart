import 'dart:ffi';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:args/args.dart';
import 'package:ffi/ffi.dart';
import 'package:format/format.dart';
import 'package:woolydart/woolydart.dart';

typedef ParseHtmlFunction = Pointer<Utf8> Function(Pointer<Utf8>);

String getPlatformLibraryFilepath() {
  return (Platform.isMacOS) ? "src/build/libwoolycore.dylib" : "src/build/libwoolycore.so";
}

String getReadabilityLibraryFilepath() {
  return (Platform.isMacOS) ? 'src/libreadability/target/release/libreadability.dylib' : 'src/libreadability/target/release/libreadability.so';
}

String cleanUpWithReadable(String html) {
  // Load the dynamic library - this will need to be compiled by hand first and requires
  // a rust toolchain!
  final File libreadablePath = File(getReadabilityLibraryFilepath());
  if (!libreadablePath.existsSync()) {
    print('\nWARNING!');
    print('\nThis sample uses a Rust library to clean up the HTML being sent');
    print(
        'for summarization and the library will need to be built before hand.');
    print('This will require a Rust toolchain.');
    print('');
    print('To build with the installed Rust compiler, execute in a shell:');
    print('');
    print('cd src/libreadability;cargo build --release');
    print('');
    print('Since this was not detected, the raw HTML will proceed unfiltered');
    print('to the summarizer and may need more context size,time to process,');
    print('or simply fail to run at all.\n');
    return html;
  }

  final dylib = DynamicLibrary.open(libreadablePath.path);

  // Look up the `parse_html` function
  final parseHtml =
      dylib.lookupFunction<ParseHtmlFunction, ParseHtmlFunction>('parse_html');

  // Call the `parse_html` function with a sample input string
  final inputPtr = html.toNativeUtf8().cast<Utf8>();
  final outputPtr = parseHtml(inputPtr);
  final output = outputPtr.toDartString();

  // NOTE: our FFI interface to free the memory allocated by the `parse_html` function
  // was failing to compile so I just disabled it for now. Won't matter since
  // this sample is just a one-shot, but it's something to consider for longer
  // lived applications.
  //freeParsedHtml(outputPtr);
  malloc.free(inputPtr);

  return output;
}

void main(List<String> args) async {
  ArgResults parsedArgs = _parseArgs(args);

  // start off by getting the content at the URL specified, or a random wikipedia page
  String pageContent;
  final urlToPull = parsedArgs.option('url') ??
      'https://en.wikipedia.org/wiki/Special:Random';
  try {
    final url = Uri.parse(urlToPull);
    final response = await http.get(url);
    if (response.statusCode == 200) {
      pageContent = response.body;
    } else {
      print('Unable to retrieve the Internet content from: $urlToPull\n');
      print('Status code of request: ${response.statusCode}');
      exit(1);
    }
  } catch (e) {
    print('Unable to retrieve the Internet content from: $urlToPull\n');
    print('Error: $e');
    exit(1);
  }

  // try to remove unnecessary HTML from the string using our C library.
  final content = cleanUpWithReadable(pageContent);

  // load the library up for ffi work; the actual filepath
  // depends on the operating system's perference for libraries.
  final libFilepath = getPlatformLibraryFilepath();
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
  // by the GGUF model; for models with large context, this may have
  // very large memory requirements.
  final contextParams = llamaModel.getDefaultContextParams()
    ..seed = 42
    ..n_ctx = (parsedArgs['contextsize'] != null)
        ? int.parse(parsedArgs.option('contextsize')!)
        : 1024 * 4;

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
    print(
        'Additionally, try running this example without the "quiet" flag for more information.');
  }

  // now we build _another_ structure, this one a simplified form of gpt_params
  // from the upstream llama.cpp library. it will specify more parameters for
  // controlling how the text is generated and sampled.
  //
  // note that we attempt to predict 200 new tokens, set the system to use 4 threads,
  // hardcode our seed again for testing, use flash attention where possible,
  // use a batch size of 128 for prompt processing and then set the hyperparameters
  // for sampling.
  final params = llamaModel.getTextGenParams()
    ..seed = 42
    ..n_threads = 6
    ..n_predict = -1
    ..top_k = 30
    ..top_p = 1.0
    ..min_p = 0.075
    ..penalty_repeat = 1.05
    ..penalty_last_n = 512
    ..ignore_eos = false
    ..flash_attn = true
    ..n_batch = 256
    ..prompt_cache_all = false;

  // build the prompt towards phi3's prompt format
  final system = "You are a helpful AI assistant.";
  final preSystemPrefix = "<|system|>\n";
  final preSystemSuffix = "<|end|>\n";
  final userPrefix = "<|user|>\n";
  final userSuffix = "<|end|>\n";
  final aiPrefix = "<|assistant|>\n";
  final stopPhrases = ["<|end|>", "<|user|>"];
  final promtpDirection =
      "Summarize the following web page into a short paragraph of text that can be easily digested by the reader:\n\n$content";
  final builtPrompt =
      '$preSystemPrefix$system$preSystemSuffix$userPrefix$promtpDirection$userSuffix$aiPrefix';

  // on the params object, we call two specilized functions to set the prompt and
  // antiprompt (phrases that, when detected, will stop text generation). this is
  // because the String values have to be processed before being used by the
  // llama.cpp library, and it is more convenient to hide that away. if client
  // code needed to, it could set `params.prompt`, `params.antiprompts` and
  // `params.antiprompt_count` manually.
  params.setPrompt(builtPrompt);
  params.setAntiprompts(stopPhrases);

  // now we actually run the text prediction using the parameters defined above.
  // the function returns a structure that has the overall success indicator
  // as well as timing information, as well as returning the full predicted output.
  final (predictResult, outputString) = llamaModel.predictText(params, nullptr);
  if (predictResult.result != 0) {
    print('Error: LlamaModel.predictText() returned ${predictResult.result}');
    exit(1);
  }

  print(outputString);

  // we could print out the predicted text as a whole string, but that's already
  // been done, piece by piece, with the `onNewToken()` callback defined below.
  // print('\n${outputString?.trim()}\n');
  final int totalCharacters = outputString!.length;

  // print out some stats from timing data returned by the text generation func.
  if (!parsedArgs.flag('quiet')) {
    print(format(
        '\nPerformance data: {} tokens ({} characters) total in {:.2f} ms ({:.2f} T/s) ; {} prompt tokens in {:.2f} ms ({:.2f} T/s)\n\n',
        predictResult.n_eval,
        totalCharacters,
        (predictResult.t_end_ms - predictResult.t_start_ms),
        1e3 /
            (predictResult.t_end_ms - predictResult.t_start_ms) *
            predictResult.n_eval,
        predictResult.n_p_eval,
        predictResult.t_p_eval_ms,
        1e3 / predictResult.t_p_eval_ms * predictResult.n_p_eval));
  }

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
  parser.addOption('url', help: 'The URL to retrieve and summarize');
  parser.addOption('contextsize',
      abbr: 'c',
      help: 'The size of the context for the LLM in tokens (default: 4096).');
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
