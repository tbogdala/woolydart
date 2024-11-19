# Change log

## v0.3.0

- Sync'd changes with upstream woolycore.
- Added `processAdditionalPrompt` to add more prompt tokens before generating text.

## v0.2.0

- Added support for upstream woolycore's ability to step-by-step run inference. This includes methods like
    `processPrompt`, `sampleNextToken`, `checkEogAndAntiprompt`, `processNextToken`, `freeGptSampler`
- Added support for upstream woolycore's ability to cache the model's state. This includes methods like
    `freezePrompt`, `freezePromptWithPrediction`, `defrostFrozenState`, `freeFrozenState`
- Added a new String? property called `loadedModelFilepath` to `LlamaModel` to track the last loaded model file.

## v0.1.0

- Initial development version. Versioning hasn't started yet and all progress is on the `main` branch.
