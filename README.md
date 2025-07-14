# SubtransLLM

A command-line tool for translating SRT subtitle files using Large Language Models (LLMs). Supports multiple AI providers including OpenAI, Anthropic, and OpenRouter.

## Features

- **Multiple LLM Providers**: OpenAI, Anthropic, and OpenRouter support
- **Batch Translation**: Process multiple subtitle entries with context for better consistency
- **Custom Prompts**: Use predefined or custom prompts for different translation styles
- **Progress Tracking**: Real-time progress indicators during translation
- **Dry Run Mode**: Preview what will be translated before processing
- **Flexible Output**: Maintains original SRT formatting and timing

## Installation

### Using uv (recommended)

```bash
git clone <repository-url>
cd subtransLLM
uv sync
```

### Using pip

```bash
git clone <repository-url>
cd subtransLLM
pip install -r requirements.txt
```

## Setup

Set your API key as an environment variable:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# For OpenRouter
export OPENROUTER_API_KEY="your-api-key"
```

Alternatively, create a `.env` file:

```
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
OPENROUTER_API_KEY=your-api-key
```

## Usage

### Basic Translation

```bash
python subtrans.py input.srt output.srt --target-lang "Spanish"
```

### Specify Provider and Model

```bash
python subtrans.py input.srt output.srt \
  --target-lang "French" \
  --provider anthropic \
  --model claude-3-sonnet-20240229
```

### Batch Translation for Better Context

```bash
python subtrans.py input.srt output.srt \
  --target-lang "German" \
  --batch-mode \
  --batch-size 10
```

### Using Custom Prompts

```bash
python subtrans.py input.srt output.srt \
  --target-lang "Japanese" \
  --custom-prompt "Translate casually: {text}"
```

Or use a prompt file:

```bash
python subtrans.py input.srt output.srt \
  --target-lang "Korean" \
  --prompt-file custom_prompt.txt
```

### Preview Translation (Dry Run)

```bash
python subtrans.py input.srt output.srt \
  --target-lang "Italian" \
  --dry-run
```

## Command Options

| Option | Short | Description |
|--------|-------|-------------|
| `--target-lang` | `-t` | Target language (required) |
| `--source-lang` | `-s` | Source language (optional) |
| `--provider` | `-p` | LLM provider: `openai`, `anthropic`, `openrouter` |
| `--api-key` | `-k` | API key (overrides environment variable) |
| `--model` | `-m` | Specific model to use |
| `--dry-run` | | Preview without translating |
| `--quiet` | `-q` | Suppress progress output |
| `--custom-prompt` | | Custom prompt template |
| `--prompt-file` | | File containing custom prompt |
| `--batch-mode` | | Use batch translation for consistency |
| `--batch-size` | | Entries per batch (default: 5) |

## Custom Prompts

Custom prompts support placeholders:
- `{text}` - The subtitle text to translate
- `{target_language}` - Target language
- `{source_language}` - Source language

### Example Prompts

See `example_prompts.txt` for various prompt styles:
- Formal translation
- Casual/informal translation
- Movie/entertainment translation
- Technical/documentary translation
- Localized translation

## Supported Providers

### OpenAI
- Default model: `gpt-3.5-turbo`
- Supports all GPT models

### Anthropic
- Default model: `claude-3-haiku-20240307`
- Supports Claude 3 models

### OpenRouter
- Default model: `meta-llama/llama-3.1-8b-instruct:free`
- Access to various open-source models

## Examples

### Translate with specific style
```bash
python subtrans.py movie.srt movie_spanish.srt \
  --target-lang "Spanish" \
  --custom-prompt "Translate this movie dialogue to {target_language}. Keep the dramatic tone and make it sound natural for Spanish speakers: {text}"
```

### Process large files efficiently
```bash
python subtrans.py lecture.srt lecture_french.srt \
  --target-lang "French" \
  --batch-mode \
  --batch-size 15 \
  --quiet
```

## Requirements

- Python 3.12+
- OpenAI API key (for OpenAI provider)
- Anthropic API key (for Anthropic provider)
- OpenRouter API key (for OpenRouter provider)

## Dependencies

- `click` - Command-line interface
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `python-dotenv` - Environment variable management
- `requests` - HTTP requests

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.