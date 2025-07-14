#!/usr/bin/env python3

import click
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import os
from dotenv import load_dotenv

load_dotenv()

class SubtitleEntry:
    def __init__(self, index: int, start_time: str, end_time: str, text: str):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text.strip()

class SRTParser:
    @staticmethod
    def parse(content: str) -> List[SubtitleEntry]:
        entries = []
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
                
            try:
                index = int(lines[0])
                time_line = lines[1]
                text_lines = lines[2:]
                
                if ' --> ' not in time_line:
                    continue
                    
                start_time, end_time = time_line.split(' --> ')
                text = '\n'.join(text_lines)
                
                entries.append(SubtitleEntry(index, start_time, end_time, text))
            except (ValueError, IndexError):
                continue
                
        return entries

    @staticmethod
    def write(entries: List[SubtitleEntry], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry.index}\n")
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{entry.text}\n\n")

class LLMTranslator:
    def __init__(self, provider: str, api_key: str, model: str = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        
        if self.provider == 'openai':
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or 'gpt-3.5-turbo'
        elif self.provider == 'openrouter':
            import openai
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            self.model = model or 'meta-llama/llama-3.1-8b-instruct:free'
        elif self.provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model or 'claude-3-haiku-20240307'
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def translate(self, text: str, target_language: str, source_language: str = None, custom_prompt: str = None) -> str:
        if custom_prompt:
            prompt = custom_prompt.format(
                text=text,
                target_language=target_language,
                source_language=source_language or "auto-detect"
            )
        else:
            source_info = f" from {source_language}" if source_language else ""
            prompt = f"Translate the following subtitle text{source_info} to {target_language}. Preserve the meaning and natural flow for subtitles. Return only the translation without any additional text:\n\n{text}"
        
        try:
            if self.provider in ['openai', 'openrouter']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")

    def translate_batch_context(self, texts: List[str], target_language: str, source_language: str = None, custom_prompt: str = None, batch_size: int = 5) -> List[str]:
        if custom_prompt:
            batch_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
            prompt = custom_prompt.format(
                text=batch_text,
                target_language=target_language,
                source_language=source_language or "auto-detect"
            )
        else:
            source_info = f" from {source_language}" if source_language else ""
            batch_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
            prompt = f"""Translate the following numbered subtitle entries{source_info} to {target_language}. Maintain context and consistency across all entries. Return only the translations in the same numbered format:

{batch_text}"""
        
        try:
            if self.provider in ['openai', 'openrouter']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text.strip()
            
            translated_lines = []
            for line in result.split('\n'):
                line = line.strip()
                if line and '. ' in line:
                    translated_lines.append(line.split('. ', 1)[1])
                elif line and not line[0].isdigit():
                    translated_lines.append(line)
            
            if len(translated_lines) != len(texts):
                raise Exception(f"Expected {len(texts)} translations, got {len(translated_lines)}")
            
            return translated_lines
            
        except Exception as e:
            raise Exception(f"Batch translation failed: {str(e)}")

    def translate_batch(self, texts: List[str], target_language: str, source_language: str = None, quiet: bool = False, custom_prompt: str = None, batch_mode: bool = False, batch_size: int = 5) -> List[str]:
        if batch_mode:
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_start = i + 1
                batch_end = min(i + batch_size, len(texts))
                
                try:
                    if not quiet:
                        click.echo(f"[{batch_start}-{batch_end}/{len(texts)}] Translating batch...")
                    elif quiet:
                        click.echo(f"Progress: {batch_end}/{len(texts)} entries")
                    
                    batch_results = self.translate_batch_context(batch, target_language, source_language, custom_prompt, batch_size)
                    results.extend(batch_results)
                    
                    if not quiet:
                        click.echo(f"[{batch_start}-{batch_end}/{len(texts)}] Batch completed")
                        
                except Exception as e:
                    if not quiet:
                        click.echo(f"Batch translation failed for entries {batch_start}-{batch_end}: {str(e)}", err=True)
                        click.echo("Falling back to individual translation...", err=True)
                    
                    for j, text in enumerate(batch):
                        try:
                            translated = self.translate(text, target_language, source_language, custom_prompt)
                            results.append(translated)
                            if not quiet:
                                click.echo(f"[{i+j+1}/{len(texts)}] Individual fallback: {text[:50]}..." if len(text) > 50 else f"[{i+j+1}/{len(texts)}] Individual fallback: {text}")
                        except Exception as e2:
                            if not quiet:
                                click.echo(f"Error translating '{text[:50]}...': {str(e2)}", err=True)
                            results.append(text)
            
            return results
        else:
            results = []
            progress_interval = max(1, len(texts) // 10)  # Show progress every 10%
            for i, text in enumerate(texts, 1):
                try:
                    translated = self.translate(text, target_language, source_language, custom_prompt)
                    results.append(translated)
                    if not quiet:
                        click.echo(f"[{i}/{len(texts)}] Translated: {text[:50]}..." if len(text) > 50 else f"[{i}/{len(texts)}] Translated: {text}")
                    elif quiet and (i % progress_interval == 0 or i == len(texts)):
                        click.echo(f"Progress: {i}/{len(texts)} entries")
                except Exception as e:
                    if not quiet:
                        click.echo(f"Error translating '{text[:50]}...': {str(e)}", err=True)
                    results.append(text)
            return results

@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--target-lang', '-t', required=True, help='Target language for translation')
@click.option('--source-lang', '-s', help='Source language (optional)')
@click.option('--provider', '-p', default='openai', type=click.Choice(['openai', 'openrouter', 'anthropic']), help='LLM provider')
@click.option('--api-key', '-k', help='API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
@click.option('--model', '-m', help='Model to use (optional)')
@click.option('--dry-run', is_flag=True, help='Show what would be translated without actually translating')
@click.option('--quiet', '-q', is_flag=True, help='Suppress translation progress output')
@click.option('--custom-prompt', help='Custom prompt template (use {text}, {target_language}, {source_language} placeholders)')
@click.option('--prompt-file', type=click.Path(exists=True, path_type=Path), help='File containing custom prompt template')
@click.option('--batch-mode', is_flag=True, help='Use batch translation for better context and consistency')
@click.option('--batch-size', default=5, type=int, help='Number of entries per batch (default: 5)')
def translate_subtitles(input_file, output_file, target_lang, source_lang, provider, api_key, model, dry_run, quiet, custom_prompt, prompt_file, batch_mode, batch_size):
    """Translate SRT subtitle files using LLM APIs."""
    
    if custom_prompt and prompt_file:
        click.echo("Error: Cannot specify both --custom-prompt and --prompt-file.", err=True)
        sys.exit(1)
    
    if prompt_file:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                custom_prompt = f.read().strip()
        except Exception as e:
            click.echo(f"Error reading prompt file: {str(e)}", err=True)
            sys.exit(1)
    
    if not api_key:
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif provider == 'openrouter':
            api_key = os.getenv('OPENROUTER_API_KEY')
        elif provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            click.echo(f"Error: API key required. Set {provider.upper()}_API_KEY environment variable or use --api-key option.", err=True)
            sys.exit(1)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        entries = SRTParser.parse(content)
        
        if not entries:
            click.echo("Error: No subtitle entries found in the input file.", err=True)
            sys.exit(1)
        
        if not quiet:
            click.echo(f"Found {len(entries)} subtitle entries")
        
        if dry_run:
            click.echo("Dry run - would translate:")
            for entry in entries[:5]:
                click.echo(f"  {entry.index}: {entry.text}")
            if len(entries) > 5:
                click.echo(f"  ... and {len(entries) - 5} more entries")
            return
        
        translator = LLMTranslator(provider, api_key, model)
        
        texts = [entry.text for entry in entries]
        translated_texts = translator.translate_batch(texts, target_lang, source_lang, quiet, custom_prompt, batch_mode, batch_size)
        
        for entry, translated_text in zip(entries, translated_texts):
            entry.text = translated_text
        
        SRTParser.write(entries, output_file)
        if not quiet:
            click.echo(f"Translation completed! Output saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    translate_subtitles()