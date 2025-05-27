import asyncio
from lib.providers.services import service
import anthropic
import os
import base64
from io import BytesIO
import sys
import json
from .message_utils import compare_messages
from .usage_tracking import *
from lib.utils.backoff import ExponentialBackoff
client = anthropic.AsyncAnthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
anthropic_backoff_manager = ExponentialBackoff(initial_delay=2.0, max_delay=32.0, factor=2, jitter=True)
MAX_RETRIES = 8
_last_messages = []

def prepare_message_content(message):
    """Convert message content to proper format without modifying original"""
    msg_copy = dict(message)
    if isinstance(msg_copy.get('content'), str):
        msg_copy['content'] = [{'type': 'text', 'text': msg_copy['content']}]
    return msg_copy

def prepare_system_message(message):
    """Prepare the system message with cache control"""
    if isinstance(message['content'], str):
        return [{'type': 'text', 'text': message['content'], 'cache_control': {'type': 'ephemeral'}}]
    else:
        text = message['content'][0]['text']
        return [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]

def prepare_formatted_messages(messages):
    """Format all non-system messages and remove existing cache control"""
    formatted_messages = [prepare_message_content(msg) for msg in messages]
    for message in formatted_messages:
        if isinstance(message['content'], list):
            for content in message['content']:
                if 'cache_control' in content:
                    del content['cache_control']
    return formatted_messages

def apply_message_caching(formatted_messages, last_messages):
    """Apply caching strategy to messages and return updated messages"""
    changed_indices = compare_messages(last_messages, formatted_messages)
    cache_candidates = [i for i in range(len(formatted_messages)) if i not in changed_indices]
    messages_to_cache = cache_candidates[-3:] if len(cache_candidates) > 3 else cache_candidates
    cached_count = 1
    for i in messages_to_cache:
        for content in formatted_messages[i]['content']:
            if cached_count < 3 and content['type'] == 'text':
                content['cache_control'] = {'type': 'ephemeral'}
                cached_count += 1
    return formatted_messages

def get_thinking_budget(context):
    """Get thinking budget from environment variable or use default"""
    thinking_level = os.environ.get('MR_THINKING_LEVEL', 'medium').lower()
    if context is not None:
        thinking_level = context.agent.get('thinking_level', thinking_level)
    budgets = {'off': 0, 'minimal': 1024, 'low': 4000, 'medium': 8000, 'high': 16000, 'very_high': 32000, 'maximum': 64000}
    if thinking_level in budgets:
        return budgets[thinking_level]
    try:
        budget = int(thinking_level)
        return max(1024, budget) if budget > 0 else 0
    except ValueError:
        return budgets['medium']

async def handle_stream_chunk(chunk, total_output, model, context, in_thinking_block):
    """Process a single chunk from the stream and yield appropriate content"""
    debug_log_response(chunk)
    if chunk.type == 'message_start':
        await track_message_start(chunk, model, context)
        return ('', in_thinking_block)
    elif chunk.type == 'content_block_start':
        if hasattr(chunk, 'content_block') and chunk.content_block.type == 'thinking':
            return ('', True)
        return ('', in_thinking_block)
    elif chunk.type == 'content_block_delta':
        if in_thinking_block:
            if os.environ.get('AH_DEBUG') == 'True':
                if hasattr(chunk.delta, 'thinking'):
            if hasattr(chunk.delta, 'thinking'):
                return (chunk.delta.thinking, in_thinking_block)
        else:
            if os.environ.get('AH_DEBUG') == 'True':
                if hasattr(chunk.delta, 'text'):
            return (chunk.delta.text, in_thinking_block)
    elif chunk.type == 'content_block_stop':
        if in_thinking_block:
            return ('', False)
        return ('', in_thinking_block)
    elif chunk.type == 'message_delta':
        await track_message_delta(chunk, total_output, model, context)
        return ('', in_thinking_block)
    else:
        if os.environ.get('AH_DEBUG') == 'True':
        return ('', in_thinking_block)
    return ('', in_thinking_block)

@service()
async def stream_chat(model=None, messages=[], context=None, num_ctx=200000, temperature=0.0, max_tokens=32000, num_gpu_layers=0):
    global _last_messages
    if model is None:
        model_name = 'claude-3-7-sonnet-latest'
    else:
        model_name = model
    for attempt_num in range(MAX_RETRIES + 1):
        try:
            wait_time = anthropic_backoff_manager.get_wait_time(model_name)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            messages = [dict(message) for message in messages]
            max_tokens = os.environ.get('MR_MAX_TOKENS', 4000)
            max_tokens = int(max_tokens)
            thinking_budget = get_thinking_budget(context)
            thinking_enabled = thinking_budget > 0
            system = prepare_system_message(messages[0])
            formatted_messages = prepare_formatted_messages(messages[1:])
            formatted_messages = apply_message_caching(formatted_messages, _last_messages)
            _last_messages = formatted_messages.copy()
            kwargs = {'model': model_name, 'system': system, 'messages': formatted_messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': True, 'extra_headers': {'anthropic-beta': 'prompt-caching-2024-07-31,output-128k-2025-02-19'}}
            if thinking_enabled:
                kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget}
                kwargs['temperature'] = 1
                max_tokens = thinking_budget * 2
                kwargs['max_tokens'] = max_tokens
            original_stream = await client.messages.create(**kwargs)
            anthropic_backoff_manager.record_success(model_name)

            async def content_stream():
                total_output = ''
                thinking_content = ''
                in_thinking_block = False
                thinking_emitted = False
                if thinking_enabled:
                    yield '[{"reasoning": "'
                    thinking_emitted = True
                async for chunk in original_stream:
                    chunk_text, new_thinking_state = await handle_stream_chunk(chunk, total_output, model, context, in_thinking_block)
                    if new_thinking_state != in_thinking_block:
                        in_thinking_block = new_thinking_state
                        if not in_thinking_block and thinking_emitted and (chunk.type == 'content_block_stop'):
                            yield '"}] \n'
                    if chunk_text:
                        if in_thinking_block:
                            json_str = json.dumps(chunk_text)
                            without_quotes = json_str[1:-1]
                            yield without_quotes
                            thinking_content += chunk_text
                        else:
                            yield chunk_text
                            total_output += chunk_text
            return content_stream()
        except Exception as e:
            anthropic_backoff_manager.record_failure(model_name)
            if attempt_num < MAX_RETRIES:
                next_wait = anthropic_backoff_manager.get_wait_time(model_name)
                continue
            else:
                raise e

@service()
async def format_image_message(pil_image, context=None):
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image_base64}}

@service()
async def get_image_dimensions(context=None):
    return (1568, 1568, 1192464)

@service()
async def get_service_models(context=None):
    """Get available models for the service"""
    try:
        all_models = await client.models.list()
        ids = []
        for model in all_models.data:
            ids.append(model.id)
        return {'stream_chat': ids}
    except Exception as e:
        return {'stream_chat': []}