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

client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize a global backoff manager for Anthropic services
# You might want to make these parameters configurable via environment variables or a config file
anthropic_backoff_manager = ExponentialBackoff(initial_delay=2.0, max_delay=32.0, factor=2, jitter=True)
MAX_RETRIES = 8

# Store last sent messages
_last_messages = []

def prepare_message_content(message):
    """Convert message content to proper format without modifying original"""
    msg_copy = dict(message)
    if isinstance(msg_copy.get('content'), str):
        msg_copy['content'] = [{'type': 'text', 'text': msg_copy['content']}]
    return msg_copy

def prepare_system_message(message):
    """Prepare the system message with cache control"""
    # if message content is a string
    if isinstance(message['content'], str):
        return [{'type': 'text', 'text': message['content'], 'cache_control': {'type': 'ephemeral'}}]
    else:
        text = message['content'][0]['text']
        return [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
 

def prepare_formatted_messages(messages):
    """Format all non-system messages and remove existing cache control"""
    formatted_messages = [prepare_message_content(msg) for msg in messages]
    
    # Remove any existing cache_control
    for message in formatted_messages:
        if isinstance(message['content'], list):
            for content in message['content']:
                if 'cache_control' in content:
                    del content['cache_control']
                    
    return formatted_messages

def apply_message_caching(formatted_messages, last_messages):
    """Apply caching strategy to messages and return updated messages"""
    # Find changed messages
    changed_indices = compare_messages(last_messages, formatted_messages)
    
    # We can cache up to 4 sections including system
    # So we have 3 slots for messages
    # Strategy: Cache the system message and up to 3 most recent unchanged messages
    cache_candidates = [i for i in range(len(formatted_messages)) if i not in changed_indices]
    messages_to_cache = cache_candidates[-3:] if len(cache_candidates) > 3 else cache_candidates

    cached_count = 1  # Start at 1 to account for system message
    for i in messages_to_cache:
        for content in formatted_messages[i]['content']:
            if cached_count < 3 and content['type'] == 'text':
                content['cache_control'] = {'type': 'ephemeral'}
                cached_count += 1
                
    return formatted_messages


def get_thinking_budget(context):
    """Get thinking budget from environment variable or use default"""
    thinking_level = os.environ.get('MR_THINKING_LEVEL', 'medium').lower()

    print("Context:")
    print(context)

    if context is not None:
        thinking_level = context.agent.get('thinking_level', thinking_level)

    print("Context thinking level", thinking_level)
    # Define thinking budgets based on level
    budgets = {
        'off': 0,
        'minimal': 1024,  # Minimum allowed
        'low': 4000,
        'medium': 8000,
        'high': 16000,
        'very_high': 32000,
        'maximum': 64000
    }

    # Get budget from level or parse direct number
    if thinking_level in budgets:
        return budgets[thinking_level]
    
    # Try to parse as direct number
    try:
        budget = int(thinking_level)
        # Ensure minimum of 1024 if enabled
        return max(1024, budget) if budget > 0 else 0
    except ValueError:
        # Default to medium if invalid
        return budgets['medium']

async def handle_stream_chunk(chunk, total_output, model, context, in_thinking_block):
    """Process a single chunk from the stream and yield appropriate content"""
    debug_log_response(chunk)
    
    if chunk.type == 'message_start':
        await track_message_start(chunk, model, context)
        return '', in_thinking_block
    elif chunk.type == 'content_block_start':
        # Check if this is a thinking block
        if hasattr(chunk, 'content_block') and chunk.content_block.type == 'thinking':
            return '', True  # We're entering a thinking block
        return '', in_thinking_block
    elif chunk.type == 'content_block_delta':
        if in_thinking_block:
            if os.environ.get('AH_DEBUG') == 'True':
                if hasattr(chunk.delta, 'thinking'):
                    print('\033[92m' + chunk.delta.thinking + '\033[0m', end='')
            if hasattr(chunk.delta, 'thinking'):
                return chunk.delta.thinking, in_thinking_block
        else:
            if os.environ.get('AH_DEBUG') == 'True':
                if hasattr(chunk.delta, 'text'):
                    print('\033[92m' + chunk.delta.text + '\033[0m', end='') 
            return chunk.delta.text, in_thinking_block
    elif chunk.type == 'content_block_stop':
        # Check if we're exiting a thinking block
        if in_thinking_block:
            return '', False  # We're exiting a thinking block
        return '', in_thinking_block
    elif chunk.type == 'message_delta':
        await track_message_delta(chunk, total_output, model, context)
        return '', in_thinking_block
    else:
        if os.environ.get('AH_DEBUG') == 'True':
            print('\033[93m' + '-'*80 + '\033[0m')
            print('\033[93m' + str(chunk.type) + '\033[0m')
            print('\033[93m' + str(chunk) + '\033[0m')
        return '', in_thinking_block

    return '', in_thinking_block

@service()
async def stream_chat(model=None, messages=[], context=None, num_ctx=200000, temperature=0.0, max_tokens=32000, num_gpu_layers=0):
    global _last_messages
    
    # Determine model name at the beginning
    if model is None:
        print("Anthropic: model not specified; using default model claude-3-7-sonnet-latest")
        model_name = "claude-3-7-sonnet-latest"
    else:
        model_name = model
    
    for attempt_num in range(MAX_RETRIES + 1):
        try:
            # Check and honor backoff before making the API call
            wait_time = anthropic_backoff_manager.get_wait_time(model_name)
            if wait_time > 0:
                print(f"Anthropic model '{model_name}' is in backoff. Waiting for {wait_time:.2f} seconds before attempt {attempt_num + 1}.")
                await asyncio.sleep(wait_time)
            
            # Original function logic starts here
            print("anthropic stream_chat")
            messages = [dict(message) for message in messages]
            print('\033[93m' + '-'*80 + '\033[0m')
     
            max_tokens = os.environ.get("MR_MAX_TOKENS", 4000)
            max_tokens = int(max_tokens)
            
            # Get thinking budget
            thinking_budget = get_thinking_budget(context)
            thinking_enabled = thinking_budget > 0
            
            # Prepare messages
            system = prepare_system_message(messages[0])
            formatted_messages = prepare_formatted_messages(messages[1:])
            
            # Apply caching strategy
            formatted_messages = apply_message_caching(formatted_messages, _last_messages)
            _last_messages = formatted_messages.copy()

            # Debug output
            print('\033[93m' + 'formatted_messages' + '\033[0m')
            print(json.dumps(formatted_messages, indent=4))
            # Create message stream with thinking if enabled
            kwargs = {
                'model': model_name,  # Use model_name consistently
                'system': system,
                'messages': formatted_messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'stream': True,
                'extra_headers': {"anthropic-beta": "prompt-caching-2024-07-31,output-128k-2025-02-19"}
            }
            
            # Add thinking parameters if enabled
            if thinking_enabled:
                print(f"Enabling extended thinking with budget: {thinking_budget} tokens")
                kwargs['thinking'] = {
                    'type': 'enabled',
                    'budget_tokens': thinking_budget
                }
                kwargs['temperature'] = 1
                max_tokens = thinking_budget * 2
                print("-------------------------------------------------------------------")
                print("using thinking budget, adjusted max_tokens to", max_tokens)
                kwargs['max_tokens'] = max_tokens


            original_stream = await client.messages.create(**kwargs)
            
            # If successful, record success
            anthropic_backoff_manager.record_success(model_name)

            async def content_stream():
                total_output = ""
                thinking_content = ""
                in_thinking_block = False
                thinking_emitted = False
                
                # If thinking is enabled, emit the start marker
                if thinking_enabled:
                    yield '[{"reasoning": "'
                    thinking_emitted = True
                async for chunk in original_stream:
                    chunk_text, new_thinking_state = await handle_stream_chunk(
                        chunk, total_output, model, context, in_thinking_block
                    )
                    
                    # Track if we're entering or exiting a thinking block
                    if new_thinking_state != in_thinking_block:
                        in_thinking_block = new_thinking_state
                        
                        # If we're exiting a thinking block and entering content, emit the separator
                        if not in_thinking_block and thinking_emitted and chunk.type == 'content_block_stop':
                            yield '"}] \n'

                            
                    # Handle the chunk text based on whether we're in a thinking block
                    if chunk_text:
                        if in_thinking_block:
                            # For thinking blocks, escape the text like in DeepSeek
                            json_str = json.dumps(chunk_text)
                            without_quotes = json_str[1:-1]
                            yield without_quotes
                            thinking_content += chunk_text
                        else:
                            # For regular content, just yield the text
                            yield chunk_text
                            total_output += chunk_text

            return content_stream()

        except Exception as e:
            # For this test, apply backoff to all errors (not just rate limit errors)
            # In production, you might want to check for specific rate limit error messages
            
            # Record failure for this model
            anthropic_backoff_manager.record_failure(model_name)
            print(f"Anthropic error for '{model_name}' on attempt {attempt_num + 1}/{MAX_RETRIES + 1}: {e}")
            
            if attempt_num < MAX_RETRIES:
                next_wait = anthropic_backoff_manager.get_wait_time(model_name)
                print(f"Will retry after ~{next_wait:.2f} seconds.")
                continue  # Go to the next iteration of the loop to retry
            else:
                print(f"Max retries ({MAX_RETRIES + 1}) reached for '{model_name}'. Raising final error.")
                raise e  # Max retries exceeded, raise the last error

@service()
async def format_image_message(pil_image, context=None):
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_base64
        }
    }

@service()
async def get_image_dimensions(context=None):
    return 1568, 1568, 1192464


@service()
async def get_service_models(context=None):
    """Get available models for the service"""
    try:
        all_models = await client.models.list()
        print(all_models)
        ids = []
        for model in all_models.data:
            print('#####################################################')
            print(model)
            ids.append(model.id)

        return { "stream_chat": ids }
    except Exception as e:
        print('Error getting models (Anthropic):', e)
        return { "stream_chat": [] }
