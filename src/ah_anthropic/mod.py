import asyncio
from lib.providers.services import service
import anthropic
import os
import base64
from io import BytesIO
import sys
from .message_utils import compare_messages

client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Store last sent messages
_last_messages = []

def prepare_message_content(message):
    """Convert message content to proper format without modifying original"""
    msg_copy = dict(message)
    if isinstance(msg_copy.get('content'), str):
        msg_copy['content'] = [{
            "type": "text",
            "text": msg_copy['content']
        }]
    return msg_copy

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, temperature=0.0, max_tokens=2500, num_gpu_layers=0):
    try:
        global _last_messages
        print("anthropic stream_chat")
        messages = [dict(message) for message in messages]
        print('\033[93m' + '-'*80 + '\033[0m')
 
        model = "claude-3-5-sonnet-20241022"
        
        # Prepare system message
        system = [{
            "type": "text",
            "text": messages[0]['content'],
            "cache_control": { "type": "ephemeral" }
        }]
        
        # Prepare messages with proper content format
        formatted_messages = [prepare_message_content(msg) for msg in messages[1:]]

        # remove any existing cache_control
        for message in formatted_messages:
            if isinstance(message['content'], list):
                for content in message['content']:
                    if 'cache_control' in content:
                        del content['cache_control']

        # Find changed messages
        changed_indices = compare_messages(_last_messages, formatted_messages)
        
        # We can cache up to 4 sections including system
        # So we have 3 slots for messages
        # Strategy: Cache the system message and up to 3 most recent unchanged messages
        cache_candidates = [i for i in range(len(formatted_messages)) if i not in changed_indices]
        messages_to_cache = cache_candidates[-3:] if len(cache_candidates) > 3 else cache_candidates

        # Add cache control to selected messages
        for i in messages_to_cache:
            for content in formatted_messages[i]['content']:
                if content['type'] == 'text':
                    content['cache_control'] = { "type": "ephemeral" }

        # Store current messages for next comparison
        _last_messages = formatted_messages.copy()

        original_stream = await client.messages.create(
                model=model,
                system=system,
                messages=formatted_messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31,max-tokens-3-5-sonnet-2024-07-15"}
        )

        async def content_stream():
            async for chunk in original_stream:
                print("HI THERE...")
                print("x->")
                if chunk.type == 'content_block_delta':
                    print(1)
                    if os.environ.get('AH_DEBUG') == 'True':
                        print('\033[92m' + chunk.delta.text + '\033[0m', end='')
                    yield chunk.delta.text
                else:
                    print(2)
                    if os.environ.get('AH_DEBUG') == 'True':
                        print('\033[93m' + '-'*80 + '\033[0m')
                        print('\033[93m' + str(chunk.type) + '\033[0m')
                        print('\033[93m' + str(chunk) + '\033[0m')
                    yield ''

        return content_stream()

    except Exception as e:
        print('claude.ai error:', e)


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
