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

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, temperature=0.0, max_tokens=2500, num_gpu_layers=0):
    try:
        global _last_messages
        print("anthropic stream_chat")
        messages = [dict(message) for message in messages]
        print('\033[93m' + '-'*80 + '\033[0m')
 
        model = "claude-3-5-sonnet-20241022"
        system = messages[0]['content']
        system = [{
            "type": "text",
            "text": system,
            "cache_control": { "type": "ephemeral" }
        }]
        messages = messages[1:]

        # remove any existing cache_control
        for message in messages:
            if isinstance(message, dict):
                if 'content' in message:
                    if isinstance(message['content'], list):
                        for content in message['content']:
                            if 'cache_control' in content:
                                del content['cache_control']

        # Find changed messages
        changed_indices = compare_messages(_last_messages, messages)
        
        # We can cache up to 4 sections including system
        # So we have 3 slots for messages
        # Strategy: Cache the system message and up to 3 most recent unchanged messages
        cache_candidates = [i for i in range(len(messages)) if i not in changed_indices]
        messages_to_cache = cache_candidates[-3:] if len(cache_candidates) > 3 else cache_candidates

        # Add cache control to selected messages
        for i in messages_to_cache:
            if isinstance(messages[i]['content'], str):
                messages[i]['content'] = [{
                    "type": "text",
                    "text": messages[i]['content'],
                    "cache_control": { "type": "ephemeral" }
                }]
            elif isinstance(messages[i]['content'], list):
                for content in messages[i]['content']:
                    if 'type' in content and content['type'] == 'text':
                        content['cache_control'] = { "type": "ephemeral" }

        # Store current messages for next comparison
        _last_messages = messages.copy()

        original_stream = await client.messages.create(
                model=model,
                system=system,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
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