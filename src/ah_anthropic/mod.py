import asyncio
from lib.providers.services import service
import anthropic
import os
import base64
from io import BytesIO
import sys

client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, temperature=0.0, max_tokens=2500, num_gpu_layers=0):
    try:
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
            # check if converted to dict
            # if converted, remove any cache_control
            if isinstance(message, dict):
                if 'content' in message:
                    if isinstance(message['content'], list):
                        for content in message['content']:
                            if 'cache_control' in content:
                                del content['cache_control']

        for i in range(-1, -4, -1):
            if len(messages) >= abs(i):
                if isinstance(messages[i]['content'], str):
                    messages[i]['content'] = [{
                        "type": "text",
                        "text": messages[i]['content'],
                        "cache_control": { "type": "ephemeral" }
                    }]

        #new_messages = []
        #for message in messages:
        #    print(str(message)[:500])
        #    #print('\033[93m' + str(message) + '\033[0m')
        #    # print object type (should be dict)
        #    print(type(message))
        #    new_parts = []
        #   drop = False
        #    for part in message['content']:
        #        if not 'type' in part:
        #print('Dropping message')
        #drop = True
        #else:
        #new_parts.append(part)
        #if not drop:
        #new_messages.append(message)

        #print("OK messages:", new_messages)

        original_stream = await client.messages.create(
                model=model,
                system=system,
                messages=messages, #new_messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31,max-tokens-3-5-sonnet-2024-07-15"}
        )
        async def content_stream():
            async for chunk in original_stream:
                if chunk.type == 'content_block_delta':
                    if os.environ.get('AH_DEBUG') == 'True':
                        print('\033[92m' + chunk.delta.text + '\033[0m', end='')
                    yield chunk.delta.text
                else:
                    if os.environ.get('AH_DEBUG') == 'True':
                        # print all chunk data in cyan
                        print('\033[96m' + str(chunk) + '\033[0m', end='')
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
