"""Usage tracking integration for Anthropic plugin."""
import os
from datetime import datetime
from typing import Optional
from lib.providers.services import service
from lib.providers.hooks import hook

PLUGIN_ID = 'ah_anthropic'

def debug_log_response(chunk, debug_file="/tmp/anthropic_debug.log"):
    """Log detailed response data for analysis."""
    timestamp = datetime.now().isoformat()
    
    with open(debug_file, 'a') as f:
        f.write(f"\n--- Response Chunk at {timestamp} ---\n")
        f.write(f"Chunk type: {type(chunk)}\n")
        f.write(f"Chunk dir: {dir(chunk)}\n")
        
        try:
            f.write(f"Chunk dict: {chunk.__dict__}\n")
        except:
            f.write("Could not get chunk.__dict__\n")
            
        try:
            f.write(f"Chunk str: {str(chunk)}\n")
        except:
            f.write("Could not convert chunk to string\n")
            
        try:
            f.write(f"Chunk repr: {repr(chunk)}\n")
        except:
            f.write("Could not get chunk repr\n")
            
        f.write("-" * 80 + "\n")

@service()
async def register_cost_types(context=None):
    """Register Anthropic API cost types"""
    if not context:
        return
        
    await context.register_cost_type(
        PLUGIN_ID,
        'stream_chat.input_tokens',
        'Claude stream_chat input token cost',
        'tokens',
        context
    )
    
    await context.register_cost_type(
        PLUGIN_ID,
        'stream_chat.output_tokens',
        'Claude stream_chat output token cost',
        'tokens',
        context
    )

@service()
async def set_default_costs(context=None):
    """Set default costs for Claude API usage.
    These costs are approximate and should be updated based on actual pricing.
    See: https://anthropic.com/pricing
    """
    if not context:
        return

    # Claude-3-Sonnet pricing (approximate)
    await context.set_cost(
        PLUGIN_ID,
        'stream_chat.input_tokens',
        0.000003,  # $3 per million tokens
        'claude-3-5-sonnet-20241022',
        context
    )
    
    await context.set_cost(
        PLUGIN_ID,
        'stream_chat.output_tokens',
        0.000015,  # $15 per million tokens
        'claude-3-5-sonnet-20241022',
        context
    )

async def track_message_start(chunk, model: str, context=None):
    """Track usage from message_start event - input tokens only"""
    if not context or not hasattr(chunk, 'message') or not hasattr(chunk.message, 'usage'):
        return

    try:
        usage = chunk.message.usage
        metadata = {
            'cache_creation_tokens': usage.cache_creation_input_tokens,
            'cache_read_tokens': usage.cache_read_input_tokens
        }
        
        # Track input tokens
        await context.track_usage(
            PLUGIN_ID,
            'stream_chat.input_tokens',
            usage.input_tokens,
            metadata,
            context,
            model
        )
    except Exception as e:
        print(f"Error tracking message start usage: {e}")
        throw(e)

async def track_message_delta(chunk, total_output: str, model: str, context=None):
    """Track usage from message_delta event - output tokens only"""
    print("track_message_delta")
    if not context or not hasattr(chunk, 'usage'):
        return

    print("track_message_delta 2")
    try:
        metadata = {'total_output_length': len(total_output)}
        
        # Track output tokens from final delta
        await context.track_usage(
            PLUGIN_ID,
            'stream_chat.output_tokens',
            chunk.usage.output_tokens,
            metadata,
            context,
            model
        )
    except Exception as e:
        print(f"Error tracking message delta usage: {e}")
        raise e

async def track_message_usage(chunk, total_output: str, model: str, context=None):
    """Track usage from a message chunk if it contains usage information."""
    if not context or not hasattr(chunk, 'usage'):
        return

    try:
        metadata = {'total_output_length': len(total_output)}
        
        # Track input tokens
        await context.track_usage(
            PLUGIN_ID,
            'stream_chat.input_tokens',
            chunk.usage.input_tokens,
            metadata,
            context,
            model
        )
        
        # Track output tokens
        await context.track_usage(
            PLUGIN_ID,
            'stream_chat.output_tokens',
            chunk.usage.output_tokens,
            metadata,
            context,
            model
        )
    except Exception as e:
        print(f"Error tracking usage: {e}")
        raise e

@hook()
async def startup(app, context=None):
    """Register cost types and set default costs during startup"""
    if context:
        await register_cost_types(context)
        await set_default_costs(context)
