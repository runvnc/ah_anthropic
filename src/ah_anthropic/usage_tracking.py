"""Usage tracking integration for Anthropic plugin."""
import os
from datetime import datetime
from typing import Optional
from lib.providers.services import service, hook

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
        'claude.input_tokens',
        'Claude API input token cost',
        'tokens',
        context
    )
    
    await context.register_cost_type(
        PLUGIN_ID,
        'claude.output_tokens',
        'Claude API output token cost',
        'tokens',
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
            'claude.input_tokens',
            usage.input_tokens,
            metadata,
            context,
            model
        )
    except Exception as e:
        print(f"Error tracking message start usage: {e}")

async def track_message_delta(chunk, total_output: str, model: str, context=None):
    """Track usage from message_delta event - output tokens only"""
    if not context or not hasattr(chunk, 'usage'):
        return

    try:
        metadata = {'total_output_length': len(total_output)}
        
        # Track output tokens from final delta
        await context.track_usage(
            PLUGIN_ID,
            'claude.output_tokens',
            chunk.usage.output_tokens,
            metadata,
            context,
            model
        )
    except Exception as e:
        print(f"Error tracking message delta usage: {e}")

@hook()
async def startup(app, context=None):
    """Register cost types during startup"""
    if context:
        await register_cost_types(context)
