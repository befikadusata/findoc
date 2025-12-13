"""
Attention Visualization Module

This module provides functions for visualizing attention weights from transformer models
to help understand which parts of the input text were most influential in classification decisions.
"""

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from typing import List, Dict, Optional
import re


def visualize_attention_weights(tokens_with_weights: List[Dict[str, float]], 
                               title: str = "Token Attention Visualization",
                               figsize: tuple = (12, 3)) -> str:
    """
    Visualize attention weights for tokens using matplotlib.
    
    Args:
        tokens_with_weights: List of dictionaries with 'token' and 'attention_weight' keys
        title: Title for the visualization
        figsize: Figure size as (width, height)
    
    Returns:
        str: Base64-encoded image data URL for embedding in HTML
    """
    if not tokens_with_weights:
        return ""
    
    # Extract tokens and weights
    tokens = [item['token'] for item in tokens_with_weights]
    weights = [item['attention_weight'] for item in tokens_with_weights]
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a color map based on attention weights
    colors = plt.cm.viridis(np.array(weights) / max(weights))
    
    # Create a horizontal bar chart
    y_pos = np.arange(len(tokens))
    bars = ax.barh(y_pos, weights, color=colors)
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels([format_token_label(token) for token in tokens])
    ax.set_xlabel('Attention Weight')
    ax.set_title(title)
    
    # Add value labels on bars
    for i, (weight, bar) in enumerate(zip(weights, bars)):
        ax.text(weight, i, f'{weight:.3f}', va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    
    # Convert to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode as base64
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    plt.close()  # Close the figure to free memory
    
    return f"data:image/png;base64,{graphic}"


def visualize_token_attention_inline(tokens_with_weights: List[Dict[str, float]], 
                                    max_display: int = 20) -> str:
    """
    Create an inline visualization of token attention weights using HTML spans.
    
    Args:
        tokens_with_weights: List of dictionaries with 'token' and 'attention_weight' keys
        max_display: Maximum number of tokens to display
    
    Returns:
        str: HTML string with color-coded tokens based on attention weights
    """
    if not tokens_with_weights:
        return ""
    
    # Take only the requested number of tokens
    tokens_subset = tokens_with_weights[:max_display]
    
    # Calculate min and max weights to normalize
    weights = [item['attention_weight'] for item in tokens_subset]
    min_weight = min(weights)
    max_weight = max(weights)
    
    # If all weights are the same, set min to be slightly less than max
    if min_weight == max_weight:
        min_weight = max_weight * 0.9 if max_weight != 0 else 0.1
    
    range_weight = max_weight - min_weight
    
    # Generate HTML with color-coded spans
    html_parts = []
    for item in tokens_subset:
        token = item['token']
        weight = item['attention_weight']
        
        # Normalize weight to 0-1 range
        if range_weight != 0:
            norm_weight = (weight - min_weight) / range_weight
        else:
            norm_weight = 0.5  # Middle value if all weights are the same
        
        # Map to color intensity (0 = light, 1 = dark)
        # Use a color from yellow to red to indicate importance
        r = 255
        g = int(255 * (1 - norm_weight))
        b = int(255 * (1 - norm_weight))
        
        color = f"rgb({r}, {g}, {b})"
        html_parts.append(f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Weight: {weight:.3f}">{token}</span>')
    
    return " ".join(html_parts)


def format_token_label(token: str) -> str:
    """
    Format a token for display in visualization.
    
    Args:
        token: The token string
    
    Returns:
        str: Formatted token string for visualization
    """
    # Clean up special tokens for better display
    if token.startswith("##"):
        # Handle subword tokens from BPE
        return token[2:]  # Remove the '##' prefix
    elif token in ['[CLS]', '[SEP]', '[PAD]']:
        return f'[{token[1:-1]}]'  # Keep special tokens but make them more readable
    else:
        # Remove special characters that might cause display issues
        return re.sub(r'[^\w\s-]', '', token)


def create_attention_summary(tokens_with_weights: List[Dict[str, float]], 
                           top_n: int = 5) -> Dict:
    """
    Create a summary of the attention weights.
    
    Args:
        tokens_with_weights: List of dictionaries with 'token' and 'attention_weight' keys
        top_n: Number of top tokens to include in summary
    
    Returns:
        Dict: Summary statistics about the attention weights
    """
    if not tokens_with_weights:
        return {}
    
    weights = [item['attention_weight'] for item in tokens_with_weights]
    
    # Get top tokens
    sorted_tokens = sorted(tokens_with_weights, key=lambda x: x['attention_weight'], reverse=True)
    top_tokens = sorted_tokens[:top_n]
    
    return {
        'total_tokens': len(tokens_with_weights),
        'avg_attention': np.mean(weights),
        'std_attention': np.std(weights),
        'max_attention': np.max(weights),
        'min_attention': np.min(weights),
        'top_tokens': [{'token': item['token'], 'weight': item['attention_weight']} for item in top_tokens],
        'attention_distribution': weights
    }


if __name__ == "__main__":
    # Example usage
    sample_tokens = [
        {'token': 'INVOICE', 'attention_weight': 0.85},
        {'token': 'Date', 'attention_weight': 0.72},
        {'token': ':', 'attention_weight': 0.15},
        {'token': '2023', 'attention_weight': 0.68},
        {'token': '-', 'attention_weight': 0.20},
        {'token': '06', 'attention_weight': 0.18},
        {'token': '-', 'attention_weight': 0.15},
        {'token': '15', 'attention_weight': 0.65},
        {'token': 'Customer', 'attention_weight': 0.70},
        {'token': ':', 'attention_weight': 0.12}
    ]
    
    # Create and save the visualization
    img_url = visualize_attention_weights(sample_tokens)
    print(f"Generated image URL (first 100 chars): {img_url[:100]}...")
    
    # Create inline visualization
    inline_html = visualize_token_attention_inline(sample_tokens)
    print(f"Inline visualization HTML: {inline_html}")
    
    # Create summary
    summary = create_attention_summary(sample_tokens)
    print(f"Attention summary: {summary}")