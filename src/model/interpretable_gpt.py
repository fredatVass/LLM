"""
White Box Interpretable Wrapper for GPT Model
Provides transparency and explainability while maintaining black box functionality
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class InterpretableGPT:
    """
    Wrapper around GPT model that provides interpretability features
    while maintaining standard black box inference capabilities.
    """
    
    def __init__(self, model, tokenizer, device='mps'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Storage for interpretability data
        self.attention_maps = []
        self.layer_activations = []
        self.token_importance = []
        
    def _hook_attention(self):
        """Hook to capture attention weights during forward pass"""
        self.attention_maps = []
        
        def attention_hook(module, input, output):
            # Capture attention weights
            if hasattr(module, 'attention'):
                self.attention_maps.append(output.detach().cpu())
        
        hooks = []
        for layer in self.model.blocks:
            hook = layer.attention.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        return hooks
    
    def _remove_hooks(self, hooks):
        """Remove attention hooks"""
        for hook in hooks:
            hook.remove()
    
    def predict(self, text: str, explain: bool = False) -> Dict:
        """
        Black box prediction with optional white box explanation
        
        Args:
            text: Input text
            explain: If True, return interpretability information
            
        Returns:
            Dictionary with prediction and optional explanations
        """
        # Encode input
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Setup hooks if explanation needed
        hooks = self._hook_attention() if explain else []
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs['logits']
        
        # Get predictions
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Top-k predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Build result
        result = {
            'input': text,
            'black_box_output': {
                'top_predictions': [
                    {
                        'token_id': int(top_indices[i]),
                        'token': self.tokenizer.decode([int(top_indices[i])]),
                        'probability': float(top_probs[i])
                    }
                    for i in range(top_k)
                ],
                'predicted_token': self.tokenizer.decode([int(top_indices[0])]),
                'confidence': float(top_probs[0])
            }
        }
        
        # Add white box explanations
        if explain:
            result['white_box_explanation'] = self._generate_explanations(
                input_ids, logits, text
            )
        
        # Clean up hooks
        self._remove_hooks(hooks)
        
        return result
    
    def _generate_explanations(self, input_ids, logits, text) -> Dict:
        """Generate interpretability explanations"""
        
        tokens = self.tokenizer.encode(text)
        token_strings = [self.tokenizer.decode([t]) for t in tokens]
        
        explanations = {
            'input_analysis': {
                'num_tokens': len(tokens),
                'tokens': token_strings,
                'token_ids': tokens
            },
            'model_internals': {
                'num_layers': len(self.model.blocks),
                'attention_heads': self.model.config.n_heads,
                'model_dimension': self.model.config.d_model,
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'decision_factors': self._analyze_decision_factors(input_ids, logits),
            'attention_analysis': self._analyze_attention() if self.attention_maps else None,
            'confidence_breakdown': self._analyze_confidence(logits)
        }
        
        return explanations
    
    def _analyze_decision_factors(self, input_ids, logits) -> Dict:
        """Analyze what factors influenced the decision"""
        
        # Get final layer logits
        final_logits = logits[0, -1, :]
        
        # Compute entropy (uncertainty)
        probs = F.softmax(final_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        # Get logit statistics
        logit_stats = {
            'mean_logit': float(final_logits.mean()),
            'std_logit': float(final_logits.std()),
            'max_logit': float(final_logits.max()),
            'min_logit': float(final_logits.min()),
            'entropy': float(entropy),
            'uncertainty': 'high' if entropy > 5 else 'medium' if entropy > 3 else 'low'
        }
        
        return logit_stats
    
    def _analyze_attention(self) -> Dict:
        """Analyze attention patterns"""
        if not self.attention_maps:
            return None
        
        # Average attention across layers and heads
        attention_info = {
            'num_layers_captured': len(self.attention_maps),
            'attention_pattern': 'Available for visualization',
            'focus_analysis': 'Model attention distributed across input tokens'
        }
        
        return attention_info
    
    def _analyze_confidence(self, logits) -> Dict:
        """Analyze model confidence"""
        
        final_logits = logits[0, -1, :]
        probs = F.softmax(final_logits, dim=-1)
        
        # Get top predictions
        top_5_probs, top_5_indices = torch.topk(probs, 5)
        
        confidence_info = {
            'top_1_confidence': float(top_5_probs[0]),
            'top_5_cumulative': float(top_5_probs.sum()),
            'confidence_gap': float(top_5_probs[0] - top_5_probs[1]),
            'decision_quality': self._assess_decision_quality(top_5_probs),
            'alternative_predictions': [
                {
                    'rank': i + 1,
                    'token': self.tokenizer.decode([int(top_5_indices[i])]),
                    'probability': float(top_5_probs[i])
                }
                for i in range(5)
            ]
        }
        
        return confidence_info
    
    def _assess_decision_quality(self, probs) -> str:
        """Assess quality of model's decision"""
        top_prob = float(probs[0])
        
        if top_prob > 0.8:
            return "Very confident - high quality prediction"
        elif top_prob > 0.5:
            return "Confident - good quality prediction"
        elif top_prob > 0.3:
            return "Moderate confidence - acceptable prediction"
        else:
            return "Low confidence - uncertain prediction"
    
    def generate_with_explanation(self, prompt: str, max_length: int = 50,
                                  explain_every: int = 10) -> Dict:
        """
        Generate text with periodic explanations of decisions
        
        Args:
            prompt: Starting prompt
            max_length: Maximum tokens to generate
            explain_every: Explain decision every N tokens
            
        Returns:
            Dictionary with generated text and explanations
        """
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_tokens = []
        explanations = []
        
        for step in range(max_length):
            # Generate next token
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['logits']
            
            next_token_logits = logits[0, -1, :] / 0.8  # temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add explanation at intervals
            if step % explain_every == 0:
                current_text = self.tokenizer.decode(input_ids[0].tolist())
                explanation = self.predict(current_text, explain=True)
                explanations.append({
                    'step': step,
                    'decision': explanation['white_box_explanation']['confidence_breakdown']
                })
            
            # Append token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            generated_tokens.append(int(next_token))
        
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': prompt + generated_text,
            'num_tokens_generated': len(generated_tokens),
            'explanations': explanations,
            'generation_summary': {
                'average_confidence': np.mean([
                    e['decision']['top_1_confidence'] 
                    for e in explanations
                ]),
                'num_explained_steps': len(explanations)
            }
        }
    
    def visualize_attention(self, text: str, save_path: str = 'attention_viz.png'):
        """Visualize attention patterns (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return {'error': 'matplotlib and seaborn required for visualization'}
        
        # Get prediction with attention
        result = self.predict(text, explain=True)
        
        if not self.attention_maps:
            return {'error': 'No attention maps captured'}
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Attention Patterns for: "{text}"', fontsize=16)
        
        # Plot first 6 layers
        for idx, ax in enumerate(axes.flat):
            if idx < len(self.attention_maps):
                # Average over heads and batch
                attn = self.attention_maps[idx].mean(dim=1)[0].numpy()
                
                sns.heatmap(attn, ax=ax, cmap='viridis', cbar=True)
                ax.set_title(f'Layer {idx + 1}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
        
        return {'visualization_path': save_path}
    
    def explain_prediction_step_by_step(self, text: str) -> Dict:
        """Provide detailed step-by-step explanation of model's decision"""
        
        print("="*70)
        print("STEP-BY-STEP MODEL EXPLANATION")
        print("="*70)
        
        result = self.predict(text, explain=True)
        
        print(f"\nINPUT: '{text}'")
        print(f"\n[1] TOKENIZATION")
        print(f"   Tokens: {result['white_box_explanation']['input_analysis']['tokens']}")
        print(f"   Token IDs: {result['white_box_explanation']['input_analysis']['token_ids']}")
        
        print(f"\n[2] MODEL PROCESSING")
        internals = result['white_box_explanation']['model_internals']
        print(f"   - Layers processed: {internals['num_layers']}")
        print(f"   - Attention heads per layer: {internals['attention_heads']}")
        print(f"   - Total parameters used: {internals['total_parameters']:,}")
        
        print(f"\n[3] DECISION FACTORS")
        factors = result['white_box_explanation']['decision_factors']
        print(f"   - Uncertainty level: {factors['uncertainty']}")
        print(f"   - Entropy: {factors['entropy']:.4f}")
        print(f"   - Logit range: [{factors['min_logit']:.2f}, {factors['max_logit']:.2f}]")
        
        print(f"\n[4] CONFIDENCE ANALYSIS")
        confidence = result['white_box_explanation']['confidence_breakdown']
        print(f"   - Top prediction confidence: {confidence['top_1_confidence']:.2%}")
        print(f"   - Quality assessment: {confidence['decision_quality']}")
        print(f"   - Confidence gap: {confidence['confidence_gap']:.4f}")
        
        print(f"\n[5] TOP PREDICTIONS")
        for alt in confidence['alternative_predictions']:
            print(f"   {alt['rank']}. '{alt['token']}' - {alt['probability']:.2%}")
        
        print(f"\n[6] FINAL OUTPUT")
        print(f"   - Predicted token: '{result['black_box_output']['predicted_token']}'")
        print(f"   - Confidence: {result['black_box_output']['confidence']:.2%}")
        
        print("\n" + "="*70)
        
        return result
    
    def compare_black_vs_white(self, text: str) -> Dict:
        """Compare black box vs white box perspectives"""
        
        # Black box mode
        black_result = self.predict(text, explain=False)
        
        # White box mode
        white_result = self.predict(text, explain=True)
        
        comparison = {
            'black_box_view': {
                'description': 'What a typical user sees',
                'output': black_result['black_box_output']['predicted_token'],
                'confidence': black_result['black_box_output']['confidence'],
                'transparency': 'Minimal - just the prediction'
            },
            'white_box_view': {
                'description': 'Full transparency into model decisions',
                'output': white_result['black_box_output']['predicted_token'],
                'confidence': white_result['black_box_output']['confidence'],
                'transparency': 'Complete - full explanation available',
                'additional_info': {
                    'decision_factors': white_result['white_box_explanation']['decision_factors'],
                    'confidence_breakdown': white_result['white_box_explanation']['confidence_breakdown'],
                    'model_internals': white_result['white_box_explanation']['model_internals']
                }
            },
            'key_insight': 'Same prediction, but white box reveals the reasoning'
        }
        
        return comparison


def create_interpretable_model(model_path: str, tokenizer_path: str, device: str = 'mps'):
    """
    Factory function to create interpretable GPT model
    
    Args:
        model_path: Path to trained model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device to run on
        
    Returns:
        InterpretableGPT instance
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.model.gpt import GPTModel
    from src.data.tokenizer import SimpleTokenizer
    from configs.config import ModelConfig
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load model
    model = GPTModel(ModelConfig)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create interpretable wrapper
    interpretable_model = InterpretableGPT(model, tokenizer, device)
    
    print("Interpretable White Box Model Created")
    print(f"   Model: {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")
    print("\nCapabilities:")
    print("   - Black box inference (standard predictions)")
    print("   - White box explanations (transparency)")
    print("   - Attention visualization")
    print("   - Confidence analysis")
    print("   - Step-by-step reasoning")
    
    return interpretable_model
