"""
Transformer architecture options

Implements:
- Trainable linear attention (as specified in theory)
- More realistic transformer model 

Mary Letey
January 2026
With help from ChatGPT 5.2 Thinking
"""

import functools
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct


@struct.dataclass
class TransformerConfig:
    n_layers: int = 3
    n_hidden: int = 128
    n_out: int = 1
    max_len: int = 100

    # MLP Configus
    n_mlp_layers: int = 2          
    mlp_multiplier: int = 4 # projecting to larger hidden dim in MLP       

    # Regularisation
    dropout_rate: float = 0.1

    # if False, hidden = input D
    use_input_projection: bool = True  

    # We usually just want single logit for y_{ell+1}
    return_final_logits_only: bool = True

    # Toggle on for theory-defined linear attention module (single layer)
    pure_linear_self_att: bool = False

    def to_model(self):
        return Transformer(self)


class SingleHeadSelfAttention(nn.Module):
    """
    Single-head self-attention.
    For mask=None, this is full (all-to-all) attention. I ALWAYS USE THIS. 
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, mask: Optional[jnp.ndarray] = None, idxs=None, use_bias: bool = False) -> jnp.ndarray:
        H = inputs.shape[-1]
        dense = functools.partial(
            nn.Dense,
            features=H,
            use_bias=use_bias,
        )

        self.sow("intermediates", "inputs", inputs)

        query = dense(name="query")(inputs)   # (B, L, Hidden)
        key = dense(name="key")(inputs)       # (B, L, Hidden)
        value = dense(name="value")(inputs)   # (B, L, Hidden)
        depth = query.shape[-1]

        # (B, L, L)
        attn_logits = jnp.einsum("...qd,...kd->...qk", query, key)
        attn_logits = attn_logits / jnp.sqrt(depth).astype(attn_logits.dtype)
        self.sow("intermediates", "raw_att", attn_logits)

        if mask is not None:
            # Flax make_*_mask gives (B, 1, L, L); squeeze only the head axis
            m = jnp.squeeze(mask, axis=1)  # (B, L, L)
            neg_inf = jnp.finfo(attn_logits.dtype).min
            attn_logits = jnp.where(m, attn_logits, neg_inf)

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (B, L, L)
        self.sow("intermediates", "attention_weights", attn_weights)

        attn_out = attn_weights @ value  # (B, L, H)
        return attn_out

    
class TransformerBlock(nn.Module):
    """
    Kinda-standard pre-norm block:
      x = x + Dropout(SelfAttn(LN(x)))
      x = x + Dropout(FFN(LN(x)))
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, decoder_mask=None, deterministic=True, idxs=None):
        assert inputs.ndim == 3
        H = inputs.shape[-1]

        # self-attn
        x = nn.LayerNorm()(inputs)
        x = SingleHeadSelfAttention(self.config)(x, decoder_mask, idxs=idxs)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.config.mlp_multiplier * H)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.config.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(H)(y)
        y = nn.Dropout(rate=self.config.dropout_rate)(y, deterministic=deterministic)

        return x + y


class LinearSelfAttentionBlock(nn.Module):
    """
    Implements exactly the paper's linear self-attention:
    Returns A(Z) = Z + VZ (KZ)^T (QZ) / ell
    where Z ∈ R^{(d+1)x(ell+1)} has tokens as columns.
    Note that inputs in this pipeline are (B, ell+1, d+1). 
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, decoder_mask=None, idxs=None):
        assert inputs.ndim == 3
        B, lp1, dp1 = inputs.shape
        ell = lp1 - 1  # ℓ (number of training examples)
        # Z has shape: (B, D, L)
        Z = jnp.swapaxes(inputs, 1, 2)

        # K, Q, V ∈ R^{(D×D)} acting on rows of Z
        # K = self.param("K", nn.initializers.lecun_normal(), (dp1, dp1)) 
        # Q = self.param("Q", nn.initializers.lecun_normal(), (dp1, dp1))
        # V = self.param("V", nn.initializers.lecun_normal(), (dp1, dp1))
        K = self.param("K", lambda key, shape: nn.initializers.lecun_normal()(key, shape) / np.sqrt(dp1-1), (dp1, dp1))
        Q = self.param("Q", lambda key, shape: nn.initializers.lecun_normal()(key, shape) / np.sqrt(dp1-1), (dp1, dp1))
        V = self.param("V", lambda key, shape: nn.initializers.lecun_normal()(key, shape) / np.sqrt(dp1-1), (dp1, dp1))

        # KZ, QZ, VZ: (B, D, L)
        KZ = jnp.einsum("de,bel->bdl", K, Z)
        QZ = jnp.einsum("de,bel->bdl", Q, Z)
        VZ = jnp.einsum("de,bel->bdl", V, Z)

        # ---- supports only (exclude last column) ----
        KZs = KZ[:, :, :ell]  # (B, D, ell)
        QZs = QZ[:, :, :ell]  # (B, D, ell)

        # S_support = (KZ_support)^T (QZ_support): (B, ell, ell)
        S = jnp.einsum("bld,bdm->blm", jnp.swapaxes(KZs, 1, 2), QZs)

        # Now only update the query column using support-built S:
        # We need the column-vector of interactions between supports and the query.
        # Use QZ_query against KZ_support:
        QZq = QZ[:, :, ell:ell+1]  # (B, D, 1)
        s_q = jnp.einsum("bld,bdm->blm", jnp.swapaxes(KZs, 1, 2), QZq)  # (B, ell, 1)

        # VZ_support times s_q gives (B, D, 1); add to Z_query
        VZs = VZ[:, :, :ell]  # (B, D, ell)
        delta_q = (1.0 / ell) * jnp.einsum("bdl,blm->bdm", VZs, s_q)  # (B, D, 1)

        Zq = Z[:, :, ell:ell+1]  # (B, D, 1)
        Aq = Zq + delta_q        # (B, D, 1)

        # return full A where only query column updated (supports unchanged)
        A = Z.at[:, :, ell:ell+1].set(Aq)
        return jnp.swapaxes(A, 1, 2)

        # # (KZ)^T (QZ): (B, L, L)
        # KTQ = jnp.einsum("bld,bdm->blm", jnp.swapaxes(KZ, 1, 2), QZ) 
        # # A = Z + VZ (KZ)^T (QZ) / ell: (B, D, L)
        # A = Z + (1.0 / ell) * jnp.einsum("bdl,blm->bdm", VZ, KTQ)

        # # Return to convention: (B, L, D)
        # return jnp.swapaxes(A, 1, 2)

class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = True):
        """
        inputs: (B, L, d+1) 
        Output:
          - if return_final_logits_only and n_out==1: (B,)
          - else: (B, n_out) or (B, L, n_out) depending on flags
        """
        config = self.config
        y = inputs

        if config.pure_linear_self_att:
            y = LinearSelfAttentionBlock(config=config)(y)
            return y[:,-1,-1]
        
        else:
            # Continuous "embedding": project to hidden width
            if config.use_input_projection:
                y = nn.Dense(features=config.n_hidden)(y)  # (B, L, Hidden)

            # All-to-all attention: no causal mask, no positional encodings
            decoder_mask = None

            for _ in range(config.n_layers):
                y = TransformerBlock(config=config)(y,decoder_mask=decoder_mask,deterministic=deterministic)

            # Final normalization + regression head
            y = nn.LayerNorm()(y)
            logits = nn.Dense(config.n_out)(y)  # (B, L, n_out)

            if config.return_final_logits_only:
                logits = logits[:, -1, :]  # (B, n_out)
                if config.n_out == 1:
                    logits = logits[:, 0]  # (B,)

            return logits