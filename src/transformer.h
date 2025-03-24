#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <stdexcept>

#include "matrix.h"

namespace training {

inline static double relu(double x) {
    return (x > 0) ? x : 0.0;
}

// The Transformer model for converting raw PCM to text.
class Transformer {
public:
    // Model hyperparameters.
    int numEncoderLayers;
    int numDecoderLayers;
    int d_model;
    int numHeads;
    int d_ff;

    // A singular attention head: defines the projection matrices for Query, Key, and Value.
    // Each head projects from the full model dimension (d_model) to a smaller head dimension.
    struct AttentionHead {
        matrix W_q; // Query projection: (d_model x head_dim)
        matrix W_k; // Key projection: (d_model x head_dim)
        matrix W_v; // Value projection: (d_model x head_dim)

        // Constructor: initializes the weight matrices.
        AttentionHead(int d_model, int head_dim);
    };

    // Encoder block.
    struct EncoderBlock {
        // Multi-head attention: each block has numHeads attention heads.
        std::vector<AttentionHead> heads;
        // Output projection that combines the concatenated head outputs.
        matrix W_o; // (d_model x d_model)

        // LayerNorm parameters for attention sub-layer.
        matrix gamma1; // (1 x d_model)
        matrix beta1;  // (1 x d_model)

        // Feed-forward sub-layer weights.
        matrix W1; // (d_model x d_ff)
        matrix W2; // (d_ff x d_model)

        // LayerNorm parameters for feed-forward sub-layer.
        matrix gamma2; // (1 x d_model)
        matrix beta2;  // (1 x d_model)

        // Constructor: creates numHeads attention heads.
        EncoderBlock(int d_model, int d_ff, int numHeads);
    };

    // Decoder block.
    struct DecoderBlock {
        // Masked self-attention heads.
        std::vector<AttentionHead> selfAttnHeads;
        matrix W_o_mask; // Output projection for masked self-attention (d_model x d_model)

        matrix gamma1_mask; // LayerNorm parameters for self-attention.
        matrix beta1_mask;

        // Encoder-decoder (cross) attention heads.
        std::vector<AttentionHead> crossAttnHeads;
        matrix W_o_encdec; // Output projection for cross-attention (d_model x d_model)

        matrix gamma2_encdec; // LayerNorm parameters for cross-attention.
        matrix beta2_encdec;

        // Feed-forward sub-layer weights.
        matrix W1; // (d_model x d_ff)
        matrix W2; // (d_ff x d_model)

        // LayerNorm parameters for feed-forward sub-layer.
        matrix gamma3; // (1 x d_model)
        matrix beta3;  // (1 x d_model)

        // Constructor: creates numHeads attention heads for both self and cross attention.
        DecoderBlock(int d_model, int d_ff, int numHeads);
    };

    // The stack of encoder and decoder layers.
    std::vector<EncoderBlock> encoderLayers;
    std::vector<DecoderBlock> decoderLayers;

    // Final linear projection from decoder output to vocabulary logits.
    matrix finalLinear; // (d_model x vocab_size)

    // Cache for decoder output
    matrix decoderOutput;  

    // Transformer constructor.
    Transformer(int numEncoderLayers, int numDecoderLayers, int d_model, int numHeads, int d_ff, int vocab_size);

    void set(matrix* dest, matrix* src);

    void serializeWeights(const std::string &filename);

    void deserializeWeights(const std::string &filename);

    // --- Forward methods (stubs) --

    // Encoder forward pass.
    matrix encode(const matrix &input);

    // Decoder forward pass; mask is used for masked self-attention.
    matrix decode(const matrix &decoderInput, const matrix &encoderOutput, const matrix &mask);

    // Full transformer forward pass.
    matrix forward(const matrix &input, const matrix &decoderInput, const matrix &mask);

    // --- Utility functions (stubs) ---

    // Multi-head attention: given a set of heads and an input,
    // computes attention over all heads and concatenates their outputs.
    matrix multiHeadAttention(const std::vector<AttentionHead>& heads, const matrix &input, const matrix &mask, const matrix &outputProjection);

    // Layer normalization.
    matrix layerNorm(const matrix &input, const matrix &gamma, const matrix &beta);

    // Feed-forward network: linear -> ReLU -> linear with residual connection & normalization.
    matrix feedForward(const matrix &input, const matrix &W1, const matrix &W2, const matrix &gamma, const matrix &beta);

    struct Cache {
        // Encoder caches per layer.
        std::vector<matrix> encoderInputs;    // Input x fed into each encoder block.
        std::vector<matrix> encoderAttn;        // The multi-head attention outputs for each encoder block.
        std::vector<matrix> encoderLN1;         // Outputs after first layer normalization for each encoder block.
        // Cached feed-forward intermediates:
        std::vector<matrix> encoderHidden;      // = input * W1 (cached in feed-forward).
        std::vector<matrix> encoderHiddenPrime; // = encoderHidden.apply(relu) (cached in feed-forward).
        std::vector<matrix> encoderFF;          // = encoderHiddenPrime * W2.
        std::vector<matrix> encoderResidual;    // = (u + encoderFF) or (input + encoderFF), as appropriate.
        std::vector<matrix> encoderOutputs;     // Final output from each encoder block.

        // Decoder caches per layer.
        std::vector<matrix> decoderInputs;      // Input x fed into each decoder block.
        std::vector<matrix> decoderSelfAttn;    // Outputs of the masked self-attention in each decoder block.
        std::vector<matrix> decoderLN1;         // Outputs after the first layer normalization in decoder blocks.
        std::vector<matrix> decoderCrossAttn;   // Outputs of the encoder-decoder (cross) attention.
        std::vector<matrix> decoderLN2;         // Outputs after the second layer normalization in decoder blocks.
        // Cached feed-forward intermediates:
        std::vector<matrix> decoderHidden;      // = input * W1 for the feed-forward branch.
        std::vector<matrix> decoderHiddenPrime; // = decoderHidden.apply(relu).
        std::vector<matrix> decoderFF;          // = decoderHiddenPrime * W2.
        std::vector<matrix> decoderResidual;    // = (x2 + decoderFF) in decoder blocks.
        std::vector<matrix> decoderOutputs;     // Final output from each decoder block.

        // Final outputs.
        matrix finalDecoderOutput;              // The output of the decoder before final projection.
        matrix finalLogits;                     // The final logits computed before softmax.

        Cache(int numEncoderLayers, int numDecoderLayers, int seq_len, int d_model, int vocab_size);
    };

    Cache forwardCache;

};

} // Namespace training

#endif // TRANSFORMER_H
