#include <stdexcept>
#include <cmath>
#include <fstream>

#include "transformer.h"
#include "model.h"

namespace training {

// ----- AttentionHead -----
Transformer::AttentionHead::AttentionHead(int d_model, int head_dim)
    : W_q(d_model, head_dim), W_k(d_model, head_dim), W_v(d_model, head_dim)
{
    W_q.randomize(0, 1);
    W_k.randomize(0, 1);
    W_v.randomize(0, 1);
}

// ----- EncoderBlock -----
Transformer::EncoderBlock::EncoderBlock(int d_model, int d_ff, int numHeads)
    : W_o(d_model, d_model),
      gamma1(1, d_model), beta1(1, d_model),
      W1(d_model, d_ff), W2(d_ff, d_model),
      gamma2(1, d_model), beta2(1, d_model)
{
    int head_dim = d_model / numHeads;
    
    // Initialize gamma1 and gamma2 to ones, beta1 and beta2 to zeros.
    for (int j = 0; j < d_model; j++) {
        gamma1(0, j) = 1.0;
        beta1(0, j) = 0.0;
        gamma2(0, j) = 1.0;
        beta2(0, j) = 0.0;
    }
    
    // Create attention heads.
    for (int i = 0; i < numHeads; i++) {
        heads.push_back(AttentionHead(d_model, head_dim));
    }
    
    // Initialize projection and feed-forward weights.
    W_o.randomize(0, 1);
    W1.randomize(0, 1);
    W2.randomize(0, 1);
}

// ----- DecoderBlock -----
Transformer::DecoderBlock::DecoderBlock(int d_model, int d_ff, int numHeads)
    : W_o_mask(d_model, d_model),
      gamma1_mask(1, d_model), beta1_mask(1, d_model),
      W_o_encdec(d_model, d_model),
      gamma2_encdec(1, d_model), beta2_encdec(1, d_model),
      W1(d_model, d_ff), W2(d_ff, d_model),
      gamma3(1, d_model), beta3(1, d_model)
{
    int head_dim = d_model / numHeads;
    
    // Initialize layer normalization parameters for masked self-attention,
    // encoder-decoder attention, and feed-forward sub-layer.
    for (int j = 0; j < d_model; j++) {
        gamma1_mask(0, j) = 1.0;
        beta1_mask(0, j) = 0.0;
        gamma2_encdec(0, j) = 1.0;
        beta2_encdec(0, j) = 0.0;
        gamma3(0, j) = 1.0;
        beta3(0, j) = 0.0;
    }
    
    // Create masked self-attention heads.
    for (int i = 0; i < numHeads; i++) {
        selfAttnHeads.push_back(AttentionHead(d_model, head_dim));
    }
    // Create encoder-decoder cross-attention heads.
    for (int i = 0; i < numHeads; i++) {
        crossAttnHeads.push_back(AttentionHead(d_model, head_dim));
    }
    
    // Initialize projection and feed-forward weights.
    W_o_mask.randomize(0, 1);
    W_o_encdec.randomize(0, 1);
    W1.randomize(0, 1);
    W2.randomize(0, 1);
}

// ----- Transformer Constructor -----
Transformer::Transformer(int numEncoderLayers, int numDecoderLayers, int d_model, int numHeads, int d_ff, int vocab_size)
    : numEncoderLayers(numEncoderLayers), numDecoderLayers(numDecoderLayers),
      d_model(d_model), numHeads(numHeads), d_ff(d_ff),
      finalLinear(d_model, vocab_size),
      decoderOutput(seq_len, d_model), // Needs to be set to seq_len X d_model
      forwardCache(numEncoderLayers, numDecoderLayers, seq_len, d_model, vocab_size)
{
    // Initialize the final projection.
    finalLinear.randomize(0, 1);
    
    // Build encoder layers.
    for (int i = 0; i < numEncoderLayers; i++) {
        encoderLayers.push_back(EncoderBlock(d_model, d_ff, numHeads));
    }
    // Build decoder layers.
    for (int i = 0; i < numDecoderLayers; i++) {
        decoderLayers.push_back(DecoderBlock(d_model, d_ff, numHeads));
    }
}

void Transformer::set(matrix* dest, matrix* src){
    dest->copy(*src);
}

void Transformer::serializeWeights(const std::string &filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Serialize the final projection matrix.
    finalLinear.serializeMatrix(out);
    
    // Write the number of encoder layers.
    out << encoderLayers.size() << "\n";
    for (const auto &block : encoderLayers) {
        block.W_o.serializeMatrix(out);
        block.gamma1.serializeMatrix(out);
        block.beta1.serializeMatrix(out);
        block.W1.serializeMatrix(out);
        block.W2.serializeMatrix(out);
        block.gamma2.serializeMatrix(out);
        block.beta2.serializeMatrix(out);
        
        // Write number of attention heads.
        out << block.heads.size() << "\n";
        for (const auto &head : block.heads) {
            head.W_q.serializeMatrix(out);
            head.W_k.serializeMatrix(out);
            head.W_v.serializeMatrix(out);
        }
    }
    
    // Write the number of decoder layers.
    out << decoderLayers.size() << "\n";
    for (const auto &block : decoderLayers) {
        block.W_o_mask.serializeMatrix(out);
        block.gamma1_mask.serializeMatrix(out);
        block.beta1_mask.serializeMatrix(out);
        block.W_o_encdec.serializeMatrix(out);
        block.gamma2_encdec.serializeMatrix(out);
        block.beta2_encdec.serializeMatrix(out);
        block.W1.serializeMatrix(out);
        block.W2.serializeMatrix(out);
        block.gamma3.serializeMatrix(out);
        block.beta3.serializeMatrix(out);
        
        // Write number of self-attention heads.
        out << block.selfAttnHeads.size() << "\n";
        for (const auto &head : block.selfAttnHeads) {
            head.W_q.serializeMatrix(out);
            head.W_k.serializeMatrix(out);
            head.W_v.serializeMatrix(out);
        }
        
        // Write number of cross-attention heads.
        out << block.crossAttnHeads.size() << "\n";
        for (const auto &head : block.crossAttnHeads) {
            head.W_q.serializeMatrix(out);
            head.W_k.serializeMatrix(out);
            head.W_v.serializeMatrix(out);
        }
    }
    out.close();
}

void Transformer::deserializeWeights(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    // Helper lambda to read a matrix from the stream and copy its contents
    // into the destination matrix using the set() method.
    auto readAndSet = [this, &in](matrix* dest) {
        int r, c;
        in >> r >> c;
        matrix tmp(r, c);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                in >> tmp(i, j);
            }
        }
        set(dest, &tmp);
    };
    
    // Deserialize finalLinear.
    readAndSet(&finalLinear);
    
    // Read the number of encoder layers.
    size_t encoderCount;
    in >> encoderCount;
    if (encoderCount != encoderLayers.size()) {
        throw std::runtime_error("Mismatch in encoder layers count");
    }
    for (auto &block : encoderLayers) {
        readAndSet(&block.W_o);
        readAndSet(&block.gamma1);
        readAndSet(&block.beta1);
        readAndSet(&block.W1);
        readAndSet(&block.W2);
        readAndSet(&block.gamma2);
        readAndSet(&block.beta2);
        
        size_t headsCount;
        in >> headsCount;
        if (headsCount != block.heads.size()) {
            throw std::runtime_error("Mismatch in encoder attention heads count");
        }
        for (auto &head : block.heads) {
            readAndSet(&head.W_q);
            readAndSet(&head.W_k);
            readAndSet(&head.W_v);
        }
    }
    
    // Read the number of decoder layers.
    size_t decoderCount;
    in >> decoderCount;
    if (decoderCount != decoderLayers.size()) {
        throw std::runtime_error("Mismatch in decoder layers count");
    }
    for (auto &block : decoderLayers) {
        readAndSet(&block.W_o_mask);
        readAndSet(&block.gamma1_mask);
        readAndSet(&block.beta1_mask);
        readAndSet(&block.W_o_encdec);
        readAndSet(&block.gamma2_encdec);
        readAndSet(&block.beta2_encdec);
        readAndSet(&block.W1);
        readAndSet(&block.W2);
        readAndSet(&block.gamma3);
        readAndSet(&block.beta3);
        
        size_t selfHeadsCount;
        in >> selfHeadsCount;
        if (selfHeadsCount != block.selfAttnHeads.size()) {
            throw std::runtime_error("Mismatch in decoder self-attention heads count");
        }
        for (auto &head : block.selfAttnHeads) {
            readAndSet(&head.W_q);
            readAndSet(&head.W_k);
            readAndSet(&head.W_v);
        }
        
        size_t crossHeadsCount;
        in >> crossHeadsCount;
        if (crossHeadsCount != block.crossAttnHeads.size()) {
            throw std::runtime_error("Mismatch in decoder cross-attention heads count");
        }
        for (auto &head : block.crossAttnHeads) {
            readAndSet(&head.W_q);
            readAndSet(&head.W_k);
            readAndSet(&head.W_v);
        }
    }
    
    in.close();
}

Transformer::Cache::Cache(int numEncoderLayers, int numDecoderLayers, int seq_len, int d_model, int vocab_size)
    : finalDecoderOutput(seq_len, d_model), finalLogits(seq_len, vocab_size) {

    // For each encoder layer, initialize matrices with (seq_len x d_model).
    encoderInputs.resize(numEncoderLayers, matrix(seq_len, d_model));
    encoderAttn.resize(numEncoderLayers, matrix(seq_len, d_model));
    encoderLN1.resize(numEncoderLayers, matrix(seq_len, d_model));
    encoderFF.resize(numEncoderLayers, matrix(seq_len, d_model));
    encoderOutputs.resize(numEncoderLayers, matrix(seq_len, d_model));

    // For each decoder layer, initialize matrices with (seq_len x d_model).
    decoderInputs.resize(numDecoderLayers, matrix(seq_len, d_model));
    decoderSelfAttn.resize(numDecoderLayers, matrix(seq_len, d_model));
    decoderLN1.resize(numDecoderLayers, matrix(seq_len, d_model));
    decoderCrossAttn.resize(numDecoderLayers, matrix(seq_len, d_model));
    decoderLN2.resize(numDecoderLayers, matrix(seq_len, d_model));
    decoderFF.resize(numDecoderLayers, matrix(seq_len, d_model));
    decoderOutputs.resize(numDecoderLayers, matrix(seq_len, d_model));
}

// ----- Utility Functions -----

// Multi-head attention
matrix Transformer::multiHeadAttention(const std::vector<AttentionHead>& heads, const matrix &input, const matrix &mask, const matrix &outputProjection) {
    // Input shape: (seq_len x d_model).
    // Each head projects d_model -> head_dim, where head_dim = d_model / numHeads.
    int seq_len = input.rows;
    int num_heads = heads.size();
    int head_dim = heads[0].W_q.cols;
    
    // The concatenated result will have shape (seq_len x (num_heads * head_dim)).
    matrix concatenated(seq_len, num_heads * head_dim);

    // Process each head.
    for (int h = 0; h < num_heads; h++) {
        const AttentionHead &head = heads[h];
        
        // Compute linear projections: Q, K, V.
        matrix Q = input * head.W_q;  // (seq_len x head_dim)
        matrix K = input * head.W_k;  // (seq_len x head_dim)
        matrix V = input * head.W_v;  // (seq_len x head_dim)
        
        // Compute attention scores: scores = Q * K^T  (shape: seq_len x seq_len)
        matrix scores = Q * K.transpose();
        
        // Scale scores by 1/sqrt(head_dim)
        scores = scores / std::sqrt(static_cast<double>(head_dim));
        
        // If a mask is provided (non-empty), add it element-wise.
        if (mask.rows == seq_len && mask.cols == seq_len){
            scores = scores + mask;
        }
        
        // Apply softmax row-wise to obtain attention weights.
        matrix attn_weights = scores.softmax();  // (seq_len x seq_len)
        
        // Compute head output: head_output = attn_weights * V  (shape: seq_len x head_dim)
        matrix head_output = attn_weights * V;
        
        // Place head_output into the corresponding columns of the concatenated matrix.
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                concatenated(i, h * head_dim + j) = head_output(i, j);
            }
        }
    }
    // Return the concatenated multi-head outputs
    matrix output = concatenated * outputProjection;
    return output;
}

// Layer normalization
matrix Transformer::layerNorm(const matrix &input, const matrix &gamma, const matrix &beta) {
    matrix result = input;

    const double epsilon = 1e-5;
    int seq_len = input.rows;
    int d_model = input.cols;

    for (int i = 0; i < seq_len; ++i) {
        // Compute mean for row i.
        double sum = 0.0;
        for (int j = 0; j < d_model; ++j) {
            sum += input(i, j);
        }
        double mean = sum / d_model;
        
        // Compute variance for row i.
        double var_sum = 0.0;
        for (int j = 0; j < d_model; ++j) {
            double diff = input(i, j) - mean;
            var_sum += diff * diff;
        }
        double variance = var_sum / d_model;
        double denom = std::sqrt(variance + epsilon);
        
        // Normalize and apply gamma and beta.
        for (int j = 0; j < d_model; ++j) {
            result(i, j) = gamma(0, j) * ((input(i, j) - mean) / denom) + beta(0, j);
        }
    }
    return result;
}

// Feed-forward network
matrix Transformer::feedForward(const matrix &input, const matrix &W1, const matrix &W2, const matrix &gamma, const matrix &beta) {
    matrix hidden = input * W1;

    hidden = hidden.apply(relu);

    matrix ff = hidden*W2;
    matrix residual = input + ff;

    matrix output = layerNorm(residual, gamma, beta);
    return output;
}

// ----- Encoder Forward Pass -----
matrix Transformer::encode(const matrix &input) {

    forwardCache.encoderInputs.clear();
    forwardCache.encoderAttn.clear();
    forwardCache.encoderLN1.clear();
    forwardCache.encoderHidden.clear();
    forwardCache.encoderHiddenPrime.clear();
    forwardCache.encoderFF.clear();
    forwardCache.encoderResidual.clear();
    forwardCache.encoderOutputs.clear();

    matrix x = input;
    for (int i = 0; i < numEncoderLayers; i++) {
        EncoderBlock &block = encoderLayers[i];
        // Cache the input
        forwardCache.encoderInputs.push_back(x);

        // --- Multi-Head Attention ---
        matrix attn = multiHeadAttention(block.heads, x, matrix(0, 0), block.W_o); // No mask.
        forwardCache.encoderAttn.push_back(attn);

        // Apply residual and layer norm
        matrix u = layerNorm(x + attn, block.gamma1, block.beta1);
        forwardCache.encoderLN1.push_back(u);

        // --- Feed-Forward Branch ---
        // Compute the hidden layer: hidden = u * W1
        matrix hidden = u * block.W1;
        // Apply ReLU
        matrix hiddenPrime = hidden.apply(relu);
        // Compute the feed-forward output: ff = hiddenPrime * W2
        matrix ff = hiddenPrime * block.W2;
        // Compute the residual: residual = u + ff
        matrix residual = u + ff;

        // Cache feed-forward
        forwardCache.encoderHidden.push_back(hidden);
        forwardCache.encoderHiddenPrime.push_back(hiddenPrime);
        forwardCache.encoderFF.push_back(ff);
        forwardCache.encoderResidual.push_back(residual);

        // Final layer norm
        x = layerNorm(residual, block.gamma2, block.beta2);
        forwardCache.encoderOutputs.push_back(x);
    }
    return x;
}

// ----- Decoder Forward Pass -----
matrix Transformer::decode(const matrix &decoderInput, const matrix &encoderOutput, const matrix &mask) {

    forwardCache.decoderInputs.clear();
    forwardCache.decoderSelfAttn.clear();
    forwardCache.decoderLN1.clear();
    forwardCache.decoderCrossAttn.clear();
    forwardCache.decoderLN2.clear();
    forwardCache.decoderHidden.clear();
    forwardCache.decoderHiddenPrime.clear();
    forwardCache.decoderFF.clear();
    forwardCache.decoderResidual.clear();
    forwardCache.decoderOutputs.clear();

    matrix x = decoderInput;
    for (int i = 0; i < numDecoderLayers; i++) {
        DecoderBlock &block = decoderLayers[i];
        // Cache the input
        forwardCache.decoderInputs.push_back(x);

        // --- Masked Self-Attention ---
        matrix selfAttn = multiHeadAttention(block.selfAttnHeads, x, mask, block.W_o_mask);
        forwardCache.decoderSelfAttn.push_back(selfAttn);
        // Residual connection + layer norm
        x = layerNorm(x + selfAttn, block.gamma1_mask, block.beta1_mask);
        forwardCache.decoderLN1.push_back(x);

        // --- Encoder-Decoder (Cross) Attention ---
        matrix encdecAttn = multiHeadAttention(block.crossAttnHeads, encoderOutput, matrix(0, 0), block.W_o_encdec);
        forwardCache.decoderCrossAttn.push_back(encdecAttn);
        // Residual connection + layer norm
        x = layerNorm(x + encdecAttn, block.gamma2_encdec, block.beta2_encdec);
        forwardCache.decoderLN2.push_back(x);

        // --- Feed-Forward Branch ---
        // Compute the hidden layer: hidden = x * W1
        matrix hidden = x * block.W1;
        // Apply ReLU
        matrix hiddenPrime = hidden.apply(relu);
        // Compute the feed-forward output: ff = hiddenPrime * W2
        matrix ff = hiddenPrime * block.W2;
        // Compute the residual: residual = x + ff
        matrix residual = x + ff;

        // Cache feed-forward
        forwardCache.decoderHidden.push_back(hidden);
        forwardCache.decoderHiddenPrime.push_back(hiddenPrime);
        forwardCache.decoderFF.push_back(ff);
        forwardCache.decoderResidual.push_back(residual);

        // Final layer norm
        x = layerNorm(residual, block.gamma3, block.beta3);
        forwardCache.decoderOutputs.push_back(x);
    }
    forwardCache.finalDecoderOutput = x;
    return x;
}

// ----- Full Transformer Forward Pass -----
matrix Transformer::forward(const matrix &input, const matrix &decoderInput, const matrix &mask) {
    matrix encoderOutput = encode(input);
    decoderOutput = decode(decoderInput, encoderOutput, mask);
    matrix logits = decoderOutput * finalLinear;
    forwardCache.finalLogits = logits;
    return logits.softmax();
}

} // Namespace training
