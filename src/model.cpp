#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>

#include "model.h"
#include "matrix.h"
#include "writer.h"
#include "listener.h"
#include "data.h"
#include "cblas.h"

extern std::unordered_map<std::string, int> phonemeDict;
extern std::unordered_map<int, std::string> indexToPhoneme;
extern Listener gListener;
extern Writer gWriter;

using namespace echo;

namespace training {

matrix createMask(int seq_len) {
    matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            // Allow attention only to positions <= i.
            mask(i, j) = (j <= i) ? 0.0 : -std::numeric_limits<double>::infinity();
        }
    }
    return mask;
}

// Model constructor: initializes the transformer with the given hyperparameters.
Model::Model(int numEncoderLayers, int numDecoderLayers, int d_model, int numHeads, int d_ff, int vocab_size)
    : transformer(numEncoderLayers, numDecoderLayers, d_model, numHeads, d_ff, vocab_size) {}

Model::Model(int numEncoderLayers, int numDecoderLayers, int d_model, int numHeads, int d_ff, int vocab_size, const std::string &weightFile)
    : transformer(numEncoderLayers, numDecoderLayers, d_model, numHeads, d_ff, vocab_size) {
    // Load the transformer weights from file.
    transformer.deserializeWeights(weightFile);
}

matrix Model::forwardPass(const matrix &encoderInput, const matrix &decoderInput, const matrix &mask) {
    return transformer.forward(encoderInput, decoderInput, mask);
}

void Model::pushSnippet(const std::vector<BYTE> &snippet, int seq_len, int feature_dim) {
    // Ensure snippet size is a multiple of 2.
    if (snippet.size() % 2 != 0) {
        std::cerr << "Snippet size (" << snippet.size() 
                  << ") is not a multiple of 2, cannot convert to int16_t.\n";
        return;
    }
    // Convert BYTE snippet to int16_t vector.
    std::vector<int16_t> snippet16(snippet.size() / 2);
    std::memcpy(snippet16.data(), snippet.data(), snippet.size());

    size_t block_size = seq_len * feature_dim;
    size_t num_full_blocks = snippet16.size() / block_size;
    size_t remainder = snippet16.size() % block_size;
    size_t total_blocks = num_full_blocks + (remainder > 0 ? 1 : 0);

    for (size_t b = 0; b < total_blocks; ++b) {
        std::vector<int16_t> blockData;
        blockData.reserve(block_size);

        size_t start_idx = b * block_size;
        size_t end_idx = start_idx + block_size;
        if (end_idx > snippet16.size()) {
            blockData.insert(blockData.end(), snippet16.begin() + start_idx, snippet16.end());
            // Pad with zeros.
            blockData.resize(block_size, 0);
        } else {
            blockData.insert(blockData.end(), snippet16.begin() + start_idx, snippet16.begin() + end_idx);
        }

        // Create a matrix from blockData.
        matrix inputMatrix = createMatrixFromPCM(blockData, seq_len, feature_dim);
        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            inputBuffer.push(inputMatrix);
        }
        bufferCV.notify_one();
    }
}

void Model::startPredictor() {
    predictorRunning = true;
    grabberRunning = true;
    grabberThread = std::thread(&Model::grabber, this);
    predictorThread = std::thread(&Model::predictor, this);
}

void Model::stopPredictor() {
    {
        std::lock_guard<std::mutex> lock(bufferMutex);
        predictorRunning = false;
        grabberRunning = false;
    }
    bufferCV.notify_all();
    if (predictorThread.joinable())
        predictorThread.join();
    if (grabberThread.joinable())
        grabberThread.join();
}

void Model::grabber(){
    while (grabberRunning) {
        std::vector<BYTE> snippet = gListener.getNextAudioSnippet();
        if (!snippet.empty()) {
            this->pushSnippet(snippet, seq_len, feature_dim);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void Model::predictor() {
    while (predictorRunning) {
        matrix inputMatrix(seq_len, feature_dim);
        {
            std::unique_lock<std::mutex> lock(bufferMutex);
            bufferCV.wait(lock, [this] { return !inputBuffer.empty() || !predictorRunning; });
            if (!predictorRunning && inputBuffer.empty())
                break;
            inputMatrix = inputBuffer.front();
            inputBuffer.pop();
        }


        // Ignore samples with no audio content.
        if (inputMatrix.abs_average() <= 0.4) {
            continue;
        }

        // Create a deep copy for the decoder input.
        matrix decoderMatrix = inputMatrix.shift(1);
        matrix mask = createMask(inputMatrix.rows);

        matrix logits = forwardPass(inputMatrix, decoderMatrix, mask);
        if (logits.rows > 0) {
            int vocabSize = logits.cols;
            int bestIndex = 0;
            float bestScore = logits(0, 0);
            for (int j = 1; j < vocabSize; j++) {
                if (logits(0, j) > bestScore) {
                    bestScore = logits(0, j);
                    bestIndex = j;
                }
            }
            std::string predictedPhoneme = (indexToPhoneme.find(bestIndex) != indexToPhoneme.end())
                                            ? indexToPhoneme[bestIndex]
                                            : "Unknown phoneme";
            // Enqueue the predicted phoneme via the writer.
            gWriter.enqueueText(predictedPhoneme);
        } else {
            std::cout << "Model predictor: no logits computed." << std::endl;
        }
    }
}

// ===== Loss Functions =====
double Model::crossEntropyLoss(const matrix &logits, int trueIndex) {
    matrix probs = logits.softmax();
    double epsilon = 1e-6;
    return -std::log(probs(0, trueIndex) + epsilon);
}

matrix Model::crossEntropyLossGrad(const matrix &logits, int trueIndex) {
    matrix probs = logits.softmax();
    // Subtract 1 from the true class probability.
    probs(0, trueIndex) -= 1.0;
    return probs;
}

// ===== Matrix Multiplication Backward =====
Model::MatrixGrads Model::matmulBackward(const matrix &A, const matrix &B, const matrix &dC) {
    MatrixGrads grads;
    // dA = dC * B^T, dB = A^T * dC.
    grads.dA = dC * B.transpose();
    grads.dB = A.transpose() * dC;
    return grads;
}

// ===== ReLU Backward =====
matrix Model::reluBackward(const matrix &input, const matrix &dOut) {
    matrix dInput(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++){
        for (int j = 0; j < input.cols; j++){
            dInput(i,j) = (input(i,j) > 0) ? dOut(i,j) : 0.0;
        }
    }
    return dInput;
}

// ===== SGD Update =====
void Model::sgdUpdate(matrix &param, const matrix &grad, double learningRate) {
    int n = param.rows * param.cols;
    // cblas_daxpy performs: Y := a * X + Y. Here a = -learningRate.
    // std::cout << "Parameters: " << std::endl;
    // print_matrix(param);
    // std::cout << "Gradients: " << std::endl;
    // print_matrix(grad);
    cblas_daxpy(n, -learningRate, grad.data.data(), 1, param.data.data(), 1);
    // std::cout << "Result:" << std::endl;
    // print_matrix(param);
}

// ===== Layer Normalization Backward =====
// layerNormBackward: Given the original input (used in layerNorm),
// the upstream gradient dOut, and the scaling parameter gamma,
// recompute the mean and variance and return the gradient with respect to input.
matrix Model::layerNormBackward(const matrix &input, const matrix &dOut, const matrix &gamma) {
    matrix dInput(input.rows, input.cols);
    const double epsilon = 1e-5;
    for (int i = 0; i < input.rows; i++){
        int d = input.cols;
        double sum = 0.0;
        for (int j = 0; j < d; j++){
            sum += input(i,j);
        }
        double mean = sum / d;
        double var_sum = 0.0;
        for (int j = 0; j < d; j++){
            double diff = input(i,j) - mean;
            var_sum += diff * diff;
        }
        double variance = var_sum / d;
        double denom = std::sqrt(variance + epsilon);
        double sum_dOut = 0.0, sum_dOut_norm = 0.0;
        for (int j = 0; j < d; j++){
            double norm_val = (input(i,j) - mean) / denom;
            sum_dOut += dOut(i,j) * gamma(0,j);
            sum_dOut_norm += dOut(i,j) * gamma(0,j) * norm_val;
        }
        for (int j = 0; j < d; j++){
            double norm_val = (input(i,j) - mean) / denom;
            dInput(i,j) = (1.0 / denom) * ( dOut(i,j) * gamma(0,j)
                            - (sum_dOut / d)
                            - norm_val * (sum_dOut_norm / d));
        }
    }
    return dInput;
}

// ===== Feed-Forward Network Backward =====
Model::FeedForwardGrads Model::feedForwardBackward(
        const matrix &input,
        const matrix &W1, const matrix &W2,
        const matrix &gamma, const matrix &beta,
        const matrix &dLoss,
        const matrix &cached_hidden,         // = input * W1
        const matrix &cached_hidden_prime,     // = cached_hidden.apply(relu)
        const matrix &cached_ff,               // = cached_hidden_prime * W2
        const matrix &cached_residual          // = input + cached_ff) 
    ){
    FeedForwardGrads grads;
    matrix dResidual = layerNormBackward(cached_residual, dLoss, gamma);
    matrix dFF = dResidual;
    MatrixGrads mg2 = matmulBackward(cached_hidden_prime, W2, dFF);
    grads.dW2 = mg2.dB;
    
    matrix dHidden_prime = mg2.dA;
    matrix dHidden = reluBackward(cached_hidden, dHidden_prime);
    
    MatrixGrads mg1 = matmulBackward(input, W1, dHidden);
    grads.dW1 = mg1.dB;
    
    grads.dInput = mg1.dA + dResidual;
    
    return grads;
}

// ===== Multi-Head Attention Helpers =====
matrix Model::extractHead(const matrix &concatenated, int headIndex, int head_dim) {
    int seq_len = concatenated.rows;
    matrix head(seq_len, head_dim);
    for (int i = 0; i < seq_len; i++){
        for (int j = 0; j < head_dim; j++){
            head(i,j) = concatenated(i, headIndex * head_dim + j);
        }
    }
    return head;
}

matrix Model::softmaxBackward(const matrix &softmax_out, const matrix &dOut) {
    matrix dScores(softmax_out.rows, softmax_out.cols);
    for (int i = 0; i < softmax_out.rows; i++){
        double dot = 0.0;
        for (int j = 0; j < softmax_out.cols; j++){
            dot += dOut(i,j) * softmax_out(i,j);
        }
        for (int j = 0; j < softmax_out.cols; j++){
            dScores(i,j) = softmax_out(i,j) * (dOut(i,j) - dot);
        }
    }
    return dScores;
}

Model::MultiHeadAttentionGrads Model::multiHeadAttentionBackward(
    const std::vector<matrix> &W_qs, const std::vector<matrix> &W_ks,
    const std::vector<matrix> &W_vs, const matrix &W_o,
    const matrix &input,
    const matrix &dOutput)
{
    MultiHeadAttentionGrads grads;
    int num_heads = W_qs.size();
    int head_dim = W_qs[0].cols;
    grads.dW_q.resize(num_heads, matrix(W_qs[0].rows, W_qs[0].cols));
    grads.dW_k.resize(num_heads, matrix(W_ks[0].rows, W_ks[0].cols));
    grads.dW_v.resize(num_heads, matrix(W_vs[0].rows, W_vs[0].cols));
    grads.dW_o = matrix(W_o.rows, W_o.cols);
    grads.dInput = matrix(input.rows, input.cols);

    // Recompute concatenated multi-head outputs:
    matrix concatenated(input.rows, num_heads * head_dim);
    std::vector<matrix> Qs, Ks, Vs, attn_weights;
    for (int h = 0; h < num_heads; h++){
        matrix Q = input * W_qs[h];
        matrix K = input * W_ks[h];
        matrix V = input * W_vs[h];
        Qs.push_back(Q);
        Ks.push_back(K);
        Vs.push_back(V);
        matrix scores = Q * K.transpose();
        scores = scores / std::sqrt(static_cast<double>(head_dim));
        matrix attn = scores.softmax();
        attn_weights.push_back(attn);
        matrix head_output = attn * V;
        for (int i = 0; i < input.rows; i++){
            for (int j = 0; j < head_dim; j++){
                concatenated(i, h * head_dim + j) = head_output(i,j);
            }
        }
    }
    matrix dConcatenated = dOutput * W_o.transpose();
    grads.dW_o = concatenated.transpose() * dOutput;
    // Backprop per head.
    for (int h = 0; h < num_heads; h++){
        matrix dHeadOutput = extractHead(dConcatenated, h, head_dim);
        MatrixGrads mg_attn = matmulBackward(attn_weights[h], Vs[h], dHeadOutput);
        matrix dAttn = mg_attn.dA;
        matrix dV = mg_attn.dB;
        matrix dScores = softmaxBackward(attn_weights[h], dAttn);
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        matrix dQ = (dScores * Ks[h]) * scale;
        matrix dK = (dScores.transpose() * Qs[h]) * scale;
        MatrixGrads mg_q = matmulBackward(input, W_qs[h], dQ);
        grads.dW_q[h] = mg_q.dB;
        grads.dInput = grads.dInput + mg_q.dA;
        MatrixGrads mg_k = matmulBackward(input, W_ks[h], dK);
        grads.dW_k[h] = mg_k.dB;
        grads.dInput = grads.dInput + mg_k.dA;
        MatrixGrads mg_v = matmulBackward(input, W_vs[h], dV);
        grads.dW_v[h] = mg_v.dB;
        grads.dInput = grads.dInput + mg_v.dA;
    }
    return grads;
}

// ===== Encoder Block Backward =====
Model::EncoderBlockGrads Model::encoderBlockBackward(
    const matrix &blockInput,
    const matrix &dOut,
    int layerIndex,  // pass the layer index to access the correct cached values
    const std::vector<matrix> &W_qs, const std::vector<matrix> &W_ks,
    const std::vector<matrix> &W_vs, const matrix &W_o,
    const matrix &W1, const matrix &W2,
    const matrix &gamma1, const matrix &gamma2)
{
    EncoderBlockGrads grads;

    // Retrieve cached values from the forward pass.
    matrix attn = transformer.forwardCache.encoderAttn[layerIndex];
    matrix u = transformer.forwardCache.encoderLN1[layerIndex];
    matrix ff = transformer.forwardCache.encoderFF[layerIndex];
    // matrix out = transformer.forwardCache.encoderOutputs[layerIndex];  // may be used if needed

    // Use cached values to compute gradients.
    matrix dLN2 = layerNormBackward(u + ff, dOut, gamma2);

    // Pass the cached feed-forward intermediates.
    FeedForwardGrads ffGrads = feedForwardBackward(
        u, W1, W2, gamma2, matrix(1, gamma2.cols), dLN2,
        transformer.forwardCache.encoderHidden[layerIndex],      // cached_hidden
        transformer.forwardCache.encoderHiddenPrime[layerIndex],  // cached_hidden_prime
        ff,                                                      // cached_ff (already computed)
        transformer.forwardCache.encoderResidual[layerIndex]       // cached_residual (u + ff)
    );

    matrix dAfterFF = dLN2 + ffGrads.dInput;
    matrix dLN1 = layerNormBackward(blockInput + attn, dAfterFF, gamma1);
    MultiHeadAttentionGrads mhaGrads = multiHeadAttentionBackward(W_qs, W_ks, W_vs, W_o, blockInput, dLN1);
    
    grads.dW_q = mhaGrads.dW_q;
    grads.dW_k = mhaGrads.dW_k;
    grads.dW_v = mhaGrads.dW_v;
    grads.dW_o = mhaGrads.dW_o;
    grads.dW1 = ffGrads.dW1;
    grads.dW2 = ffGrads.dW2;
    grads.dInput = mhaGrads.dInput + dLN1;
    
    return grads;
}


Model::DecoderBlockGrads Model::decoderBlockBackward(
    const matrix &blockInput,
    const matrix &dOut,
    int layerIndex,  // used to index into the cache
    const std::vector<matrix> &selfW_qs, const std::vector<matrix> &selfW_ks, const std::vector<matrix> &selfW_vs,
    const matrix &W_o_mask,
    const std::vector<matrix> &crossW_qs, const std::vector<matrix> &crossW_ks, const std::vector<matrix> &crossW_vs,
    const matrix &W_o_encdec,
    const matrix &W1, const matrix &W2,
    const matrix &gamma1_mask,
    const matrix &gamma2_encdec,
    const matrix &gamma3)
{
    DecoderBlockGrads grads;

    // Retrieve cached forward-pass values.
    matrix selfAttn = transformer.forwardCache.decoderSelfAttn[layerIndex];
    matrix x1       = transformer.forwardCache.decoderLN1[layerIndex];
    matrix encdecAttn = transformer.forwardCache.decoderCrossAttn[layerIndex];
    matrix x2       = transformer.forwardCache.decoderLN2[layerIndex];
    matrix ff       = transformer.forwardCache.decoderFF[layerIndex];

    // Backprop through the final layer norm (after feed-forward).
    matrix dLN3 = layerNormBackward(x2 + ff, dOut, gamma3);
    
    // Backprop through the feed-forward network.

    // Pass cached intermediates for the feed-forward branch.
    FeedForwardGrads ffGrads = feedForwardBackward(
        x2, W1, W2, gamma3, matrix(1, gamma3.cols), dLN3,
        transformer.forwardCache.decoderHidden[layerIndex],      // cached_hidden
        transformer.forwardCache.decoderHiddenPrime[layerIndex],  // cached_hidden_prime
        ff,                                                      // cached_ff
        transformer.forwardCache.decoderResidual[layerIndex]       // cached_residual (x2 + ff)
    );

    matrix dAfterFF = dLN3 + ffGrads.dInput;
    
    // Backprop through the encoder-decoder (cross-attention) branch.
    matrix dLN2 = layerNormBackward(x1 + encdecAttn, dAfterFF, gamma2_encdec);
    MultiHeadAttentionGrads crossGrads = multiHeadAttentionBackward(
        crossW_qs, crossW_ks, crossW_vs, W_o_encdec, blockInput, dLN2);
    matrix dX1 = dLN2 + crossGrads.dInput;
    
    // Backprop through the masked self-attention branch.
    matrix dLN1 = layerNormBackward(blockInput + selfAttn, dX1, gamma1_mask);
    MultiHeadAttentionGrads selfGrads = multiHeadAttentionBackward(
        selfW_qs, selfW_ks, selfW_vs, W_o_mask, blockInput, dLN1);
    matrix dInput_total = dLN1 + selfGrads.dInput;
    
    // Aggregate the gradients.
    grads.dW_q_self   = selfGrads.dW_q;
    grads.dW_k_self   = selfGrads.dW_k;
    grads.dW_v_self   = selfGrads.dW_v;
    grads.dW_o_mask   = selfGrads.dW_o;
    grads.dW_q_cross  = crossGrads.dW_q;
    grads.dW_k_cross  = crossGrads.dW_k;
    grads.dW_v_cross  = crossGrads.dW_v;
    grads.dW_o_encdec = crossGrads.dW_o;
    grads.dW1         = ffGrads.dW1;
    grads.dW2         = ffGrads.dW2;
    grads.dInput      = dInput_total;
    
    return grads;
}

// ----- Training Loop -----
void Model::trainModel(const std::vector<Sample> &dataset, double learningRate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;
        for (const auto &sample : dataset) {
            matrix mask = createMask(sample.features.rows);
            // Forward pass 
            matrix logits = forwardPass(sample.features, sample.features.shift(1), mask);

            double loss = crossEntropyLoss(logits, sample.label);
            totalLoss += loss;
            // std::cout << "Computed Cross Entropy Loss: " << loss << std::endl;
            matrix dLogits = crossEntropyLossGrad(logits, sample.label);
            
            // Backprop through final projection.
            matrix dFinalLinear = transformer.decoderOutput.transpose() * dLogits;
            matrix dDecoderOut = dLogits * transformer.finalLinear.transpose();
            sgdUpdate(transformer.finalLinear, dFinalLinear, learningRate);
            
            // Backprop through decoder layers (assume we can re-run the forward for each block).
            matrix dDecoderInput = dDecoderOut;
            // For simplicity, assume blockInput for each decoder layer is the same as sample.features.
            // In a real implementation, you would re-run the per-layer forward pass.
            for (int i = static_cast<int>(transformer.decoderLayers.size()) - 1; i >= 0; i--) {
                std::vector<matrix> selfW_qs, selfW_ks, selfW_vs;
                for (const auto &head : transformer.decoderLayers[i].selfAttnHeads) {
                    selfW_qs.push_back(head.W_q);
                    selfW_ks.push_back(head.W_k);
                    selfW_vs.push_back(head.W_v);
                }
                std::vector<matrix> crossW_qs, crossW_ks, crossW_vs;
                for (const auto &head : transformer.decoderLayers[i].crossAttnHeads) {
                    crossW_qs.push_back(head.W_q);
                    crossW_ks.push_back(head.W_k);
                    crossW_vs.push_back(head.W_v);
                }
                // Here we assume 'blockInput' is sample.features (or re-run forward for layer i)
                matrix blockInput = sample.features;
                DecoderBlockGrads decGrads = decoderBlockBackward(
                    blockInput,
                    dDecoderInput,
                    i, // Pass the current layer index.
                    selfW_qs, selfW_ks, selfW_vs,
                    transformer.decoderLayers[i].W_o_mask,
                    crossW_qs, crossW_ks, crossW_vs,
                    transformer.decoderLayers[i].W_o_encdec,
                    transformer.decoderLayers[i].W1, transformer.decoderLayers[i].W2,
                    transformer.decoderLayers[i].gamma1_mask,
                    transformer.decoderLayers[i].gamma2_encdec,
                    transformer.decoderLayers[i].gamma3
                );

                for (size_t h = 0; h < transformer.decoderLayers[i].selfAttnHeads.size(); h++) {
                    sgdUpdate(transformer.decoderLayers[i].selfAttnHeads[h].W_q, decGrads.dW_q_self[h], learningRate);
                    sgdUpdate(transformer.decoderLayers[i].selfAttnHeads[h].W_k, decGrads.dW_k_self[h], learningRate);
                    sgdUpdate(transformer.decoderLayers[i].selfAttnHeads[h].W_v, decGrads.dW_v_self[h], learningRate);
                }
                sgdUpdate(transformer.decoderLayers[i].W_o_mask, decGrads.dW_o_mask, learningRate);
                for (size_t h = 0; h < transformer.decoderLayers[i].crossAttnHeads.size(); h++) {
                    sgdUpdate(transformer.decoderLayers[i].crossAttnHeads[h].W_q, decGrads.dW_q_cross[h], learningRate);
                    sgdUpdate(transformer.decoderLayers[i].crossAttnHeads[h].W_k, decGrads.dW_k_cross[h], learningRate);
                    sgdUpdate(transformer.decoderLayers[i].crossAttnHeads[h].W_v, decGrads.dW_v_cross[h], learningRate);
                }
                sgdUpdate(transformer.decoderLayers[i].W_o_encdec, decGrads.dW_o_encdec, learningRate);
                sgdUpdate(transformer.decoderLayers[i].W1, decGrads.dW1, learningRate);
                sgdUpdate(transformer.decoderLayers[i].W2, decGrads.dW2, learningRate);
                dDecoderInput = decGrads.dInput;
            }
            
            // Backprop through encoder layers.
            matrix dEncoderOut = dDecoderInput;
            for (int i = static_cast<int>(transformer.encoderLayers.size()) - 1; i >= 0; i--) {
                std::vector<matrix> W_qs, W_ks, W_vs;
                for (const auto &head : transformer.encoderLayers[i].heads) {
                    W_qs.push_back(head.W_q);
                    W_ks.push_back(head.W_k);
                    W_vs.push_back(head.W_v);
                }
                // Here we assume blockInput for encoder layer is sample.features.
                matrix blockInput = sample.features;
                EncoderBlockGrads encGrads = encoderBlockBackward(
                    blockInput,
                    dEncoderOut,
                    i, // Pass the current layer index.
                    W_qs, W_ks, W_vs,
                    transformer.encoderLayers[i].W_o,
                    transformer.encoderLayers[i].W1, transformer.encoderLayers[i].W2,
                    transformer.encoderLayers[i].gamma1, transformer.encoderLayers[i].gamma2
                );

                for (size_t h = 0; h < transformer.encoderLayers[i].heads.size(); h++) {
                    sgdUpdate(transformer.encoderLayers[i].heads[h].W_q, encGrads.dW_q[h], learningRate);
                    sgdUpdate(transformer.encoderLayers[i].heads[h].W_k, encGrads.dW_k[h], learningRate);
                    sgdUpdate(transformer.encoderLayers[i].heads[h].W_v, encGrads.dW_v[h], learningRate);
                }
                sgdUpdate(transformer.encoderLayers[i].W_o, encGrads.dW_o, learningRate);
                sgdUpdate(transformer.encoderLayers[i].W1, encGrads.dW1, learningRate);
                sgdUpdate(transformer.encoderLayers[i].W2, encGrads.dW2, learningRate);
                dEncoderOut = encGrads.dInput;
            }
        }
        std::cout << "Epoch " << epoch 
                  << " average loss: " << totalLoss / dataset.size() << std::endl;
        
        std::string filename = "transformer_weights_epoch_" + std::to_string(epoch) + ".dat";
        transformer.serializeWeights(filename);
    }
}

} // namespace training
