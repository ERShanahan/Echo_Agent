#ifndef TRAINING_H
#define TRAINING_H

#include <queue>
#include <condition_variable>
#include <thread>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include "transformer.h"
#include "matrix.h"
#include "listener.h"
#include "writer.h"
#include "data.h"

using namespace echo;

namespace training {

matrix createMask(int seq_len);

const double N = 1;
const double H = 2;
const double D = 256;
const double Dff = 1024;

const int seq_len = 500;
const int feature_dim = 256;

class Model {
public:
    // The transformer model that will be trained.
    Transformer transformer;

    // Constructor initializes the transformer with the given hyperparameters.
    // numEncoderLayers, numDecoderLayers: number of layers in the encoder and decoder.
    // d_model: model dimension.
    // numHeads: number of attention heads.
    // d_ff: feed-forward dimension.
    Model(int numEncoderLayers, int numDecoderLayers, int d_model, int numHeads, int d_ff, int vocab_size);

    // Constructor overload that loads transformer weights.
    Model(int numEncoderLayers, int numDecoderLayers, int d_model, int numHeads, int d_ff, int vocab_size, const std::string &weightFile);

    // Performs a forward pass using the transformer.
    // encoderInput: Input matrix for the encoder.
    // decoderInput: Input matrix for the decoder.
    // mask: Attention mask (can be an empty matrix if not used).
    // Returns: Logits matrix (before softmax) for the generated output.
    matrix forwardPass(const matrix &encoderInput, const matrix &decoderInput, const matrix &mask);

    // Push a raw audio snippet (BYTE vector) into the trainer.
    // The snippet will be converted, split into blocks, and each block pushed into the input buffer.
    void pushSnippet(const std::vector<BYTE> &snippet, int seq_len, int feature_dim);

    // Loop to grab inputMatrix from the inputBuffer.
    void grabber();

    // Starts the predictor thread.
    void startPredictor();
    // Stops the predictor thread.
    void stopPredictor();

    // Example training loop skeleton (to be integrated into your Model class):
    // For each sample:
    //   1. Run forwardPass (which must cache intermediate values).
    //   2. Compute loss using crossEntropyLoss.
    //   3. Compute dLogits via crossEntropyLossGrad.
    //   4. Run backward passes (e.g., encoderBlockBackward, multi-head attention, etc.).
    //   5. Update every parameter via sgdUpdate.
    void trainModel(const std::vector<Sample> &dataset, double learningRate, int epochs);

    // ===== Loss Functions =====

    // Compute the cross-entropy loss from logits (1 x vocab_size)
    // using the matrix::softmax method for numerical stability.
    double crossEntropyLoss(const matrix &logits, int trueIndex);

    // Compute the gradient of the cross-entropy loss with respect to logits.
    // dL/dz = softmax(logits) - one_hot(trueIndex)
    matrix crossEntropyLossGrad(const matrix &logits, int trueIndex);

    // ===== Basic Backprop Functions =====

    // Structure to hold gradients for a matrix multiplication operation.
    struct MatrixGrads {
        matrix dA;
        matrix dB;
        MatrixGrads() : dA(0, 0), dB(0, 0) {}
    };

    // Given C = A * B, compute gradients dA and dB using the chain rule.
    MatrixGrads matmulBackward(const matrix &A, const matrix &B, const matrix &dC);

    // ReLU backward: Given the original input and the gradient flowing out,
    // return dInput, where the gradient is passed only where input > 0.
    matrix reluBackward(const matrix &input, const matrix &dOut);

    // SGD update: param = param - learningRate * grad (using cblas_daxpy)
    void sgdUpdate(matrix &param, const matrix &grad, double learningRate);

    // ===== Layer Normalization =====

    // Backward pass for layer normalization.
    // Uses the cached mean and variance (from LayerNormCache) to compute the gradient with respect to the input.
    matrix layerNormBackward(const matrix &input, const matrix &dOut, const matrix &gamma);

    // ===== Feed-Forward Network =====

    struct FeedForwardGrads {
        matrix dInput; // Gradient propagated to the feed-forward block’s input.
        matrix dW1;    // Gradient for the first weight matrix.
        matrix dW2;    // Gradient for the second weight matrix.
        FeedForwardGrads() : dInput(0,0), dW1(0,0), dW2(0,0) {}
    };

    // Backward pass for the feed-forward network.
    // Assumes that the forward pass computed:
    //    hidden = input * W1,
    //    hidden_prime = relu(hidden),
    //    ff = hidden_prime * W2,
    // and then (after adding a residual connection and layer normalization)
    // the loss gradient dLoss is provided.
    FeedForwardGrads feedForwardBackward(
        const matrix &input,
        const matrix &W1, const matrix &W2,
        const matrix &gamma, const matrix &beta,
        const matrix &dLoss,
        const matrix &cached_hidden,         // = input * W1
        const matrix &cached_hidden_prime,     // = cached_hidden.apply(relu)
        const matrix &cached_ff,               // = cached_hidden_prime * W2
        const matrix &cached_residual          // = input + cached_ff) 
    );
    // ===== Multi-Head Attention =====

    struct MultiHeadAttentionGrads {
        std::vector<matrix> dW_q; // Gradients for each head’s W_q.
        std::vector<matrix> dW_k; // Gradients for each head’s W_k.
        std::vector<matrix> dW_v; // Gradients for each head’s W_v.
        matrix dW_o;            // Gradient for the output projection matrix.
        matrix dInput;          // Gradient with respect to the input of the attention.
        MultiHeadAttentionGrads() : dW_o(0,0), dInput(0,0) {}
    };

    // Helper to extract the slice for head 'headIndex' from the concatenated matrix.
    // Assumes concatenated has dimensions (seq_len x (num_heads*head_dim)).
    matrix extractHead(const matrix &concatenated, int headIndex, int head_dim);

    // Given the softmax output and the gradient flowing from the next layer,
    // compute the gradient with respect to the scores (dScores).
    // (This version assumes processing row-by-row.)
    matrix softmaxBackward(const matrix &softmax_out, const matrix &dOut);

    // Backward pass for multi-head attention.
    // W_qs, W_ks, W_vs are vectors (one per head) of the projection matrices,
    // W_o is the output projection,
    // input is the original input to attention,
    // dOutput is the gradient from the next layer (after the output projection),
    // and cache holds all cached values from the forward pass.
    MultiHeadAttentionGrads multiHeadAttentionBackward(
        const std::vector<matrix> &W_qs, const std::vector<matrix> &W_ks,
        const std::vector<matrix> &W_vs, const matrix &W_o,
        const matrix &input,
        const matrix &dOutput);

    // ===== Encoder Block Backward =====

    struct EncoderBlockGrads {
        std::vector<matrix> dW_q;
        std::vector<matrix> dW_k;
        std::vector<matrix> dW_v;
        matrix dW_o;
        matrix dW1;
        matrix dW2;
        matrix dInput; // Gradient with respect to the input of the encoder block.
        EncoderBlockGrads() : dW_o(0,0), dW1(0,0), dW2(0,0), dInput(0,0) {}
    };

    // Backward pass for an encoder block.
    // W_qs, W_ks, W_vs, W_o are the parameters for the multi-head attention,
    // W1 and W2 for the feed-forward network,
    // gamma1 and gamma2 are the layer norm scaling parameters for the two sub-layers.
    EncoderBlockGrads encoderBlockBackward(
        const matrix &input,      // Original input to the encoder block.
        const matrix &dOut,       // Upstream gradient from this block’s output.
        int layerIndex,
        const std::vector<matrix> &W_qs, const std::vector<matrix> &W_ks,
        const std::vector<matrix> &W_vs, const matrix &W_o,
        const matrix &W1, const matrix &W2,
        const matrix &gamma1, const matrix &gamma2);

    struct DecoderBlockGrads {
        // Gradients for the masked self-attention branch.
        std::vector<matrix> dW_q_self;
        std::vector<matrix> dW_k_self;
        std::vector<matrix> dW_v_self;
        matrix dW_o_mask;
        // Gradients for the encoder-decoder (cross) attention branch.
        std::vector<matrix> dW_q_cross;
        std::vector<matrix> dW_k_cross;
        std::vector<matrix> dW_v_cross;
        matrix dW_o_encdec;
        // Gradients for the feed-forward network.
        matrix dW1;
        matrix dW2;
        // Gradient to propagate to the input of this decoder block.
        matrix dInput;
        DecoderBlockGrads() : dW_o_mask(0,0), dW_o_encdec(0,0), dW1(0,0), dW2(0,0), dInput(0,0) {}
    };

    DecoderBlockGrads decoderBlockBackward(const matrix &blockInput,
        const matrix &dOut,
        int layerIndex,
        const std::vector<matrix> &selfW_qs, const std::vector<matrix> &selfW_ks, const std::vector<matrix> &selfW_vs,
        const matrix &W_o_mask,
        const std::vector<matrix> &crossW_qs, const std::vector<matrix> &crossW_ks, const std::vector<matrix> &crossW_vs,
        const matrix &W_o_encdec,
        const matrix &W1, const matrix &W2,
        const matrix &gamma1_mask,
        const matrix &gamma2_encdec,
        const matrix &gamma3);

private:
    // Predictor thread function.
    void predictor();

    // Input buffer of matrices for prediction.
    std::queue<matrix> inputBuffer;
    std::mutex bufferMutex;
    std::condition_variable bufferCV;
    std::thread predictorThread;
    std::thread grabberThread;
    bool predictorRunning;
    bool grabberRunning;
};

} // namespace training

#endif // TRAINING_H
