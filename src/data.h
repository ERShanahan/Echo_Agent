#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>

#include "matrix.h"

namespace training {

// Structure to store a phoneme segment
struct Phoneme {
    int start;          // Start index (e.g., in samples or time units)
    int end;            // End index
    std::string label;  // Phonetic label
};

// Structure to store a word segment
struct Word {
    int start;          // Start index (e.g., in samples or time units)
    int end;            // End index
    std::string text;   // Word text
};

struct PhonemeData {
    Phoneme phoneme;
    std::vector<int16_t> pcm;
};

struct WordData {
    Word word;
    std::vector<int16_t> pcm;
};

struct SentenceData {
    std::string sentence;
    std::vector<int16_t> pcm;
};

struct Sample {
    matrix features;
    int label;
    Sample() : features(0,0) {}
};

// Load the phonetic segmentation from a .phn file.
// Each line in the file is expected to have:
//   <start> <end> <phoneme>
// Example: "0 1000 sh"
std::vector<Phoneme> loadPhonemes(const std::string& filename);

// Load the word segmentation from a .wrd file.
// Each line in the file is expected to have:
//   <start> <end> <word>
// Example: "0 1500 she"
std::vector<Word> loadWords(const std::string& filename);

// Load the sentence transcription from a text file.
// This file typically contains the full sentence on one line.
std::string loadSentence(const std::string& filename);

void loadData(
    int count,
    const std::string& directory,
    std::vector<std::pair<std::string, std::vector<Phoneme>>>& phonemeFiles,
    std::vector<std::pair<std::string, std::vector<Word>>>& wordFiles,
    std::vector<std::pair<std::string, std::string>>& sentenceFiles
);

std::vector<int16_t> extractSegmentPCMNist(const std::string& nistWavFilename, int start, int end);

std::vector<int16_t> extractEntirePCM(const std::string& nistWavFilename);

void extractData(
    int count,
    const std::string& directory,
    std::vector<PhonemeData>& allPhonemeData,
    std::vector<WordData>& allWordData,
    std::vector<SentenceData>& allSentenceData
);

template <typename E>
inline void printData(const std::string& dataName, const std::vector<E>& data, int iterations){
    std::cout << "\nFirst " << iterations << " of " << dataName << " data entries:" << std::endl;
    size_t count = std::min(static_cast<size_t>(iterations), data.size());
    for (size_t i = 0; i < count; i++) {
        // Print the label or text based on the data type.
        if constexpr (std::is_same_v<E, PhonemeData>) {
            std::cout << "Phoneme: " << data[i].phoneme.label;
        } else if constexpr (std::is_same_v<E, WordData>) {
            std::cout << "Word: " << data[i].word.text;
        } else if constexpr (std::is_same_v<E, SentenceData>) {
            std::cout << "Sentence: " << data[i].sentence;
        } else {
            std::cout << dataName;
        }
        
        std::cout << ", PCM Samples: ";
        size_t sampleCount = std::min(static_cast<size_t>(10), data[i].pcm.size());
        for (size_t j = 0; j < sampleCount; j++) {
            std::cout << data[i].pcm[j] << " ";
        }
        std::cout << std::endl;
    }
}

matrix createMatrixFromPCM(const std::vector<int16_t>& pcm, int seq_len, int feature_dim);

std::unordered_map<std::string, int> buildPhonemeDictionary(const std::vector<PhonemeData>& phonemeData);

Sample createSample(const PhonemeData &data, int seq_len, int feature_dim);

std::vector<Sample> vectorize(std::vector<PhonemeData> &dataset);

std::pair<std::vector<Sample>, std::vector<Sample>> splitTrainTestSamples(const std::vector<Sample>& samples, double testRatio);

} // Namespace Training

#endif // DATA_H
