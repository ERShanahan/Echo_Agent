#include "data.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <type_traits>
#include <random>
#include <algorithm>
#include <cmath>

#include "audio.h"
#include "matrix.h"
#include "model.h"

extern std::unordered_map<std::string, int> phonemeDict;

namespace fs = std::filesystem;
namespace training {

std::vector<Phoneme> loadPhonemes(const std::string& filename) {
    std::vector<Phoneme> phonemes;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open phoneme file: " << filename << std::endl;
        return phonemes;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        int start, end;
        std::string label;
        if (!(iss >> start >> end >> label)) {
            std::cerr << "Error parsing line in phoneme file: " << line << std::endl;
            continue;
        }
        phonemes.push_back({ start, end, label });
    }
    return phonemes;
}

std::vector<Word> loadWords(const std::string& filename) {
    std::vector<Word> words;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open word file: " << filename << std::endl;
        return words;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        int start, end;
        std::string text;
        if (!(iss >> start >> end >> text)) {
            std::cerr << "Error parsing line in word file: " << line << std::endl;
            continue;
        }
        words.push_back({ start, end, text });
    }
    return words;
}

std::string loadSentence(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open sentence file: " << filename << std::endl;
        return "";
    }
    
    std::string sentence;
    // Read the first (or only) line of the file that contains the full sentence.
    std::getline(infile, sentence);
    return sentence;
}

void loadData(
    int count,
    const std::string& directory,
    std::vector<std::pair<std::string, std::vector<Phoneme>>>& phonemeFiles,
    std::vector<std::pair<std::string, std::vector<Word>>>& wordFiles,
    std::vector<std::pair<std::string, std::string>>& sentenceFiles
) {
    // Recursively iterate through the directory.
    int cnt = 0;
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (cnt==count) break;
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            std::string extension = entry.path().extension().string();

            // Check file type by extension.
            if (extension == ".PHN") {
                auto phonemes = loadPhonemes(filePath);
                phonemeFiles.push_back({ filePath, phonemes });
            } else if (extension == ".WRD") {
                auto words = loadWords(filePath);
                wordFiles.push_back({ filePath, words });
            } else if (extension == ".TXT") {
                auto sentence = loadSentence(filePath);
                sentenceFiles.push_back({ filePath, sentence });
            }
        }
        cnt++;
    }
}

std::vector<int16_t> extractSegmentPCMNist(const std::string& nistWavFilename, int start, int end) {
    // Create an AudioCapture instance
    echo::AudioCapture audioCap;
    
    // Use the readNistFile function to read the file.
    if (!audioCap.readNistFile(nistWavFilename)) {
        std::cerr << "Error: Unable to read NIST wav file: " << nistWavFilename << std::endl;
        return {};
    }

    // Get the raw data (assumed to be 16-bit PCM stored as BYTEs).
    const std::vector<BYTE>& rawData = audioCap.getCapturedData();
    
    // Ensure the size is a multiple of 2.
    if (rawData.size() % 2 != 0) {
        std::cerr << "Error: Raw PCM data size is not even in file: " << nistWavFilename << std::endl;
        return {};
    }
    
    size_t numSamples = rawData.size() / 2;
    if (start < 0 || end > static_cast<int>(numSamples) || start >= end) {
        std::cerr << "Error: Invalid sample range [" << start << ", " << end 
                  << "] for file: " << nistWavFilename << std::endl;
        return {};
    }
    
    // Convert rawData (vector<BYTE>) to vector<int16_t>.
    std::vector<int16_t> samples(numSamples);
    std::memcpy(samples.data(), rawData.data(), rawData.size());
    
    // Extract and return the segment [start, end).
    std::vector<int16_t> segment(samples.begin() + start, samples.begin() + end);
    return segment;
}

std::vector<int16_t> extractEntirePCM(const std::string& wavFilename) {
    // Create an AudioCapture instance and read the file.
    echo::AudioCapture audioCap;
    if (!audioCap.readNistFile(wavFilename)) {
        std::cerr << "Error: Unable to read NIST wav file: " << wavFilename << std::endl;
        return {};
    }
    const std::vector<BYTE>& rawData = audioCap.getCapturedData();
    if (rawData.size() % 2 != 0) {
        std::cerr << "Error: Raw PCM data size is not even in file: " << wavFilename << std::endl;
        return {};
    }
    size_t numSamples = rawData.size() / 2;
    std::vector<int16_t> samples(numSamples);
    std::memcpy(samples.data(), rawData.data(), rawData.size());
    return samples;
}

void extractData(
    int count,
    const std::string& directory,
    std::vector<PhonemeData>& allPhonemeData,
    std::vector<WordData>& allWordData,
    std::vector<SentenceData>& allSentenceData
){
    // Vectors to store loaded segmentation data.
    std::vector<std::pair<std::string, std::vector<Phoneme>>> phonemeFiles;
    std::vector<std::pair<std::string, std::vector<Word>>> wordFiles;
    std::vector<std::pair<std::string, std::string>> sentenceFiles;

    loadData(count, directory, phonemeFiles, wordFiles, sentenceFiles);

    for (const auto& filePair : phonemeFiles) {
        // Determine the corresponding WAV file by replacing the extension with ".WAV".
        fs::path segPath(filePair.first);
        segPath.replace_extension(".WAV");
        std::string wavFilename = segPath.string();

        // For each phoneme in the file, extract its PCM segment.
        for (const auto& ph : filePair.second) {
            auto pcmSegment = extractSegmentPCMNist(wavFilename, ph.start, ph.end);
            if (pcmSegment.empty()) {
                std::cerr << "Warning: Could not extract PCM for phoneme '"
                            << ph.label << "' from file: " << wavFilename << std::endl;
                continue;
            }
            allPhonemeData.push_back({ ph, pcmSegment });
        }
    }
    // std::cout << "\nExtracted PCM segments for " << allPhonemeData.size() << " phonemes." << std::endl;

    for (const auto& filePair : wordFiles) {
        fs::path segPath(filePair.first);
        segPath.replace_extension(".WAV");
        std::string wavFilename = segPath.string();

        for (const auto& wd : filePair.second) {
            auto pcmSegment = extractSegmentPCMNist(wavFilename, wd.start, wd.end);
            if (pcmSegment.empty()) {
                std::cerr << "Warning: Could not extract PCM for word '"
                            << wd.text << "' from file: " << wavFilename << std::endl;
                continue;
            }
            allWordData.push_back({ wd, pcmSegment });
        }
    }
    // std::cout << "\nExtracted PCM segments for " << allWordData.size() << " words." << std::endl;

    for (const auto& filePair : sentenceFiles) {
        fs::path segPath(filePair.first);
        segPath.replace_extension(".WAV");
        std::string wavFilename = segPath.string();

        // For sentence files, we assume the entire WAV file corresponds to the sentence.
        auto fullPCM = extractEntirePCM(wavFilename);
        if (fullPCM.empty()) {
            std::cerr << "Warning: Could not extract full PCM for sentence from file: " << wavFilename << std::endl;
            continue;
        }
        allSentenceData.push_back({ filePair.second, fullPCM });
    }
    // std::cout << "\nExtracted PCM segments for " << allSentenceData.size() << " sentences." << std::endl;
}

matrix createMatrixFromPCM(const std::vector<int16_t>& pcm, int seq_len, int feature_dim) {
    matrix m(seq_len, feature_dim);
    // Ensure there are enough samples; in practice add error checking.
    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < feature_dim; j++) {
            int idx = i * feature_dim + j;
            if (idx < pcm.size())
                m(i, j) = static_cast<float>(pcm[idx]) * scale;
            else
                m(i, j) = 0.0f;
        }
    }
    return m;
}

std::unordered_map<std::string, int> buildPhonemeDictionary(const std::vector<PhonemeData>& phonemeData) {
    std::unordered_map<std::string, int> phonemeDict;
    int index = 0;
    for (const auto &entry : phonemeData) {
        const std::string &label = entry.phoneme.label;
        // Insert the label if it is not already in the dictionary.
        if (phonemeDict.find(label) == phonemeDict.end()) {
            phonemeDict[label] = index++;
        }
    }
    return phonemeDict;
}

Sample createSample(const PhonemeData &data, int seq_len, int feature_dim) {
    Sample sample;
    
    // Calculate the required number of PCM samples.
    size_t block_size = static_cast<size_t>(seq_len * feature_dim);
    
    // Prepare a vector to hold exactly block_size samples.
    std::vector<int16_t> blockData;
    
    // If there is not enough PCM data, copy all available samples and pad with zeros.
    if (data.pcm.size() < block_size) {
        blockData = data.pcm;
        blockData.resize(block_size, 0);
    } else {
        // Otherwise, take only the first block_size samples.
        blockData.assign(data.pcm.begin(), data.pcm.begin() + block_size);
    }
    
    // Build the features matrix from the PCM block.
    sample.features = createMatrixFromPCM(blockData, seq_len, feature_dim);
    
     // Convert the phoneme to a string (adjust if Phoneme is already a std::string).
    std::string phonemeStr = static_cast<std::string>(data.phoneme.label);

    // Look up the integer label using the phoneme dictionary.
    sample.label = phonemeDict[phonemeStr];
    
    return sample;
}

std::vector<Sample> vectorize(std::vector<PhonemeData> &dataset){
    std::vector<Sample> data;
    for (const auto &phoneme : dataset) {
        data.push_back(createSample(phoneme, seq_len, feature_dim));
    }
    return data;
}

// Splits the input vector 'samples' into a training vector and a testing vector.
// 'testRatio' should be a value between 0.0 and 1.0 indicating the fraction of data for testing.
std::pair<std::vector<Sample>, std::vector<Sample>> splitTrainTestSamples(
    const std::vector<Sample>& samples, double testRatio) 
{
    std::vector<Sample> shuffled = samples;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shuffled.begin(), shuffled.end(), gen);
    
    size_t testCount = static_cast<size_t>(std::ceil(testRatio * shuffled.size()));
    
    std::vector<Sample> test(shuffled.begin(), shuffled.begin() + testCount);
    std::vector<Sample> train(shuffled.begin() + testCount, shuffled.end());
    
    return std::make_pair(train, test);
}

} // Namespace Training
