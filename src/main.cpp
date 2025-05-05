#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <ctime>
#include <sstream>
#include <iterator>

#include "data.h"
#include "keys.h"
#include "cmd.h"
#include "audio.h"
#include "cblas.h"
#include "matrix.h"
#include "model.h"
#include "listener.h"
#include "writer.h"

namespace fs = std::filesystem;
using namespace echo;
using namespace training;

std::vector<PhonemeData> allPhonemeData;
std::vector<WordData> allWordData;
std::vector<SentenceData> allSentenceData;

std::unordered_map<std::string, int> phonemeDict;
std::unordered_map<int, std::string> indexToPhoneme;

Listener gListener;
Writer gWriter;
Model* gModel = nullptr;


int main(int argc, char* argv[]) {

    printEcho(true);

    std::string input;
    while(true){
        std::cout << ">>>";
        std::getline(std::cin, input);
        
        std::transform(input.begin(), input.end(), input.begin(), ::tolower);

        std::istringstream iss(input);
        std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss},
                                         std::istream_iterator<std::string>{} };

        if(tokens.empty()) continue;

        std::string cmd = tokens[0];

        if (cmd == help_cmd){

            std::cout << "Available commands:\n"
                      << "  help  - Show this help message\n"
                      << "  load [samples] - Load segmentation data and extract sample audio segment\n"
                      << "  train [epochs] - Train the model for [epochs] episodes\n"
                      << "  tune  - Tuning functionality (not implemented)\n"
                      << "  start - Start the program (not implemented)\n"
                      << "  stop  - Stop the program (not implemented)\n"
                      << "  quit  - Exit the application\n";

        }else if (cmd == load_cmd){

            if (tokens[1] == "data"){
                int count = 100;
                if (tokens.size() > 2) {
                    try {
                        count = std::stoi(tokens[2]);
                    } catch (const std::exception &e) {
                        std::cout << "Invalid argument for load command. Using default value 100." << std::endl;
                    }
                }

                std::cout << "Loading Data..." << std::endl;

                std::time_t curr_time = std::time(0);
                const std::string directory = "archive/data/TRAIN";

                extractData(count, directory, allPhonemeData, allWordData, allSentenceData);

                phonemeDict = buildPhonemeDictionary(allPhonemeData);
                for (const auto &pair : phonemeDict) {
                    indexToPhoneme[pair.second] = pair.first;
                }

                std::time_t next_time = std::time(0);

                std::cout << "Done loading " << count << " data in " << next_time - curr_time << " seconds. Num of Phonemes: " << phonemeDict.size() << std::endl;
                
            }else if (tokens[1] == "model"){
                
                if (allPhonemeData.empty()) {
                    std::cout << "No data loaded. Please use the 'load data' command before creating model." << std::endl;
                    continue;
                }

                if (tokens.size() > 2) {
                    try {
                        gModel = new Model(N, N, D, H, Dff, phonemeDict.size(), tokens[2]);
                        std::cout << "Model created from weight file: " << tokens[2] << std::endl;
                    } catch (const std::exception &e) {
                        std::cout << "Invalid argument for weight file." << std::endl;
                    }
                }else{
                    gModel = new Model(N, N, D, H, Dff, phonemeDict.size());
                    std::cout << "New model created." << std::endl;
                }

            }else{

                std::cout << "Unrecognized argument for load command: " << tokens[1] << "."<< std::endl;
           
            }

        }else if (cmd == train_cmd){

            // Ensure that data has been loaded.
            if (allPhonemeData.empty()) {
                std::cout << "No data loaded. Please use the load command before training." << std::endl;
                continue;
            }

            int epochs = 100;
            if (tokens.size() > 1) {
                try {
                    epochs = std::stoi(tokens[1]);
                } catch (const std::exception &e) {
                    std::cout << "Invalid argument for train command. Using default value 100." << std::endl;
                }
            }

            std::vector<Sample> vectorizedData = vectorize(allPhonemeData);

            std::cout << "Vectorized Data..." << std::endl;

            //Split train test here
            std::pair<std::vector<Sample>, std::vector<Sample>> trainTestPair = splitTrainTestSamples(vectorizedData, 0.2);

            std::cout << "Data split into testing and training..." << std::endl;

            std::cout << "Starting Training on Loaded Data... " << std::endl;

            // Train the Model.
            gModel->trainModelMiniBatch(trainTestPair.first, 0.01, epochs, 100);

            int correct = 0;
            double totalLoss = 0.0;
            int N = trainTestPair.second.size();
            int V = phonemeDict.size();

            // initialize confusion matrix
            std::vector<std::vector<int>> confMat(V, std::vector<int>(V, 0));

            for (auto &sample : trainTestPair.second) {
                matrix mask = createMask(sample.features.rows);
                matrix logits = gModel->forwardPass(
                    sample.features,
                    sample.features.shift(1),
                    mask
                );
                // loss
                double loss = gModel->crossEntropyLoss(logits, sample.label);
                totalLoss += loss;

                // prediction
                int pred = 0;
                double best = logits(0,0);
                for (int j = 1; j < logits.cols; ++j) {
                    if (logits(0,j) > best) {
                        best = logits(0,j);
                        pred = j;
                    }
                }
                if (pred == sample.label) ++correct;
                confMat[sample.label][pred]++;
            }

            double avgLoss  = totalLoss / N;
            double accuracy = 100.0 * correct / N;

            // print summary
            std::cout << "\n=== Test set evaluation ===\n"
                    << "  Examples:       " << N      << "\n"
                    << "  Avg. loss:      " << avgLoss << "\n"
                    << "  Accuracy:       " << accuracy << "%\n\n";

            // (optional) print a small confusion‐matrix snippet:
            std::cout << "Confusion matrix (true × pred):\n";
            std::cout << "    ";
            for (int j = 0; j < V; ++j) std::cout << std::setw(4) << j;
            std::cout << "\n";
            for (int i = 0; i < V; ++i) {
                std::cout << std::setw(3) << i << ":";
                for (int j = 0; j < V; ++j) {
                    std::cout << std::setw(4) << confMat[i][j];
                }
                std::cout << "\n";
            }


        }else if (cmd == test_cmd){

            if (allPhonemeData.empty()) {
                std::cout << "No data loaded. Please use the load command before training." << std::endl;
                continue;
            }

            std::vector<Sample> vectorizedData = vectorize(allPhonemeData);
            std::pair<std::vector<Sample>, std::vector<Sample>> trainTestPair = splitTrainTestSamples(vectorizedData, 0.2);

            int correct = 0;
            int total = 0;
            double totalLoss = 0.0;
            for (const Sample &sample : trainTestPair.second) {
                // Create a mask based on the input sample dimensions.
                matrix mask = createMask(sample.features.rows);
                
                // Perform a forward pass with the sample.
                matrix logits = gModel->forwardPass(sample.features, sample.features.shift(1), mask);
                
                // Compute the cross entropy loss for the sample.
                double loss = gModel->crossEntropyLoss(logits, sample.label);
                totalLoss += loss;
                
                // Determine the predicted label (assuming logits has 1 row and vocab_size columns).
                int predLabel = 0;
                double bestScore = logits(0, 0);
                for (int j = 1; j < logits.cols; j++) {
                    if (logits(0, j) > bestScore) {
                        bestScore = logits(0, j);
                        predLabel = j;
                    }
                }
                
                // Count correct predictions.
                if (predLabel == sample.label) {
                    correct++;
                }
                total++;
            }
            
            double avgLoss = totalLoss / total;
            double accuracy = (100.0 * correct) / total;
            std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
            std::cout << "Average Test Loss: " << avgLoss << std::endl;

        }else if (cmd == start_cmd){

            std::cout << "Running Program..." << std::endl;

            if (phonemeDict.empty()){
                std::cout << "Dictionary empty, please load data first." << std::endl;
                continue;
            }

            gListener.start(600);
            gWriter.start();
            std::cout << "Audio Listener started. Text writer started.\nFocus the intended application...\n"; 

            gModel->startPredictor();

        }else if (cmd == stop_cmd){

            std::cout << "Stopping Program..." << std::endl;

            gListener.stop();
            gWriter.stop();
            gModel->stopPredictor();
            std::cout << "Audio Listener and Writer stopped." << std::endl;

        } else if (cmd == clear_cmd) {

            #ifdef _WIN32
                system("cls");
            #else
                system("clear");
            #endif
            printEcho(false);

        }else if (cmd == quit_cmd){

            std::cout << "Exiting App..." << std::endl;
            break;
        
        }else{

            std::cout << "Unrecognized Command: " << input << std::endl;
        
        }
    }

    return 0;
}
