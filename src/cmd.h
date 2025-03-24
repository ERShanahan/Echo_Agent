#include <iostream>
#include <chrono>
#include <thread>
#include <string>

namespace echo {

const char* letterE[5] = {
    " _____ ",
    "| ____|",
    "|  _|  ",
    "| |__  ",
    "|_____|"
};

const char* letterC[5] = {
    "  ____ ",
    " / ___|",
    "| |    ",
    "| |    ",
    " \\____|"
};

const char* letterH[5] = {
    " _   _ ",
    "| | | |",
    "| |_| |",
    "|  _  |",
    "|_| |_|"
};

const char* letterO[5] = {
    "  ___  ",
    " / _ \\ ",
    "| | | |",
    "| |_| |",
    " \\___/ "
};   

const char* letterA[5] = {
    "  __   ",
    " |   \\",
    " |  - \\",
    " |  |  |",
    " |__|__|"
};

const char* letterG[5] = {
    "   ____ ",
    "   / ___|",
    " | |  _ ",
    "| |_| |",
    " \\____|"
};

const char* letterN[5] = {
    " _   _ ",
    "|\\  | |",
    "| \\ | |",
    "|  \\| |",
    "|   \\_|"
};

const char* letterT[5] = {
    "_______ ",
    "__   __ ",
    "  | |   ",
    "  | |   ",
    "  | |   "
};

inline void printEcho(bool delay){
    std::cout << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << letterE[i] << " " 
                  << letterC[i] << " " 
                  << letterH[i] << " " 
                  << letterO[i] << "   "
                  << letterA[i] << " " 
                  << letterG[i] << " " 
                  << letterE[i] << " " 
                  << letterN[i] << " " 
                  << letterT[i] << std::endl;
        if (delay)
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    std::cout << std::endl;
}

const std::string help_cmd = "help";
const std::string tune_cmd = "tune";
const std::string start_cmd = "start";
const std::string stop_cmd = "stop";
const std::string quit_cmd = "quit";
const std::string load_cmd = "load";
const std::string train_cmd = "train";
const std::string test_cmd = "test";
const std::string clear_cmd = "clear";

} // namespace echo
