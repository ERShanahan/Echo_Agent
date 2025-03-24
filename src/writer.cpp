#include "writer.h"
#include <iostream>
#include "keys.h" // Assume simulateTyping is declared here
#include <ctime>

namespace echo {

Writer::Writer() : running(false) {}

Writer::~Writer() {
    stop();
}

void Writer::start() {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (running)
            return;
        running = true;
    }
    writerThread = std::thread(&Writer::writerThreadFunc, this);
}

void Writer::stop() {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        running = false;
    }
    queueCV.notify_all();
    if (writerThread.joinable()) {
        writerThread.join();
    }
}

void Writer::enqueueText(const std::string &text) {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        textQueue.push(text);
    }
    queueCV.notify_one();
}

void Writer::writerThreadFunc() {
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [this]() { return !textQueue.empty() || !running; });
        if (!running && textQueue.empty()) {
            break;
        }
        // Retrieve and remove the next string.
        std::string textToType = textQueue.front();
        textQueue.pop();
        lock.unlock();

        // Simulate typing the string.
        simulateTyping(textToType.c_str());
        // std::time_t currentTime = std::time(0);
        // std::cout << "Simulated Typing at time: " << currentTime << std::endl;
    }
}

} // namespace echo
