#ifndef WRITER_H
#define WRITER_H

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace echo {

class Writer {
public:
    Writer();
    ~Writer();

    // Start the writer thread.
    void start();
    // Stop the writer thread.
    void stop();

    // Enqueue a new text string to be typed.
    void enqueueText(const std::string &text);

private:
    // Thread function that processes the text queue.
    void writerThreadFunc();

    std::queue<std::string> textQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::thread writerThread;
    bool running;
};

} // namespace echo

#endif // WRITER_H
