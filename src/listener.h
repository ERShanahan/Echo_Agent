#ifndef LISTENER_H
#define LISTENER_H

#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>

#include "audio.h"

namespace echo {

class Listener {
public:
    Listener();
    ~Listener();

    // Starts continuous capture and begins draining data every drainIntervalMilliseconds.
    void start(int drainIntervalMilliseconds);

    // Stops continuous capture.
    void stop();

    // Returns (and removes) the next captured snippet from the snippet buffer.
    std::vector<BYTE> getNextAudioSnippet();

private:
    // Thread function that continuously reads available audio packets.
    void captureThreadFunc();

    // Thread function that periodically drains captured audio into the snippet buffer.
    void drainThreadFunc(int drainIntervalMilliseconds);

    // The AudioCapture instance (assumes modifications for thread-safe access).
    AudioCapture audioCapture;

    // Thread for continuous capture.
    std::thread captureThread;
    // Thread for draining data periodically.
    std::thread drainThread;

    // Buffer for storing audio snippets.
    std::vector<std::vector<BYTE>> audioBuffer;
    std::mutex bufferMutex;  // Protects audioBuffer

    // Flag to signal threads to keep running.
    std::atomic<bool> running;
};

} // namespace echo

#endif // LISTENER_H
