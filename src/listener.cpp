#include "listener.h"
#include <iostream>
#include <chrono>
#include <ctime>

namespace echo {

Listener::Listener() : running(false) {}

Listener::~Listener() {
    stop();
}

void Listener::start(int drainIntervalMilliseconds) {
    if (running.load()) return; // Already running

    // Initialize the audio capture device.
    if (!audioCapture.initialize()) {
        std::cerr << "AudioCapture initialization failed." << std::endl;
        return;
    }

    HRESULT hr = audioCapture.pAudioClient->Start();
    if (FAILED(hr)) {
        std::cerr << "AudioClient Start failed: 0x" << std::hex << hr << std::endl;
        return;
    }

    running.store(true);
    // Start the continuous capture thread.
    captureThread = std::thread(&Listener::captureThreadFunc, this);
    // Start the drain thread that periodically extracts data.
    drainThread = std::thread(&Listener::drainThreadFunc, this, drainIntervalMilliseconds);
}

void Listener::stop() {
    running.store(false);
    // Signal AudioCapture to stop (if needed).
    audioCapture.stopCapture();
    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (drainThread.joinable()) {
        drainThread.join();
    }
    audioCapture.cleanup();
}

std::vector<BYTE> Listener::getNextAudioSnippet() {
    std::lock_guard<std::mutex> lock(bufferMutex);
    if (audioBuffer.empty()) {
        return {};
    }
    std::vector<BYTE> snippet = audioBuffer.front();
    audioBuffer.erase(audioBuffer.begin());
    return snippet;
}

// This thread continuously polls the capture device for available audio data.
void Listener::captureThreadFunc() {
    while (running.load()) {
        UINT32 packetLength = 0;
        HRESULT hr = audioCapture.pCaptureClient->GetNextPacketSize(&packetLength);
        // std::time_t currentTime = std::time(0);
        // std::cout << "Captured audio at time: " << currentTime << std::endl;
        if (FAILED(hr)) {
            std::cerr << "[DEBUG] GetNextPacketSize failed: 0x" << std::hex << hr << std::endl;
            continue;
        }
        while (packetLength != 0) {
            BYTE* pData = nullptr;
            UINT32 numFramesAvailable = 0;
            DWORD dwFlags = 0;
            hr = audioCapture.pCaptureClient->GetBuffer(&pData, &numFramesAvailable, &dwFlags, nullptr, nullptr);
            if (FAILED(hr)) {
                std::cerr << "[DEBUG] GetBuffer failed: 0x" << std::hex << hr << std::endl;
                break;
            }
            size_t dataSize = numFramesAvailable * audioCapture.pwfx->nBlockAlign;
            {
                // Use the AudioCapture's own mutex to safely append data.
                std::lock_guard<std::mutex> lock(audioCapture.captureMutex);
                audioCapture.capturedData.insert(audioCapture.capturedData.end(), pData, pData + dataSize);
            }
            hr = audioCapture.pCaptureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) {
                std::cerr << "[DEBUG] ReleaseBuffer failed: 0x" << std::hex << hr << std::endl;
                break;
            }
            hr = audioCapture.pCaptureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                std::cerr << "[DEBUG] GetNextPacketSize failed: 0x" << std::hex << hr << std::endl;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// This thread runs concurrently and, at fixed intervals, drains the audio data
// captured so far into audioBuffer without stopping the capture.
void Listener::drainThreadFunc(int drainIntervalMilliseconds) {
    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(drainIntervalMilliseconds));
        std::vector<BYTE> snippet;
        {
            // Drain capturedData in a thread-safe manner.
            std::lock_guard<std::mutex> lock(audioCapture.captureMutex);
            snippet = audioCapture.capturedData;
            audioCapture.capturedData.clear();
        }
        if (!snippet.empty()) {
            {
                std::lock_guard<std::mutex> lock(bufferMutex);
                audioBuffer.push_back(snippet);
            }
        } else {
            std::cout << "[DEBUG] Drained snippet is empty.\n";
        }
    }
}

} // namespace echo
