#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <mutex>

#include "keys.h"

namespace echo{

class AudioCapture {
public:

    std::mutex captureMutex;

    AudioCapture();
    ~AudioCapture();

    // Initializes COM, retrieves the default capture device, and sets up WASAPI.
    bool initialize();

    // Starts capturing audio until a key is pressed.
    bool startCapture();

    // Stops capture (sets an internal flag; not used in this example loop).
    bool stopCapture();

    // Returns the raw captured audio data.
    const std::vector<BYTE>& getCapturedData() const;

    // Writes data to .wav file format
    bool writeWavFile(const std::string&);

    // Reads raw data from .wav file
    bool readWavFile(const std::string&);

    // Reads raw data from Nist wav file (from timit database)
    bool readNistFile(const std::string&);

    // Helper function to clean up COM objects and memory.
    void cleanup();

    // COM interface pointers.
    IMMDeviceEnumerator* pEnumerator;
    IMMDevice* pDevice;
    IAudioClient* pAudioClient;
    IAudioCaptureClient* pCaptureClient;
    WAVEFORMATEX* pwfx;

    bool capturing;
    std::vector<BYTE> capturedData;

};

} // Namespace echo

#endif // AUDIO_H

