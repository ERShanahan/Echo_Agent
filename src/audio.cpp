#include "audio.h"
#include <iostream>
#include <conio.h>
#include <stdio.h>
#include <fstream>
#include <cstring>


#include "keys.h"

namespace echo {

AudioCapture::AudioCapture()
    : pEnumerator(nullptr), pDevice(nullptr), pAudioClient(nullptr),
      pCaptureClient(nullptr), pwfx(nullptr), capturing(false)
{
}

AudioCapture::~AudioCapture() {
    cleanup();
}

bool AudioCapture::initialize() {
    HRESULT hr = CoInitialize(nullptr);
    if (FAILED(hr)) {
        std::cerr << "CoInitialize failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator), (void**)&pEnumerator);
    if (FAILED(hr)) {
        std::cerr << "CoCreateInstance failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    hr = pEnumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &pDevice);
    if (FAILED(hr)) {
        std::cerr << "GetDefaultAudioEndpoint failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
    if (FAILED(hr)) {
        std::cerr << "Device Activate failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        std::cerr << "GetMixFormat failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Initialize the audio client in shared mode for 1-second buffer duration.
    hr = pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, pwfx, nullptr);
    if (FAILED(hr)) {
        std::cerr << "AudioClient Initialize failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    hr = pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pCaptureClient);
    if (FAILED(hr)) {
        std::cerr << "GetService for IAudioCaptureClient failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}

bool AudioCapture::startCapture() {
    HRESULT hr = pAudioClient->Start();
    if (FAILED(hr)) {
        std::cerr << "AudioClient Start failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    capturing = true;
    
    // Capture loop: poll for available audio packets.
    while (capturing) {
        UINT32 packetLength = 0;
        hr = pCaptureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            std::cerr << "GetNextPacketSize failed: 0x" << std::hex << hr << std::endl;
            break;
        }
        
        while (packetLength != 0) {
            BYTE* pData = nullptr;
            UINT32 numFramesAvailable = 0;
            DWORD dwFlags = 0;
            
            hr = pCaptureClient->GetBuffer(&pData, &numFramesAvailable, &dwFlags, nullptr, nullptr);
            if (FAILED(hr)) {
                std::cerr << "GetBuffer failed: 0x" << std::hex << hr << std::endl;
                capturing = false;
                break;
            }
            
            // For verification: store the raw audio data.
            // Note: The actual byte count may depend on the format (channels, bits per sample).
            size_t dataSize = numFramesAvailable * pwfx->nBlockAlign;
            capturedData.insert(capturedData.end(), pData, pData + dataSize);
            
            hr = pCaptureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) {
                std::cerr << "ReleaseBuffer failed: 0x" << std::hex << hr << std::endl;
                capturing = false;
                break;
            }
            
            hr = pCaptureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                std::cerr << "GetNextPacketSize failed: 0x" << std::hex << hr << std::endl;
                capturing = false;
                break;
            }
        }
    }
    
    hr = pAudioClient->Stop();
    if (FAILED(hr)) {
        std::cerr << "AudioClient Stop failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}

bool AudioCapture::stopCapture() {
    capturing = false;
    return true;
}

const std::vector<BYTE>& AudioCapture::getCapturedData() const {
    return capturedData;
}

void AudioCapture::cleanup() {
    if (pCaptureClient) {
        pCaptureClient->Release();
        pCaptureClient = nullptr;
    }
    if (pAudioClient) {
        pAudioClient->Release();
        pAudioClient = nullptr;
    }
    if (pDevice) {
        pDevice->Release();
        pDevice = nullptr;
    }
    if (pEnumerator) {
        pEnumerator->Release();
        pEnumerator = nullptr;
    }
    if (pwfx) {
        CoTaskMemFree(pwfx);
        pwfx = nullptr;
    }
    CoUninitialize();
}

bool AudioCapture::writeWavFile(const std::string& filename) {
    // Ensure we have data
    if (capturedData.empty()) {
        std::cerr << "No audio data to write." << std::endl;
        return false;
    }

    // Ensure pwfx is valid (e.g., set by readNistFile or readWavFile)
    if (!pwfx || pwfx->nChannels == 0 || pwfx->nSamplesPerSec == 0 || pwfx->wBitsPerSample == 0) {
        std::cerr << "pwfx is not set to a valid PCM format. "
                  << "Make sure readNistFile or readWavFile was called first.\n";
        return false;
    }

    // Open file
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
        return false;
    }

    // Extract format parameters
    const uint16_t channels      = pwfx->nChannels;
    const uint32_t sampleRate    = pwfx->nSamplesPerSec;
    const uint16_t bitsPerSample = pwfx->wBitsPerSample;
    const uint16_t blockAlign    = pwfx->nBlockAlign;
    const uint32_t byteRate      = pwfx->nAvgBytesPerSec;

    // Standard PCM (wFormatTag=1) => 16-byte fmt chunk
    const uint16_t audioFormat   = 1;   
    const uint32_t fmtChunkSize  = 16;  

    // Data size is just the size of capturedData
    const uint32_t dataSize      = static_cast<uint32_t>(capturedData.size());

    // The overall RIFF chunk size = 36 + dataSize 
    // (Because: 12-byte RIFF header + 24-byte fmt chunk + 8-byte data header = 44 total minus 8 for the standard chunk-size field => 36 offset).
    const uint32_t chunkSize     = 36 + dataSize;

    // --- RIFF header ---
    outFile.write("RIFF", 4);
    outFile.write(reinterpret_cast<const char*>(&chunkSize), 4);
    outFile.write("WAVE", 4);

    // --- fmt chunk ---
    outFile.write("fmt ", 4);
    outFile.write(reinterpret_cast<const char*>(&fmtChunkSize), 4);
    outFile.write(reinterpret_cast<const char*>(&audioFormat), 2);
    outFile.write(reinterpret_cast<const char*>(&channels), 2);
    outFile.write(reinterpret_cast<const char*>(&sampleRate), 4);
    outFile.write(reinterpret_cast<const char*>(&byteRate), 4);
    outFile.write(reinterpret_cast<const char*>(&blockAlign), 2);
    outFile.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    // --- data chunk ---
    outFile.write("data", 4);
    outFile.write(reinterpret_cast<const char*>(&dataSize), 4);
    outFile.write(reinterpret_cast<const char*>(capturedData.data()), dataSize);

    outFile.close();

    std::cout << "WAV file written successfully: " << filename << "\n"
              << "Format: " << channels << " channel(s), " 
              << sampleRate << " Hz, " 
              << bitsPerSample << " bits.\n"
              << "Data size: " << dataSize << " bytes.\n";

    return true;
}

bool AudioCapture::readWavFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return false;
    }

    // --- Read and verify the RIFF header ---
    char header[12] = {0};
    inFile.read(header, 12);
    if (inFile.gcount() < 12 || std::strncmp(header, "RIFF", 4) != 0 || std::strncmp(header + 8, "WAVE", 4) != 0) {
        std::cerr << "File is not a valid RIFF/WAVE file." << std::endl;
        return false;
    }

    // --- Process chunks ---
    bool fmtFound = false;
    bool dataFound = false;
    uint16_t audioFormat = 0, numChannels = 0, bitsPerSample = 0;
    uint32_t sampleRate = 0, byteRate = 0, dataSize = 0;
    uint16_t blockAlign = 0;

    // Loop until both chunks are found.
    while (inFile.good() && (!fmtFound || !dataFound)) {
        char chunkId[4] = {0};
        inFile.read(chunkId, 4);
        if (inFile.gcount() < 4)
            break;  // No more data.

        uint32_t subchunkSize = 0;
        inFile.read(reinterpret_cast<char*>(&subchunkSize), sizeof(subchunkSize));

        if (std::strncmp(chunkId, "fmt ", 4) == 0) {
            fmtFound = true;
            inFile.read(reinterpret_cast<char*>(&audioFormat), sizeof(audioFormat));
            inFile.read(reinterpret_cast<char*>(&numChannels), sizeof(numChannels));
            inFile.read(reinterpret_cast<char*>(&sampleRate), sizeof(sampleRate));
            inFile.read(reinterpret_cast<char*>(&byteRate), sizeof(byteRate));
            inFile.read(reinterpret_cast<char*>(&blockAlign), sizeof(blockAlign));
            inFile.read(reinterpret_cast<char*>(&bitsPerSample), sizeof(bitsPerSample));

            // If there are extra bytes in the fmt chunk, skip them.
            if (subchunkSize > 16) {
                inFile.seekg(subchunkSize - 16, std::ios::cur);
            }
        }
        else if (std::strncmp(chunkId, "data", 4) == 0) {
            dataFound = true;
            dataSize = subchunkSize;
            capturedData.resize(dataSize);
            inFile.read(reinterpret_cast<char*>(capturedData.data()), dataSize);
        }
        else {
            // Skip any other chunk.
            inFile.seekg(subchunkSize, std::ios::cur);
        }
    }

    if (!fmtFound || !dataFound) {
        std::cerr << "Failed to locate required chunks (fmt or data) in the WAV file." << std::endl;
        return false;
    }

    // --- Update internal format structure (pwfx) based on file header ---
    if (pwfx) {
        CoTaskMemFree(pwfx);
        pwfx = nullptr;
    }
    pwfx = reinterpret_cast<WAVEFORMATEX*>(CoTaskMemAlloc(sizeof(WAVEFORMATEX)));
    if (!pwfx) {
        std::cerr << "Memory allocation for WAVEFORMATEX failed." << std::endl;
        return false;
    }
    pwfx->wFormatTag      = audioFormat;
    pwfx->nChannels       = numChannels;
    pwfx->nSamplesPerSec  = sampleRate;
    pwfx->nAvgBytesPerSec = byteRate;
    pwfx->nBlockAlign     = blockAlign;
    pwfx->wBitsPerSample  = bitsPerSample;
    pwfx->cbSize          = 0;  // No extra format bytes for PCM

    // Optionally, print header information.
    std::cout << "WAV file loaded: " << filename << "\n";
    std::cout << "Audio Format: " << audioFormat
              << ", Channels: " << numChannels
              << ", Sample Rate: " << sampleRate
              << ", Bits per Sample: " << bitsPerSample
              << ", Data Size: " << dataSize << " bytes" << std::endl;

    return true;
}

bool AudioCapture::readNistFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return false;
    }

    // --- 1. Validate the NIST header prefix ---
    std::string line;
    std::getline(inFile, line);
    if (line.find("NIST_1A") == std::string::npos) {
        std::cerr << "File is not a valid NIST file." << std::endl;
        return false;
    }

    // --- 2. Read the declared header size (e.g. 1024) ---
    std::getline(inFile, line);
    int headerSize = 0;
    try {
        headerSize = std::stoi(line);
    } catch (std::exception& e) {
        std::cerr << "Error parsing header size: " << e.what() << std::endl;
        return false;
    }

    // --- 3. Initialize default values (if TIMIT-like) ---
    int sampleRate    = 16000;
    int channelCount  = 1;
    int sampleBits    = 16;
    bool bigEndian    = false;

    // If the NIST file has fields like:
    //   channel_count -i 1
    //   sample_rate -i 16000
    //   sample_sig_bits -i 16
    //   sample_byte_format -s2 01
    //
    // we'll parse them before "end_head".
    bool endHeadFound = false;
    while (std::getline(inFile, line)) {
        // Check if we reached the end of the ASCII header
        if (line.find("end_head") != std::string::npos) {
            endHeadFound = true;
            break;
        }

        // Look for known fields and parse them:
        if (line.find("sample_rate -i") != std::string::npos) {
            // e.g. "sample_rate -i 16000"
            // We can split the line by spaces and parse the last token
            auto pos = line.rfind(' ');
            if (pos != std::string::npos) {
                sampleRate = std::stoi(line.substr(pos));
            }
        }
        else if (line.find("channel_count -i") != std::string::npos) {
            // e.g. "channel_count -i 1"
            auto pos = line.rfind(' ');
            if (pos != std::string::npos) {
                channelCount = std::stoi(line.substr(pos));
            }
        }
        else if (line.find("sample_sig_bits -i") != std::string::npos) {
            // e.g. "sample_sig_bits -i 16"
            auto pos = line.rfind(' ');
            if (pos != std::string::npos) {
                sampleBits = std::stoi(line.substr(pos));
            }
        }
        else if (line.find("sample_byte_format -s2") != std::string::npos) {
            // e.g. "sample_byte_format -s2 01" or "sample_byte_format -s2 10"
            // "01" => little-endian, "10" => big-endian
            auto pos = line.rfind(' ');
            if (pos != std::string::npos) {
                std::string byteFormat = line.substr(pos + 1);
                if (byteFormat == "10") {
                    bigEndian = true;
                }
            }
        }
    }

    if (!endHeadFound) {
        std::cerr << "Header does not contain end_head marker." << std::endl;
        return false;
    }

    // --- 4. Seek to the end of the header as declared by 'headerSize' ---
    inFile.seekg(headerSize, std::ios::beg);

    // --- 5. Read the remainder of the file into capturedData (raw PCM samples) ---
    capturedData.assign(std::istreambuf_iterator<char>(inFile),
                        std::istreambuf_iterator<char>());

    // If the file uses big-endian samples, convert to little-endian. 
    // TIMIT NIST often uses 16-bit PCM, so let's swap pairs of bytes if bigEndian is true.
    if (bigEndian && sampleBits == 16) {
        for (size_t i = 0; i + 1 < capturedData.size(); i += 2) {
            std::swap(capturedData[i], capturedData[i + 1]);
        }
    }

    // --- 6. Create or update the WAVEFORMATEX (pwfx) to reflect these parameters ---
    if (pwfx) {
        CoTaskMemFree(pwfx);
        pwfx = nullptr;
    }

    pwfx = reinterpret_cast<WAVEFORMATEX*>(CoTaskMemAlloc(sizeof(WAVEFORMATEX)));
    if (!pwfx) {
        std::cerr << "Memory allocation for WAVEFORMATEX failed." << std::endl;
        return false;
    }

    // Fill the wave format with the parsed values
    pwfx->wFormatTag      = WAVE_FORMAT_PCM;                   // Standard PCM
    pwfx->nChannels       = static_cast<WORD>(channelCount);
    pwfx->nSamplesPerSec  = static_cast<DWORD>(sampleRate);
    pwfx->wBitsPerSample  = static_cast<WORD>(sampleBits);
    pwfx->nBlockAlign     = pwfx->nChannels * (pwfx->wBitsPerSample / 8);
    pwfx->nAvgBytesPerSec = pwfx->nBlockAlign * pwfx->nSamplesPerSec;
    pwfx->cbSize          = 0;

    return true;
}

} // Namespace echo


// if (argc < 2) {
    //     std::cout << "Usage: keyboard_sim <text to type>" << std::endl;
    //     return 1;
    // }

    // std::cout << "Focus the target application. Starting in 5 seconds..." << std::endl;
    // Sleep(5000);

    // std::string textToType;
    // for (int i = 1; i < argc; ++i) {
    //     textToType += argv[i];
    //     if (i < argc - 1) {
    //         textToType += " ";
    //     }
    // }

    // std::cout << "Simulating typing: " << textToType << std::endl;
    // simulateTyping(textToType.c_str());

    // AudioCapture audio;

    // if (!audio.initialize()) {
    //     std::cerr << "Audio initialization failed.\n";
    //     return 1;
    // }
    
    // if (!audio.startCapture()) {
    //     std::cerr << "Audio capture failed.\n";
    //     return 1;
    // }
    
    // std::cout << "Audio capture stopped.\n";
    
    // const std::string infile = "SA1.wav";
    // audio.readNistFile(infile);

    // const std::vector<BYTE>& data = audio.getCapturedData();
    // std::cout << "Captured data size: " << data.size() << " bytes.\n";
    
    // if (!data.empty()) {
    //     std::cout << "Data: " << std::endl;
    //     for (size_t i = 0; i < 100 && i < data.size(); ++i) {
    //         // Print each byte in hexadecimal.
    //         std::cout << std::hex << static_cast<int>(data[i]) << " ";
    //     }
    //     std::cout << std::dec << std::endl;
    // }

    // const std::string filename = "output.wav";
    // audio.writeWavFile(filename);

    // AudioCapture readAudio;

    // readAudio.readWavFile(filename);

    // const std::string newFile = "testOutput.wav";
    // readAudio.writeWavFile(newFile);
    