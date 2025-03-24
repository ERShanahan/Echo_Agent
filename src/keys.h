#ifndef KEYMAPPINGS_H
#define KEYMAPPINGS_H

#include <iostream>
#include <cctype>

#include <Windows.h>

namespace echo {

    // A helper structure that holds the virtual key code and whether the Shift modifier is needed.
    struct KeyMapping {
        UINT vk;   // Virtual key code (VK)
        bool shift; // true if the key requires Shift to produce the desired character
    };

    // Enumeration for the alphabet (A-Z) with explicit VK codes.
    enum class Alphabet : UINT {
        A = 0x41,
        B = 0x42,
        C = 0x43,
        D = 0x44,
        E = 0x45,
        F = 0x46,
        G = 0x47,
        H = 0x48,
        I = 0x49,
        J = 0x4A,
        K = 0x4B,
        L = 0x4C,
        M = 0x4D,
        N = 0x4E,
        O = 0x4F,
        P = 0x50,
        Q = 0x51,
        R = 0x52,
        S = 0x53,
        T = 0x54,
        U = 0x55,
        V = 0x56,
        W = 0x57,
        X = 0x58,
        Y = 0x59,
        Z = 0x5A
    };

    // Helper function to convert an Alphabet value into its KeyMapping.
    inline KeyMapping GetAlphabetMapping(Alphabet letter) {
        // No Shift required for letters in their base form.
        return { static_cast<UINT>(letter), false };
    }

    // Namespace for punctuation key mappings.
    namespace Punctuation {
        // Comma: the comma key (VK_OEM_COMMA)
        const KeyMapping Comma         = { VK_OEM_COMMA, false };
        // Period: the period key (VK_OEM_PERIOD)
        const KeyMapping Period        = { VK_OEM_PERIOD, false };

        // Semicolon and Colon:
        // VK_OEM_1 is used for the semicolon key; Shift produces a colon.
        const KeyMapping Semicolon     = { VK_OEM_1, false };
        const KeyMapping Colon         = { VK_OEM_1, true };

        // Exclamation mark: typically Shift + '1' (0x31)
        const KeyMapping Exclamation   = { 0x31, true };

        // Question mark: typically Shift + the forward slash key (VK_OEM_2)
        const KeyMapping Question      = { VK_OEM_2, true };

        // HashTap: Shift + 3
        const KeyMapping HashTag       = { 0x33, true };

        // Apostrophe and Quotation:
        // VK_OEM_7 is used for the apostrophe key; Shift produces a quotation mark.
        const KeyMapping Apostrophe    = { VK_OEM_7, false };
        const KeyMapping Quotation     = { VK_OEM_7, true };

        // Hyphen and Underscore:
        // VK_OEM_MINUS is used for the hyphen key; Shift produces an underscore.
        const KeyMapping Hyphen        = { VK_OEM_MINUS, false };
        const KeyMapping Underscore    = { VK_OEM_MINUS, true };

        // Left and Right Parentheses:
        // Typically, '(' is Shift + '9' (0x39) and ')' is Shift + '0' (0x30)
        const KeyMapping LeftParenthesis  = { 0x39, true };
        const KeyMapping RightParenthesis = { 0x30, true };

        // Brackets and Braces:
        // VK_OEM_4 corresponds to the left bracket; Shift produces '{'
        const KeyMapping LeftBracket   = { VK_OEM_4, false };
        const KeyMapping LeftBrace     = { VK_OEM_4, true };
        // VK_OEM_6 corresponds to the right bracket; Shift produces '}'
        const KeyMapping RightBracket  = { VK_OEM_6, false };
        const KeyMapping RightBrace    = { VK_OEM_6, true };

        // Slash: the forward slash key (VK_OEM_2) without Shift yields '/'
        const KeyMapping Slash         = { VK_OEM_2, false };

        // Backslash: the backslash key (VK_OEM_5)
        const KeyMapping Backslash     = { VK_OEM_5, false };

        // Tilde and Grave Accent:
        // VK_OEM_3 corresponds to the key that produces a grave accent (`) unshifted and tilde (~) when shifted.
        const KeyMapping GraveAccent   = { VK_OEM_3, false };
        const KeyMapping Tilde         = { VK_OEM_3, true };
    } // namespace Punctuation

    // Function to convert a single character into its corresponding KeyMapping.
    inline KeyMapping getKeyMappingForChar(char c) {
        if (std::isalpha(c)) {
            // For letters, convert to uppercase to get the base mapping from our Alphabet enum.
            char upper = std::toupper(c);
            KeyMapping mapping = GetAlphabetMapping(static_cast<Alphabet>(upper));
            // Set the shift flag to true if the original character was uppercase.
            mapping.shift = std::isupper(c);
            return mapping;
        } else if (std::isdigit(c)) {
            // For digits, the virtual key codes for '0'-'9' match their ASCII codes.
            return { static_cast<UINT>(c), false };
        } else if (c == ' ') {
            // Space key mapping.
            return { VK_SPACE, false };
        } else {
            // Handle common punctuation characters.
            switch (c) {
                case ',': return Punctuation::Comma;
                case '.': return Punctuation::Period;
                case ';': return Punctuation::Semicolon;
                case ':': return Punctuation::Colon;
                case '!': return Punctuation::Exclamation;
                case '?': return Punctuation::Question;
                case '\'': return Punctuation::Apostrophe;
                case '"': return Punctuation::Quotation;
                case '-': return Punctuation::Hyphen;
                case '_': return Punctuation::Underscore;
                case '(' : return Punctuation::LeftParenthesis;
                case ')' : return Punctuation::RightParenthesis;
                case '[': return Punctuation::LeftBracket;
                case ']': return Punctuation::RightBracket;
                case '{': return Punctuation::LeftBrace;
                case '}': return Punctuation::RightBrace;
                case '/': return Punctuation::Slash;
                case '\\': return Punctuation::Backslash;
                case '`': return Punctuation::GraveAccent;
                case '~': return Punctuation::Tilde;
                case '#': return Punctuation::HashTag;
                default:
                    std::cerr << "Unrecognized character: " << c << std::endl;
                    return { 0, false };
            }
        }
    }

    // Function to simulate a key press (including Shift modifier if needed) using SendInput.
    inline void simulateKeyPress(const KeyMapping &keyMapping) {
        if (keyMapping.vk == 0) return;

        INPUT inputs[4] = {};
        int nInputs = 0;

        // If the key requires the Shift modifier, simulate pressing Shift.
        if (keyMapping.shift) {
            inputs[nInputs].type = INPUT_KEYBOARD;
            inputs[nInputs].ki.wVk = VK_SHIFT;
            inputs[nInputs].ki.dwFlags = 0;
            nInputs++;
        }

        // Simulate key down for the main key.
        inputs[nInputs].type = INPUT_KEYBOARD;
        inputs[nInputs].ki.wVk = keyMapping.vk;
        inputs[nInputs].ki.dwFlags = 0;
        nInputs++;

        // Simulate key up for the main key.
        inputs[nInputs].type = INPUT_KEYBOARD;
        inputs[nInputs].ki.wVk = keyMapping.vk;
        inputs[nInputs].ki.dwFlags = KEYEVENTF_KEYUP;
        nInputs++;

        // If Shift was pressed, simulate releasing Shift.
        if (keyMapping.shift) {
            inputs[nInputs].type = INPUT_KEYBOARD;
            inputs[nInputs].ki.wVk = VK_SHIFT;
            inputs[nInputs].ki.dwFlags = KEYEVENTF_KEYUP;
            nInputs++;
        }

        UINT sent = SendInput(nInputs, inputs, sizeof(INPUT));
        if (sent != static_cast<UINT>(nInputs)) {
            std::cerr << "Error: Only " << sent << " of " << nInputs << " inputs were sent." << std::endl;
        }
    }

    // Simulate typing a string by iterating over each character.
    inline void simulateTyping(const char* text) {
        for (size_t i = 0; text[i] != '\0'; ++i) {
            KeyMapping mapping = getKeyMappingForChar(text[i]);
            if (mapping.vk != 0) {
                simulateKeyPress(mapping);
                Sleep(100);
            }
        }
    }

} // namespace echo

#endif // keys
