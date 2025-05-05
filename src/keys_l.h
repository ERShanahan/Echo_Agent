#ifndef KEYMAPPINGS_LINUX_H
#define KEYMAPPINGS_LINUX_H

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/extensions/XTest.h>
#include <cctype>
#include <iostream>
#include <unistd.h>  // for usleep

namespace echo {

    // A helper structure that holds the KeySym and whether Shift is needed.
    struct KeyMapping {
        KeySym keysym;   // X11 keysym
        bool shift;      // true if we must hold Shift
    };

    // Convert an Alphabet character (A–Z) into a KeyMapping.
    inline KeyMapping GetAlphabetMapping(char letter) {
        // Map letter to uppercase Keysym, no shift for base.
        KeySym ks = XStringToKeysym(std::string(1, std::toupper(letter)).c_str());
        return { ks, false };
    }

    // Namespace for punctuation mappings via hard-coded KeySyms.
    namespace Punctuation {
        const KeyMapping Comma         = { XK_comma,           false };
        const KeyMapping Period        = { XK_period,          false };
        const KeyMapping Semicolon     = { XK_semicolon,       false };
        const KeyMapping Colon         = { XK_semicolon,       true  };
        const KeyMapping Exclamation   = { XK_1,                true  };
        const KeyMapping Question      = { XK_slash,            true  };
        const KeyMapping HashTag       = { XK_3,                true  };
        const KeyMapping Apostrophe    = { XK_apostrophe,       false };
        const KeyMapping Quotation     = { XK_apostrophe,       true  };
        const KeyMapping Hyphen        = { XK_minus,            false };
        const KeyMapping Underscore    = { XK_minus,            true  };
        const KeyMapping LeftParenthesis  = { XK_9,             true  };
        const KeyMapping RightParenthesis = { XK_0,             true  };
        const KeyMapping LeftBracket   = { XK_bracketleft,     false };
        const KeyMapping LeftBrace     = { XK_bracketleft,     true  };
        const KeyMapping RightBracket  = { XK_bracketright,    false };
        const KeyMapping RightBrace    = { XK_bracketright,    true  };
        const KeyMapping Slash         = { XK_slash,           false };
        const KeyMapping Backslash     = { XK_backslash,       false };
        const KeyMapping GraveAccent   = { XK_grave,           false };
        const KeyMapping Tilde         = { XK_grave,           true  };
        const KeyMapping Space         = { XK_space,           false };
    }

    // Convert a single character into its KeyMapping.
    inline KeyMapping getKeyMappingForChar(char c) {
        if (std::isalpha(c)) {
            KeyMapping km = GetAlphabetMapping(c);
            km.shift = std::isupper(c);
            return km;
        }
        if (std::isdigit(c)) {
            // digits: '0'..'9' → XK_0..XK_9
            KeySym ks = XStringToKeysym(std::string(1, c).c_str());
            return { ks, false };
        }
        if (c == ' ')
            return Punctuation::Space;

        switch (c) {
            case ',': return Punctuation::Comma;
            case '.': return Punctuation::Period;
            case ';': return Punctuation::Semicolon;
            case ':': return Punctuation::Colon;
            case '!': return Punctuation::Exclamation;
            case '?': return Punctuation::Question;
            case '#': return Punctuation::HashTag;
            case '\'':return Punctuation::Apostrophe;
            case '"': return Punctuation::Quotation;
            case '-': return Punctuation::Hyphen;
            case '_': return Punctuation::Underscore;
            case '(': return Punctuation::LeftParenthesis;
            case ')': return Punctuation::RightParenthesis;
            case '[': return Punctuation::LeftBracket;
            case ']': return Punctuation::RightBracket;
            case '{': return Punctuation::LeftBrace;
            case '}': return Punctuation::RightBrace;
            case '/': return Punctuation::Slash;
            case '\\':return Punctuation::Backslash;
            case '`': return Punctuation::GraveAccent;
            case '~': return Punctuation::Tilde;
            default:
                std::cerr << "Unrecognized character: " << c << std::endl;
                return { NoSymbol, false };
        }
    }

    // Simulate a key press (and optional Shift) via XTest.
    inline void simulateKeyPress(Display* dpy, KeyMapping km) {
        if (!dpy || km.keysym == NoSymbol) return;
        // Convert Keysym → Keycode
        KeyCode kc = XKeysymToKeycode(dpy, km.keysym);
        if (!kc) {
            std::cerr << "Cannot map keysym " << km.keysym << " to keycode\n";
            return;
        }

        // Press Shift if needed
        if (km.shift) {
            KeyCode shiftKc = XKeysymToKeycode(dpy, XK_Shift_L);
            XTestFakeKeyEvent(dpy, shiftKc, True, 0);
        }

        // Press & release the key
        XTestFakeKeyEvent(dpy, kc, True, 0);
        XTestFakeKeyEvent(dpy, kc, False, 0);

        // Release Shift if it was pressed
        if (km.shift) {
            KeyCode shiftKc = XKeysymToKeycode(dpy, XK_Shift_L);
            XTestFakeKeyEvent(dpy, shiftKc, False, 0);
        }

        XFlush(dpy);
    }

    // Simulate typing a NUL-terminated C-string.
    inline void simulateTyping(const char* text) {
        Display* dpy = XOpenDisplay(nullptr);
        if (!dpy) {
            std::cerr << "Cannot open X display\n";
            return;
        }

        for (size_t i = 0; text[i]; ++i) {
            KeyMapping km = getKeyMappingForChar(text[i]);
            simulateKeyPress(dpy, km);
            usleep(100 * 1000);  // 100 ms between keystrokes
        }

        XCloseDisplay(dpy);
    }

} // namespace echo

#endif // KEYMAPPINGS_LINUX_H
