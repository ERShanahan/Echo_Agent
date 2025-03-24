# Compiler and flags
CXX       = g++
CXXFLAGS  = -Wall -Wextra -std=c++17 -I.

# Name of the target executable
TARGET    = echo_agent

# Source files and corresponding object files
SRCS      = main.cpp audio.cpp data.cpp matrix.cpp transformer.cpp model.cpp listener.cpp writer.cpp
OBJS      = $(SRCS:.cpp=.o)

# Additional libraries required for WASAPI (e.g., ole32)
LIBS      = -lole32 -L. -static -lopenblas

# Default target
all: $(TARGET)

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

# Compile each .cpp file into a .o object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
