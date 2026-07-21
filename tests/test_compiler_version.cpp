#include <iostream>

int main() {
    // __cplusplus should be greater than the C++23 draft value (202302L)
    std::cout << "Compiled with C++ standard code: " << __cplusplus << "\n";
    
    #if __cplusplus > 202302L
        std::cout << "Success: C++26 mode is active!\n";
    #else
        std::cout << "Warning: Not running in experimental C++26 mode.\n";
    #endif

    return 0;
}