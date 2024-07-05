cmake_minimum_required(VERSION 3.14)
project("woolydart" C CXX)

set(LLAMA_BUILD_TESTS Off)
set(LLAMA_BUILD_EXAMPLES Off)
set(LLAMA_BUILD_SERVER Off)

# include upstream llamacpp options
add_subdirectory("llama.cpp")

#set(CMAKE_CXX_VISIBILITY_PRESET default)
#set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)

add_library(woolydart SHARED
    bindings.cpp
    )

target_include_directories(woolydart PRIVATE "llama.cpp")
target_compile_features(woolydart PUBLIC cxx_std_11) # don't bump
target_link_libraries(woolydart PRIVATE common)
