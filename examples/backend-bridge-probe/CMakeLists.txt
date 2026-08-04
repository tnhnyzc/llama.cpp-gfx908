set(TARGET llama-backend-bridge-probe)
add_executable(${TARGET} backend-bridge-probe.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
