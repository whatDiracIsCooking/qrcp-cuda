include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(TestTools
    SHARED
    test_tools.cpp
    test_tools.h
)

target_include_directories(TestTools
    PUBLIC
    ${COMMON_DIR}
    ${SRC_DIR}
    ${INCLUDE_DIR}
)

target_link_libraries(TestTools
    PRIVATE
    HelperCuda
    -lcuda
    -lcudart
)

add_subdirectory(identity)

##----------------------------------------------------------------------------##
add_executable(HouseHolderTest householder.cpp)

target_include_directories(HouseHolderTest
    PUBLIC
    ${COMMON_DIR}
    ${SRC_DIR}
    ${INCLUDE_DIR}
)

target_link_libraries(HouseHolderTest
    PUBLIC
    HouseHolder
    TestTools
    HelperCuda
    Wrappers
    -lcuda
    -lcudart
    -lcublas
    -lstdc++
)

set_target_properties(HouseHolderTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${TEST_BIN_DIR}
)
