##----------------------------------------------------------------------------##
add_executable(IdentityTest identity.cpp)

target_include_directories(IdentityTest
    PRIVATE
    ${COMMON_DIR}
    ${INCLUDE_DIR}
    ${SRC_DIR}
    ${TEST_DIR}
)

target_link_libraries(IdentityTest
    PUBLIC
    Identity
    TestTools
    HelperCuda
    Wrappers
    -lcuda
    -lcudart
    -lcublas
    -lstdc++
)

set_target_properties(IdentityTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${TEST_BIN_DIR}
)
