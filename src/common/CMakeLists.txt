##----------------------------------------------------------------------------##
add_library(HelperCuda
    SHARED
    helper_cuda.h
    streamEvent.h
    cusolverCheck.h
)

target_link_libraries(HelperCuda
    PUBLIC
    -lcuda
    -lcudart
    -lcublas
    -lcusolver
    -lstdc++
)

set_target_properties(HelperCuda PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(HelperCuda
    PUBLIC
    ${SRC_DIR}/common/include
)

add_subdirectory(wrappers)
