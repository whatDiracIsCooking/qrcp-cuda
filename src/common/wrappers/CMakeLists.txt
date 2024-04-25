##----------------------------------------------------------------------------##
add_library(Wrappers
    SHARED
    cublas.h
    malloc.h
    memcpy.h
    memset.h
)

target_link_libraries(Wrappers
    PUBLIC
    -lcuda
    -lcudart
    -lcublas
    -lstdc++
)

set_target_properties(Wrappers PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Wrappers
    PRIVATE
    ${COMMON_DIR}
)
