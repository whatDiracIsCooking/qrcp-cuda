##----------------------------------------------------------------------------##
add_library(Identity
    SHARED
    identity.cu
    identity.h
)

target_link_libraries(Identity
    PRIVATE
    HelperCuda
    -lcuda
    -lcudart
    -lstdc++
)

target_include_directories(Identity
    PUBLIC 
    ${SRC_DIR}
    ${INCLUDE_DIR}
)
