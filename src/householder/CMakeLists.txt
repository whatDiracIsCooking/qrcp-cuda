include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

##----------------------------------------------------------------------------##
add_library(HouseHolder
    SHARED
    householder.cu
    householder.h
    identity.cu
    identity.h
)

target_link_libraries(HouseHolder
    PRIVATE
    HelperCuda
    -lcuda
    -lcudart
    -lstdc++
)

target_include_directories(HouseHolder
    PUBLIC 
    ${SRC_DIR}
    ${INCLUDE_DIR}
)
