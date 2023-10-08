# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# added by fubuki, dynamic cast
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

# Declares and names the project.

project("ftpipehd_mnn")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

aux_source_directory(. SRC_LIST)  # 添加所有源文件？
aux_source_directory(./utils UTILS_LIST)
aux_source_directory(./models MODELS_LIST)
aux_source_directory(./modules MODULES_LIST)
aux_source_directory(./datasets DATASETS_LIST)
aux_source_directory(./globalStates STATES_LIST)

#file(GLOB_RECURSE SRC_LIST ./*.cpp ./*/*.cpp)
add_library( # Sets the name of the library.
        ftpipehd_mnn

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${SRC_LIST} ${UTILS_LIST} ${MODELS_LIST} ${STATES_LIST} ${MODULES_LIST} ${DATASETS_LIST})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# opencv
include_directories(
        ${CMAKE_SOURCE_DIR}/opencv/include/
)
add_library(libopencv_java4 STATIC IMPORTED)
set_target_properties(
        libopencv_java4
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/opencv/${ANDROID_ABI}/libopencv_java4.so
)

# mnn
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/data)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/datasets)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/exec)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/grad)
# include_directories(${CMAKE_SOURCE_DIR}/mnn/include/models)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/nn)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/optimizer)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/parameters)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/transformer)

# utils
include_directories(${CMAKE_SOURCE_DIR}/utils/include/)

# models
include_directories(${CMAKE_SOURCE_DIR}/models/include/)

# globalStates
include_directories(${CMAKE_SOURCE_DIR}/globalStates/include/)

# modules
include_directories(${CMAKE_SOURCE_DIR}/modules/include/)

# datasets
include_directories(${CMAKE_SOURCE_DIR}/datasets/include/)


add_library(libMNN STATIC IMPORTED)
set_target_properties(libMNN PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN.so)

#add_library(libMNN_CL STATIC IMPORTED)
#set_target_properties(libMNN_CL PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_CL.so)

#add_library(libMNN_Express STATIC IMPORTED)
#set_target_properties(libMNN_Express PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_Express.so)
#
#add_library(libMNN_Train STATIC IMPORTED)
#set_target_properties(libMNN_Train PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNNTrain.so)
#
#add_library(libMNN_CL STATIC IMPORTED)
#set_target_properties(libMNN_CL PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_CL.so)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        ftpipehd_mnn

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
        libopencv_java4
        libMNN)
        # libMNN_CL
#        libMNN_Express
#        libMNN_Train)