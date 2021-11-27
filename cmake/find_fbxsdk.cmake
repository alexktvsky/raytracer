include(cmake/find_libxml2.cmake)

set(FBXSDK_SHARED 1)

if (UNIX AND NOT APPLE)
    # Assume we are on Linux
    if (FBXSDK_SHARED)
        set(FBXSDK_LIB_EXTENSION "so")
    else()
        set(FBXSDK_LIB_EXTENSION "a")
    endif()

    if (CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
        set(FBXSDK_ARCH x64)
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL x86)
        set(FBXSDK_ARCH x86)
    else()
        message(FATAL_ERROR "Unknow CPU arch ${CMAKE_SYSTEM_PROCESSOR}.")
    endif()

    set(FBXSDK_INCLUDE_DIR /usr/*/include/fbxsdk.h)
    file(GLOB FBXSDK_INCLUDE_PATH ${FBXSDK_INCLUDE_DIR})

    set(FBXSDK_LIBDIR /usr/*/lib/libfbxsdk.${FBXSDK_LIB_EXTENSION})
    file(GLOB FBXSDK_LIBRARY ${FBXSDK_LIBDIR})
    if (NOT FBXSDK_LIBDIR)
        set(FBXSDK_LIBDIR /usr/*/gcc/${FBXSDK_ARCH}/${CMAKE_BUILD_TYPE}/libfbxsdk.${FBXSDK_LIB_EXTENSION})
        file(GLOB FBXSDK_LIBRARY ${FBXSDK_LIBDIR})
    endif()

elseif (APPLE)
    if (FBXSDK_SHARED)
        set(FBXSDK_LIB_EXTENSION "dylib")
    else()
        set(FBXSDK_LIB_EXTENSION "a")
    endif()

    set(FBXSDK_INCLUDE_DIR /Applications/Autodesk/FBX\ SDK/*/include/fbxsdk.h)
    set(FBXSDK_LIBDIR /Applications/Autodesk/FBX\ SDK/*/lib/clang/${CMAKE_BUILD_TYPE}/libfbxsdk.${FBXSDK_LIB_EXTENSION})
    file(GLOB FBXSDK_INCLUDE_PATH ${FBXSDK_INCLUDE_DIR})
    file(GLOB FBXSDK_LIBRARY ${FBXSDK_LIBDIR})

elseif (WIN32)
    if (FBXSDK_SHARED)
        set(FBXSDK_LIB_EXTENSION "dll")
    else()
        set(FBXSDK_LIB_EXTENSION "lib")
    endif()

    if (CMAKE_SYSTEM_PROCESSOR STREQUAL AMD64)
        set(FBXSDK_ARCH x64)
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL x86)
        set(FBXSDK_ARCH x86)
    else()
        message(FATAL_ERROR "Unknow CPU arch ${CMAKE_SYSTEM_PROCESSOR}.")
    endif()

    set(FBXSDK_INCLUDE_DIR C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.0.1/include/fbxsdk.h)
    set(FBXSDK_LIBDIR C:/Program\ Files/Autodesk/FBX/FBX\ SDK/*/lib/*/${FBXSDK_ARCH}/${CMAKE_BUILD_TYPE}/libfbxsdk.${FBXSDK_LIB_EXTENSION})
    file(GLOB FBXSDK_INCLUDE_PATH ${FBXSDK_INCLUDE_DIR})
    file(GLOB FBXSDK_LIBRARY ${FBXSDK_LIBDIR})
endif()


cmake_path(REMOVE_FILENAME FBXSDK_INCLUDE_PATH)

if (FBXSDK_INCLUDE_PATH AND FBXSDK_LIBRARY)
    set(FBXSDK_FOUND 1)
    list(APPEND CMAKE_REQUIRED_INCLUDE_PATHS ${FBXSDK_INCLUDE_PATH})
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${FBXSDK_LIBRARY})
endif()
