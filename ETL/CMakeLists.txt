cmake_minimum_required(VERSION 3.17)
project(DataLib)
set(CMAKE_CXX_STANDARD 20)
set(SOURCE_LIB source/Data.cpp source/DataProcessor.cpp)
set(HEADERS_LIB headers/Data.h headers/DataProcessor.h headers/GeneralizedDataContainer.h)
add_library(DataLib STATIC ${HEADERS_LIB} ${SOURCE_LIB})