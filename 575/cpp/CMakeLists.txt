cmake_minimum_required(VERSION 3.16)
project(bubble_test_simulations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)

add_executable(multi_market_bsadf_gsadf multi_market_bsadf_gsadf.cpp)
target_link_libraries(multi_market_bsadf_gsadf PRIVATE Threads::Threads)

add_executable(bsadf_monte_carlo_lag0 bsadf_monte_carlo_lag0.cpp)
target_link_libraries(bsadf_monte_carlo_lag0 PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bsadf_monte_carlo_lag0 PRIVATE OpenMP::OpenMP_CXX)
endif()
