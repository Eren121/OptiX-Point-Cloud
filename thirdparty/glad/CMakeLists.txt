set(SRC glad.c glad.h khrplatform.h)

add_library(glad ${SRC})
target_include_directories(glad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})