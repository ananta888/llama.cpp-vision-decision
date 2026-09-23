# shared engine: used by llama-parallel-decision and by llama-server's /decision endpoint
add_library(llama-decision STATIC decision-engine.cpp decision-engine.h)
target_include_directories(llama-decision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# the playground page, served by llama-server at /decision-playground
set(DECISION_PLAYGROUND ${CMAKE_CURRENT_SOURCE_DIR}/playground/index.html)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${DECISION_PLAYGROUND})
file(READ ${DECISION_PLAYGROUND} DECISION_PLAYGROUND_HTML)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/decision-playground.h
    "#pragma once\n\nstatic const char decision_playground_html[] = R\"PLAYGROUND(${DECISION_PLAYGROUND_HTML})PLAYGROUND\";\n")
target_include_directories(llama-decision PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(llama-decision PUBLIC llama-common llama)
target_compile_features(llama-decision PUBLIC cxx_std_17)
# llama-server links it into a shared library (libllama-server-impl)
set_target_properties(llama-decision PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGET llama-parallel-decision)
add_executable(${TARGET} parallel-decision.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama-decision llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
