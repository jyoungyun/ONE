add_subdirectory(core)
add_subdirectory(frontend)
add_subdirectory(cli)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_subdirectory(test)
