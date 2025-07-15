#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CudaDiffRasterizer" for configuration ""
set_property(TARGET CudaDiffRasterizer APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(CudaDiffRasterizer PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CUDA"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libCudaDiffRasterizer.a"
  )

list(APPEND _cmake_import_check_targets CudaDiffRasterizer )
list(APPEND _cmake_import_check_files_for_CudaDiffRasterizer "${_IMPORT_PREFIX}/lib/libCudaDiffRasterizer.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
