cc_library(
  arg_map_context
  SRCS arg_map_context.cc
  DEPS phi_enforce)
cc_library(
  op_utils
  SRCS op_utils.cc
  DEPS arg_map_context enforce)

set(convert_utils_deps data_type place op_utils)

if(WITH_GPU)
  set(convert_utils_deps ${convert_utils_deps} phi_gpu_info)
elseif(WITH_ROCM)
  set(convert_utils_deps ${convert_utils_deps} phi_gpu_info)
elseif(WITH_XPU)
  set(convert_utils_deps ${convert_utils_deps} phi_xpu_info)
endif()
if(WITH_CUSTOM_DEVICE)
  set(convert_utils_deps ${convert_utils_deps} device_manager)
endif()
cc_library(
  convert_utils
  SRCS convert_utils.cc
  DEPS ${convert_utils_deps})
