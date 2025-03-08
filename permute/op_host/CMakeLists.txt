set(OP_DEF_SRC 
        ${CMAKE_CURRENT_SOURCE_DIR}/permute_custom_def.cpp
)

opbuild(OPS_SRC ${OP_DEF_SRC}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)


set(OP_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/permute_custom_proto.cpp)
file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)
 
add_library(cust_op_proto SHARED
    $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
    ${OP_DEF_SRC}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)


set(OP_TILING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/permute_custom_tiling.cpp)

add_library(cust_optiling SHARED ${OP_DEF_SRC})
target_include_directories(cust_optiling PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
)
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)


file(GLOB aclnn_src ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_*.cpp)
file(GLOB aclnn_inc ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_*.h)
set(L0_OP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/permute_custom.cpp)

add_library(cust_opapi SHARED ${aclnn_src} ${L0_OP_SRC})

target_include_directories(cust_opapi PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn_kernels
)

if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_opapi PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase)

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
file(GLOB GROUP_PROTO_HEADERS ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
if (GROUP_PROTO_HEADERS)
        install(FILES ${GROUP_PROTO_HEADERS}
                DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
endif()
install(TARGETS cust_optiling
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
install(TARGETS cust_opapi
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
install(FILES ${aclnn_inc}
        DESTINATION packages/vendors/${vendor_name}/op_api/include)
