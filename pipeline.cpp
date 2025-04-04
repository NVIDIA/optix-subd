//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#include "pipeline.h"

#include "material/materialCuda.h"
#include "shadingTypes.h"

#include <OptiXToolkit/Util/Exception.h>
#include <embeddedDeviceCode.h>
#include <optix_stack_size.h>

#include <vector>

static void createModuleAndPipelineOptions(Pipeline& pipeline, Params& params, bool enableInstancing)
{
    OptixModuleCompileOptions module_compile_options = {};

    // clang-format off
#if !defined( NDEBUG )
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
    // clang-format on
    pipeline.pipeline_compile_options.usesMotionBlur = false;
    pipeline.pipeline_compile_options.traversableGraphFlags =
        enableInstancing ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline.pipeline_compile_options.numPayloadValues = 4;
    pipeline.pipeline_compile_options.numAttributeValues = 2;
    pipeline.pipeline_compile_options.allowClusteredGeometry = true;

#if !defined( NDEBUG )
    pipeline.pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;
#else
    pipeline.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#endif

    pipeline.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    OptixModuleCompileBoundValueEntry bve;
    bve.pipelineParamOffsetInBytes = offsetof(Params, bound);
    bve.sizeInBytes = sizeof(BoundValues);
    bve.boundValuePtr = &params.bound;
    bve.annotation = "bound values";
    module_compile_options.boundValues = &bve;
    module_compile_options.numBoundValues = 1;
    module_compile_options.maxRegisterCount = 160;

    // Compile modules
    OPTIX_CHECK_LOG2(optixModuleCreate(pipeline.context, &module_compile_options, &pipeline.pipeline_compile_options,
        embeddedDeviceCodeText(), embeddedDeviceCodeSize, LOG, &LOG_SIZE, &pipeline.module));

    OPTIX_CHECK_LOG2(optixModuleCreate(pipeline.context, &module_compile_options, &pipeline.pipeline_compile_options,
        embeddedDeviceCodeCHText(), embeddedDeviceCodeCHSize, LOG, &LOG_SIZE, &pipeline.moduleCH));
}

static void createPipeline(Pipeline& pipeline, Params& params, bool enableInstancing)
{
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;  // primary + occlusion

    constexpr int numProgramGroups = sizeof(ProgramGroups) / sizeof(OptixProgramGroup);

    OPTIX_CHECK_LOG2(optixPipelineCreate(pipeline.context, &pipeline.pipeline_compile_options, &pipeline_link_options,
        &pipeline.program_groups.raygen,  // ptr to first program group
        numProgramGroups, LOG, &LOG_SIZE, &pipeline.pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pipeline.program_groups.raygen, &stack_sizes, pipeline.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pipeline.program_groups.mesh_radiance, &stack_sizes, pipeline.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pipeline.program_groups.miss_radiance, &stack_sizes, pipeline.pipeline));

    uint32_t max_trace_depth = 2; // primary + occlusion
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth, &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));

    uint32_t max_traversal_depth = enableInstancing ? 2 : 1;

    OPTIX_CHECK(optixPipelineSetStackSize(pipeline.pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size, max_traversal_depth));
}

static void createProgramGroups(Pipeline& pipeline)
{
    OptixProgramGroupOptions program_group_options = {};

    //
    // Ray generation
    //
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = pipeline.module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";
        OPTIX_CHECK_LOG2(optixProgramGroupCreate(pipeline.context, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options, LOG, &LOG_SIZE, &pipeline.program_groups.raygen));
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = pipeline.module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG2(optixProgramGroupCreate(pipeline.context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options, LOG, &LOG_SIZE, &pipeline.program_groups.miss_radiance));

    }

    //
    // Hitgroup for cluster
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = pipeline.moduleCH;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OPTIX_CHECK_LOG2(optixProgramGroupCreate(pipeline.context, &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options, LOG, &LOG_SIZE, &pipeline.program_groups.mesh_radiance));
    }
}

static void cleanupProgramGroups(ProgramGroups& program_groups)
{
    if (program_groups.raygen)
        OPTIX_CHECK(optixProgramGroupDestroy(program_groups.raygen));
    if (program_groups.mesh_radiance)
        OPTIX_CHECK(optixProgramGroupDestroy(program_groups.mesh_radiance));
    if (program_groups.miss_radiance)
        OPTIX_CHECK(optixProgramGroupDestroy(program_groups.miss_radiance));

    program_groups.raygen = nullptr;
    program_groups.mesh_radiance = nullptr;
    program_groups.miss_radiance = nullptr;
}

static void createSBT( Pipeline& pipeline, std::span<MaterialCuda const> materials )
{
    using RayGenSbtRecord = Pipeline::RayGenSbtRecord;
    using MissSbtRecord = Pipeline::MissSbtRecord;
    using HitGroupSbtRecord = Pipeline::HitGroupSbtRecord;

    {
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pipeline.sbt.raygenRecord), raygen_record_size));

        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(pipeline.program_groups.raygen, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pipeline.sbt.raygenRecord), &rg_sbt, raygen_record_size,
            cudaMemcpyHostToDevice));
    }

    {
        const size_t miss_record_size = sizeof( MissSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pipeline.sbt.missRecordBase ), miss_record_size ) );

        MissSbtRecord ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( pipeline.program_groups.miss_radiance, &ms_sbt ) );

        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( pipeline.sbt.missRecordBase ), &ms_sbt, miss_record_size,
                                cudaMemcpyHostToDevice ) );

        pipeline.sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
        pipeline.sbt.missRecordCount         = 1;
    }

    {

        // CH for clusters

        std::vector<HitGroupSbtRecord> hitgroup_records;
        if ( materials.empty() )
        {
            HitGroupSbtRecord rec = {};
            rec.data.material = 0;
            OPTIX_CHECK( optixSbtRecordPackHeader( pipeline.program_groups.mesh_radiance, &rec ) );
            hitgroup_records.push_back( rec );
        }
        else
        {
            for( size_t i = 0; i < materials.size(); ++i )
            {
                HitGroupSbtRecord rec = {};
                rec.data.material = materials.data() + i;
                OPTIX_CHECK( optixSbtRecordPackHeader( pipeline.program_groups.mesh_radiance, &rec ) );
                hitgroup_records.push_back( rec );
            }
        }

        const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pipeline.sbt.hitgroupRecordBase),
            hitgroup_record_size * hitgroup_records.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pipeline.sbt.hitgroupRecordBase), hitgroup_records.data(),
            hitgroup_record_size * hitgroup_records.size(), cudaMemcpyHostToDevice));

        pipeline.sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
        pipeline.sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
    }
}

static void cleanupSBT(Pipeline& pipeline)
{
    if (pipeline.sbt.raygenRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pipeline.sbt.raygenRecord)));
    if (pipeline.sbt.hitgroupRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pipeline.sbt.hitgroupRecordBase)));
    if (pipeline.sbt.missRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pipeline.sbt.missRecordBase)));

    pipeline.sbt.raygenRecord = 0;
    pipeline.sbt.hitgroupRecordBase = 0;
    pipeline.sbt.missRecordBase = 0;
}

//
//
//

ProgramGroups::~ProgramGroups()
{
    cleanupProgramGroups(*this);
}

void Pipeline::buildOrUpdate(OptixDeviceContext& icontext, Params& iparams, std::span<MaterialCuda const> materials, bool enableInstancing, bool printSBT)
{
    // destroy old stuff
    if (pipeline)
    {
        cleanup();
    }

    context = icontext;
    params = &iparams;

    createModuleAndPipelineOptions(*this, *params, enableInstancing);
    createProgramGroups(*this);
    createPipeline(*this, *params, enableInstancing);
    createSBT( *this, materials );

}

void Pipeline::cleanup()
{

    cleanupSBT(*this);
    
    if (pipeline)
    {
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = nullptr;
    }
   
    cleanupProgramGroups(program_groups);
    
    if (module)
    {
        OPTIX_CHECK(optixModuleDestroy(module));
        module = nullptr;
    }
    if (moduleCH)
    {
        OPTIX_CHECK(optixModuleDestroy(moduleCH));
        moduleCH = nullptr;
    }
}
