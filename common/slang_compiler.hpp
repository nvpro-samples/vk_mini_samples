/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <slang.h>
#include <slang-com-ptr.h>

#include <vector>
#include <string>

// SlangCompiler class is used to compile Slang source code to SPIR-V code.

class SlangCompiler
{
private:
  slang::IGlobalSession* m_globalSession{};
  slang::ISession*       m_session{};

public:
  SlangCompiler()
  {
    slang::createGlobalSession(&m_globalSession);
    newSession();
  }

  ~SlangCompiler()
  {
    if(m_session)
      m_session->release();
    if(m_globalSession)
      m_globalSession->release();
  }

  slang::IGlobalSession* globalSession() { return m_globalSession; }
  slang::ISession*       session() { return m_session; }

  void newSession(const std::vector<std::string>&                  searchStringPaths = {},
                  const std::vector<slang::PreprocessorMacroDesc>& preprocessorMacro = {})
  {
    if(m_session)
      m_session->release();
    nvh::ScopedTimer st(__FUNCTION__);
    // Next we create a compilation session to generate SPIRV code from Slang source.
    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc  targetDesc  = {};
    targetDesc.format              = SLANG_SPIRV;
    targetDesc.profile             = m_globalSession->findProfile("spirv_1_5");
    targetDesc.flags               = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    sessionDesc.targets            = &targetDesc;
    sessionDesc.targetCount        = 1;
    // sessionDesc.allowGLSLSyntax    = true;

    // Search paths
    std::vector<const char*> searchPaths;
    if(!searchStringPaths.empty())
    {
      searchPaths.reserve(searchStringPaths.size());
      for(const auto& str : searchStringPaths)
      {
        searchPaths.push_back(str.c_str());
      }
      searchPaths.push_back(PROJECT_RELDIRECTORY);
      sessionDesc.searchPathCount = SlangInt(searchPaths.size());
      sessionDesc.searchPaths     = searchPaths.data();
    }

    // Preprocessor
    sessionDesc.preprocessorMacroCount = SlangInt(preprocessorMacro.size());
    sessionDesc.preprocessorMacros     = preprocessorMacro.data();


    m_globalSession->createSession(sessionDesc, &m_session);
  }

  slang::ICompileRequest* createCompileRequest(const std::string& filePath,
                                               const std::string& entryPointName = "main",
                                               SlangStage         stage          = SLANG_STAGE_COMPUTE)
  {
    nvh::ScopedTimer        st(__FUNCTION__);
    slang::ICompileRequest* compileRequest{};
    m_session->createCompileRequest(&compileRequest);

    // Add source file
    compileRequest->addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
    compileRequest->addTranslationUnitSourceFile(0, filePath.c_str());

    // Set entry point
    compileRequest->addEntryPoint(0, entryPointName.c_str(), stage);
    compileRequest->setTargetForceGLSLScalarBufferLayout(0, true);

    return compileRequest;
  }

  // Compile Slang source code to SPIR-V code
  bool getSpirvCode(slang::ICompileRequest* compileRequest, std::vector<uint32_t>& spirvCode)
  {
    nvh::ScopedTimer st(__FUNCTION__);

    // Get SPIR-V code
    size_t      codeSize = 0;
    const void* codePtr  = compileRequest->getEntryPointCode(0, &codeSize);

    spirvCode.resize(codeSize / sizeof(uint32_t));
    memcpy(spirvCode.data(), codePtr, codeSize);

    compileRequest->release();
    return true;
  }

  bool compileModule(const std::string&     moduleName,
                     const std::string&     entryPointName,  // "main"
                     std::vector<uint32_t>& spirvCode,
                     std::string&           error_msg)
  {
    slang::IModule* slangModule = nullptr;
    {  // Loading the Slang shader file
      nvh::ScopedTimer            st1("Slang Load Module");
      Slang::ComPtr<slang::IBlob> diagnosticBlob;
      slangModule = m_session->loadModule(moduleName.c_str(), diagnosticBlob.writeRef());
      if(diagnosticBlob != nullptr)
        error_msg = (const char*)diagnosticBlob->getBufferPointer();
      if(!slangModule)
        return false;
    }

    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    SlangResult result = slangModule->findEntryPointByName(entryPointName.c_str(), entryPoint.writeRef());
    if(SLANG_FAILED(result))
      return false;

    std::vector<slang::IComponentType*> componentTypes;
    componentTypes.push_back(slangModule);
    componentTypes.push_back(entryPoint);  // index 0


    Slang::ComPtr<slang::IComponentType> composedProgram;
    {
      nvh::ScopedTimer            st2("Slang Compose");
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      SlangResult result = m_session->createCompositeComponentType(componentTypes.data(), componentTypes.size(),
                                                                   composedProgram.writeRef(), diagnosticsBlob.writeRef());
      if(diagnosticsBlob != nullptr)
        error_msg = (const char*)diagnosticsBlob->getBufferPointer();
      if(SLANG_FAILED(result) || !error_msg.empty())
        return false;
    }

    {
      nvh::ScopedTimer            st3("Slang Get Entry Point");
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      Slang::ComPtr<slang::IBlob> outCode;
      SlangResult result = composedProgram->getEntryPointCode(0, 0, outCode.writeRef(), diagnosticsBlob.writeRef());
      if(diagnosticsBlob != nullptr)
        error_msg = (const char*)diagnosticsBlob->getBufferPointer();
      if(SLANG_FAILED(result) || !error_msg.empty())
        return false;

      size_t codeSize = outCode->getBufferSize() / sizeof(uint32_t);
      spirvCode.resize(codeSize);
      memcpy(spirvCode.data(), outCode->getBufferPointer(), outCode->getBufferSize());
    }
    return true;
  }
};
