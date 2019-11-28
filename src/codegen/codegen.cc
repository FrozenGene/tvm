/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen.cc
 * \brief Common utilities to generated C style code.
 */
#include <tvm/codegen.h>
#include <tvm/ir_pass.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/build_module.h>
#include <dmlc/memory_io.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <list>

namespace tvm {
namespace codegen {

runtime::Module Build(const Array<LoweredFunc>& funcs,
                      const std::string& target) {
  std::string mode = target;
  size_t pos = mode.find(' ');
  if (pos != std::string::npos) {
    mode = mode.substr(0, pos);
  }
  Array<LoweredFunc> transformed_funcs;
  for (const auto& x : funcs) {
    if (BuildConfig::Current()->disable_assert) {
      auto func = ir::SkipAssert(x);
      transformed_funcs.push_back(func);
    }
  }
  std::string build_f_name = "codegen.build_" + mode;
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "Target " << target << " is not enabled";
  runtime::Module m = transformed_funcs.empty() ?
                      (*bf)(funcs, target) :
                      (*bf)(transformed_funcs, target);
  return m;
}

class ImportTree final {
 public:
  ImportTree(uint64_t vertices) : vertices_(vertices), root_(0) {
    adj_list_.resize(vertices);
  }

  void AddEdge(uint64_t src, uint64_t dest) {
    adj_list_[src].push_back(dest);
  }

  void SetRoot(uint64_t root) {
    root_ = root;
  }

  uint64_t GetRoot() const {
    return root_;
  }

  uint64_t GetNumVertices() const {
    return vertices_;
  }

  void BFS(uint64_t v, std::vector<uint64_t>* order) {
    std::vector<bool> visited(vertices_, false);
    std::list<uint64_t> queue;

    visited[v] = true;
    queue.push_back(v);

    while (!queue.empty()) {
      v = queue.front();
      order->push_back(v);
      queue.pop_front();

      for (auto i = adj_list_[v].begin(); i != adj_list_[v].end(); ++i) {
        if (!visited[*i]) {
          visited[*i] = true;
          queue.push_back(*i);
        }
      }
    }
  }

 private:
  uint64_t vertices_;
  uint64_t root_;
  std::vector<std::list<uint64_t>> adj_list_;
};

std::string PackImportsToC(const runtime::Module& mod, bool system_lib) {
  std::string bin;
  dmlc::MemoryStringStream ms(&bin);
  dmlc::Stream* stream = &ms;
  std::string mod_type_key = mod->type_key();
  /*
   * Support:
   * X Module: import_modules
   *     llvm-module: import_modules
   *         cuda_module
   *         opencl_module
   *         ...
   *
   * llvm-module: import_modules
   *     cuda_module
   *     opencl_module
   *     ...
   *
   */
  uint64_t mod_size = mod->imports().size() + 1;
  for(const auto& m : mod->imports()) {
    CHECK(m->imports().size() == 0U || m->imports().size() == 1U)
      << "Only support one-level / two-levels hierarchy";
    mod_size += m->imports().size();
  }
  ImportTree import_tree(mod_size);
  uint64_t mod_index = 1;
  uint64_t next_mod_index = mod_index + 1;
  // construct import tree while keeping compatibility of old behaviour
  if (mod_type_key != "llvm" && mod_type_key != "c") {
    import_tree.SetRoot(mod_index);
    for(runtime::Module im : mod->imports()) {
      // DSO module
      if (!strcmp(im->type_key(), "llvm") || !strcmp(im->type_key(),"c")) {
        import_tree.AddEdge(mod_index++, 0);
      } else {
        mod_type_key = im->type_key();
        import_tree.AddEdge(mod_index++, next_mod_index++);
      }
      CHECK(im->imports().size() == 0U || im->imports().size() == 1U)
           << "Only support one-level / two-levels hierarchy";
      for(runtime::Module i_subm : im->imports()) {
        CHECK(i_subm->imports().size() == 0U)
          << "Only support simply one-level hierarchy";
        // DSO module
        if (!strcmp(im->type_key(), "llvm") || !strcmp(im->type_key(), "c")) {
          import_tree.AddEdge(0, mod_index++);
        } else {
          import_tree.AddEdge(mod_index++, next_mod_index++);
        }
      }
    }
  }

  if (import_tree.GetRoot()) {
    std::vector<uint64_t> module_order;
    import_tree.BFS(import_tree.GetRoot(), &module_order);
    stream->Write(import_tree.GetNumVertices());
    stream->Write(module_order);
  } else {
    // no import tree. Write 0.
    stream->Write(uint64_t(0));
    uint64_t sz = static_cast<uint64_t>(mod->imports().size());
    stream->Write(sz);
  }


  if (import_tree.GetRoot()) {
    CHECK(mod_type_key != "llvm" && mod_type_key != "c");
    stream->Write(mod_type_key);
    const_cast<runtime::Module&>(mod)->SaveToBinary(stream);
    for (runtime::Module im : mod->imports()) {
      mod_type_key = im->type_key();
      if (mod_type_key != "llvm") {
        stream->Write(mod_type_key);
        im->SaveToBinary(stream);
      }
      for(runtime::Module i_subm : im->imports()) {
        mod_type_key = i_subm->type_key();
        if (mod_type_key != "llvm") {
          stream->Write(mod_type_key);
          i_subm->SaveToBinary(stream);
        }
      }
    }
  } else {
    for (runtime::Module im : mod->imports()) {
      CHECK_EQ(im->imports().size(), 0U)
        << "Only support simply one-level hierarchy";
      std::string tkey = im->type_key();
      stream->Write(tkey);
      im->SaveToBinary(stream);
    }
  }
  // translate to C program
  std::ostringstream os;
  os << "#ifdef _WIN32\n"
     << "#define TVM_EXPORT __declspec(dllexport)\n"
     << "#else\n"
     << "#define TVM_EXPORT\n"
     << "#endif\n";
  os << "#ifdef __cplusplus\n"
     << "extern \"C\" {\n"
     << "#endif\n";
  os << "TVM_EXPORT extern const unsigned char " << runtime::symbol::tvm_dev_mblob << "[];\n";
  uint64_t nbytes = bin.length();
  os << "const unsigned char " << runtime::symbol::tvm_dev_mblob
     << "[" << bin.length() + sizeof(nbytes) << "] = {\n  ";
  os << std::hex;
  size_t nunit = 80 / 4;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    // sperators
    if (i != 0) {
      os << ",";
    }
    os << "0x" << ((nbytes >> (i * 8)) & 0xffUL);
  }
  for (size_t i = 0; i < bin.length(); ++i) {
    // sperators
    if ((i + sizeof(nbytes)) % nunit == 0) {
      os << ",\n  ";
    } else {
      os << ",";
    }
    int c = bin[i];
    os << "0x" << (c & 0xff);
  }
  os << "\n};\n";
  if (system_lib) {
    os << "extern int TVMBackendRegisterSystemLibSymbol(const char*, void*);\n";
    os << "static int " << runtime::symbol::tvm_dev_mblob << "_reg_ = "
       << "TVMBackendRegisterSystemLibSymbol(\"" << runtime::symbol::tvm_dev_mblob << "\", (void*)"
       << runtime::symbol::tvm_dev_mblob << ");\n";
  }
  os << "#ifdef __cplusplus\n"
     << "}\n"
     << "#endif\n";
  return os.str();
}
}  // namespace codegen
}  // namespace tvm
