# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import copy
import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime

def get_workload():
    mod, params = relay.testing.resnet.get_workload(num_layers=18)
    return mod, params

def verify(data):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(
            mod, "llvm", params=params, export_graph_module=False)

    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**graph_params)
    module.run()
    out = module.get_output(0).asnumpy()

    return out

def test_cpu():
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib, graph_params = relay.build_module.build(
            mod, "llvm", params=params, export_graph_module=True)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.cpu()
    complied_graph_lib.init(ctx)
    complied_graph_lib.set_input("data", data)
    complied_graph_lib.set_input(**graph_params)
    complied_graph_lib.run()
    out = complied_graph_lib.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_gpu():
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib, graph_params = relay.build_module.build(
            mod, "cuda", params=params, export_graph_module=True)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    complied_graph_lib.init(ctx)
    complied_graph_lib.set_input("data", data)
    complied_graph_lib.set_input(**graph_params)
    complied_graph_lib.run()
    out = complied_graph_lib.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_cpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib, graph_params = relay.build_module.build(
            mod, "llvm", params=params, export_graph_module=True)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.cpu()
    loaded_lib.init(ctx)
    loaded_lib.load_params(loaded_params)
    loaded_lib.set_input("data", data)
    loaded_lib.run()
    out = loaded_lib.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_gpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib, graph_params = relay.build_module.build(
            mod, "cuda", params=params, export_graph_module=True)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    loaded_lib.init(ctx)
    loaded_lib.load_params(loaded_params)
    loaded_lib.set_input("data", data)
    loaded_lib.run()
    out = loaded_lib.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_previous_cpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(
            mod, "llvm", params=params, export_graph_module=False)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.cpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_previous_gpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(
            mod, "cuda", params=params, export_graph_module=False)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_rpc_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib, graph_params = relay.build_module.build(
            mod, "llvm", params=params, export_graph_module=True)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))

    from tvm import rpc
    server = rpc.Server("localhost", use_popen=True)
    remote = rpc.connect(server.host, server.port)
    remote.upload(path_lib)
    loaded_lib = remote.load_module(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = remote.cpu()
    loaded_lib.init(ctx)
    loaded_lib.load_params(loaded_params)
    loaded_lib.set_input("data", data)
    loaded_lib.run()
    out = loaded_lib.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_previous_rpc_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(
            mod, "llvm", params=params, export_graph_module=False)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))

    from tvm import rpc
    server = rpc.Server("localhost", use_popen=True)
    remote = rpc.connect(server.host, server.port)
    remote.upload(path_lib)
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = remote.load_module(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = remote.cpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


if __name__ == "__main__":
    test_cpu()
    test_cpu_export(".so")
    test_cpu_export(".tar")
    test_gpu()
    test_gpu_export(".so")
    test_gpu_export(".tar")
    test_rpc_export(".so")
    test_rpc_export(".tar")
    test_previous_cpu_export(".so")
    test_previous_cpu_export(".tar")
    test_previous_gpu_export(".so")
    test_previous_gpu_export(".tar")
    test_previous_rpc_export(".so")
    test_previous_rpc_export(".tar")



