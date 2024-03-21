#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

import coremltools as ct
from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil.mil.passes.tests.test_passes import CONSTEXPR_FUNCS
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource

np.random.seed(0)

def test_single_layer_example():
    batch_size, input_dim, output_dim = 2, 4, 2

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(batch_size, input_dim)),]
    )
    def prog(x):
        # Weight
        W_val = (
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            .reshape(input_dim, output_dim)
            .T.astype(np.float32)
        )
        W = mb.const(val=W_val, name="const_W")

        # bias
        b_val = np.array([-0.5, 0.5]).astype(np.float32)
        b = mb.const(val=b_val, name="const_b")

        return mb.linear(x=x, weight=W, bias=b, name="lin")

    logger.info("prog:\n" + str(prog))

    mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

    feed_dict = {
        "x": np.random.rand(batch_size, input_dim).astype(np.float32),
    }
    assert mlmodel is not None

    if ct.utils._is_macos():
        prediction = mlmodel.predict(feed_dict)
        assert len(prediction) == 1


def test_conv_example():
    batch, C_in, C_out, H, W = 2, 2, 3, 7, 10
    kH, kW = 3, 5
    img_shape, seq_shape = (batch, C_in, H, W), (batch, C_in, H)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=img_shape), mb.TensorSpec(shape=seq_shape),]
    )
    def prog(img, seq):
        ## 2D convolution
        # Weight
        W_2d = np.random.rand(C_out, C_in, kH, kW).astype(np.float32)
        W_2d = mb.const(val=W_2d, name="const_W")

        # Test 1: provide only required arguments.
        conv1 = mb.conv(x=img, weight=W_2d, pad_type="valid")
        logger.info("conv1 shape: {}".format(conv1.shape))

        # Test 2: stride > 1
        conv2 = mb.conv(x=img, weight=W_2d, pad_type="valid", strides=[2, 3])
        logger.info("conv2 shape: {}".format(conv2.shape))

        # Test 3: same padding
        conv3 = mb.conv(x=img, weight=W_2d, pad_type="same", strides=[2, 3])
        logger.info("conv3 shape: {}".format(conv3.shape))

        # Test max_pool
        pool1 = mb.max_pool(
            x=img, kernel_sizes=[kH, kW], pad_type="valid", strides=[2, 3]
        )
        logger.info("pool1 shape: {}".format(pool1.shape))

        # Test max_pool
        pool2 = mb.max_pool(
            x=img, kernel_sizes=[kH, kW], pad_type="same", strides=[2, 3]
        )
        logger.info("pool2 shape: {}".format(pool2.shape))

        ## 1D convolution
        W_1d = np.random.rand(C_out, C_in, kH).astype(np.float32)
        W_1d = mb.const(val=W_1d, name="const_W_1d")
        logger.info("W_1d val: {}".format(W_1d.val))

        # Test 4: provide only required arguments for 1D.
        conv4 = mb.conv(x=seq, weight=W_1d, pad_type="valid")

        logger.info("conv4 shape: {}".format(conv4.shape))

        return conv1, conv2, conv3, pool1, pool2, conv4

    # rdar://105988903 ([Infra] re-enable the test_conv_example unit test on M1 with compute_units=ALL)
    mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork", compute_units=ct.ComputeUnit.CPU_ONLY)

    feed_dict = {
        "img": np.random.rand(*img_shape).astype(np.float32),
        "seq": np.random.rand(*seq_shape).astype(np.float32),
    }
    assert mlmodel is not None

    if ct.utils._is_macos():
        prediction = mlmodel.predict(feed_dict)
        assert len(prediction) == 6


def test_while_example():
    def body(a, b):
        return mb.add(x=a, y=b), b

    def cond(a, b):
        a_mean = mb.reduce_mean(x=a, axes=[0, 1])
        b_mean = mb.reduce_mean(x=b, axes=[0, 1])
        return mb.less(x=a_mean, y=b_mean)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2)),]
    )
    def prog(a, b):
        return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

    logger.info("prog:\n" + str(prog))

    mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

    feed_dict = {
        "a": np.random.rand(1, 2).astype(np.float32),
        "b": np.random.rand(1, 2).astype(np.float32),
    }
    assert mlmodel is not None

    if ct.utils._is_macos():
        prediction = mlmodel.predict(feed_dict)
        assert len(prediction) == 2

def test_reserved_node_names():
    @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
    def prog(x):
        return mb.square(x=x, name="tensor")

    mlmodel = ct.convert(
        prog, source="milinternal", convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_ONLY
    )

    feed_dict = {
        "x": np.random.rand(10, 20).astype(np.float32),
    }
    assert mlmodel is not None

    if ct.utils._is_macos():
        prediction = mlmodel.predict(feed_dict)
        assert len(prediction) == 1

def get_simple_topk_program(opset_version=None):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=opset_version)
    def prog(x):
        x = mb.topk(x=x, k=1, axis=-1, ascending=True)
        return x
    return prog

def get_simple_pixel_unshuffle_program(opset_version=None):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=opset_version)
    def prog(x):
        x = mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(2))
        return x
    return prog

def get_simple_topk_pixel_unshuffle_program(opset_version=None):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=opset_version)
    def prog(x):
        x = mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(2))
        x = mb.topk(x=x, k=1, axis=-1, ascending=True)
        return x
    return prog

def get_simple_nested_block_program(opset_version=None):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=opset_version)
    def prog(x):
        def true_fn():
            topk, _ = mb.topk(x=x, k=1, axis=-1, ascending=True)
            return mb.add(x=topk, y=1.)

        def false_fn():
            topk, _ = mb.topk(x=x, k=1, axis=-1, ascending=True)
            return mb.add(x=topk, y=2.)

        shape = mb.shape(x=x)
        rank = mb.shape(x=shape)
        pred = mb.squeeze(x=rank)
        return mb.cond(pred=mb.cast(x=pred, dtype="bool"), _true_fn=true_fn, _false_fn=false_fn)
    return prog

class TestMILProgramVersionHandling:
    """
    Test basic functionality of opset version handling in pymil
    """
    @staticmethod
    def test_multi_versions_op_selection():
        '''
        Builder should pick up the right version of op based on opset_version
        '''
        # pick up the oldest version (iOS13) topk by default
        prog = get_simple_topk_program()
        main_func = prog.functions["main"]
        topk_op = main_func.find_ops(op_type="topk")[0]
        assert topk_op.opset_version == ct.target.iOS13

        # pick up iOS13 version topk
        prog = get_simple_topk_program(opset_version=ct.target.iOS15)
        main_func = prog.functions["main"]
        topk_op = main_func.find_ops(op_type="topk")[0]
        assert topk_op.opset_version == ct.target.iOS13

        # pick up iOS16 version topk
        prog = get_simple_topk_program(opset_version=ct.target.iOS16)
        main_func = prog.functions["main"]
        topk_op = main_func.find_ops(op_type="topk")[0]
        assert topk_op.opset_version == ct.target.iOS16

    @staticmethod
    def test_pymil_front_end_conversion():
        prog = get_simple_topk_pixel_unshuffle_program(opset_version=ct.target.iOS16)
        mlmodel = ct.convert(
            prog, minimum_deployment_target=ct.target.iOS16, compute_units=ct.ComputeUnit.CPU_ONLY
        )

    @staticmethod
    def test_nested_block_opset_version_selection():
        # pick up the oldest version (iOS13) topk by default
        prog = get_simple_nested_block_program()
        main_func = prog.functions["main"]
        topk_ops = main_func.find_ops(op_type="topk")
        assert all([topk.opset_version == ct.target.iOS13 for topk in topk_ops])

        # pick up iOS16 version topk
        prog = get_simple_nested_block_program(opset_version=ct.target.iOS16)
        main_func = prog.functions["main"]
        topk_ops = main_func.find_ops(op_type="topk")
        assert all([topk.opset_version == ct.target.iOS16 for topk in topk_ops])

    @staticmethod
    def test_pymil_opset_version_inference():
        '''
        The program consist of pixel_unshuffle should be inferred as an iOS16 version program
        '''
        prog = get_simple_pixel_unshuffle_program()
        assert prog.functions["main"].opset_version == ct.target.iOS16

        expected_err_str = (
            "Please update the minimum_deployment_target to coremltools.target.iOS16, "
            "since op pixel_unshuffle is only available in opset coremltools.target.iOS16 or newer."
        )
        with pytest.raises(ValueError, match=expected_err_str):
            mlmodel = ct.convert(
                prog, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_ONLY
            )

    @staticmethod
    def test_pymil_front_end_conversion_early_error_out():
        prog = get_simple_topk_pixel_unshuffle_program(opset_version=ct.target.iOS16)
        expected_err_str = (
            "Please update the minimum_deployment_target to coremltools.target.iOS16, "
            "since op pixel_unshuffle is only available in opset coremltools.target.iOS16 or newer."
        )
        with pytest.raises(ValueError, match=expected_err_str):
            mlmodel = ct.convert(
                prog,
                minimum_deployment_target=ct.target.iOS15,
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )

    @staticmethod
    def test_unsupported_op_early_error_out():
        '''
        We should error out at the point when Builder tries to add an op which is only supported in a newer spec version
        '''
        expected_err_str = (
            "No available version for pixel_unshuffle in the coremltools.target.iOS15 opset. "
            "Please update the minimum_deployment_target to at least coremltools.target.iOS16"
        )
        with pytest.raises(ValueError, match=expected_err_str):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=ct.target.iOS15)
            def prog(x):
                x = mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(2))
                return x

    @staticmethod
    def test_bulid_non_compatible_program_early_error_out():
        '''
        `mb.program` API should detect potential non compatible ops in the program, and error out early
        In this example, `pixel_unshuffle` is an iO16 op, and `topk` has iOS13 and iOS16 version.
        If the builder version is not set, it is picking up the iOS13 version of topk, which would
        potentially create an invalid program.
        In this case, `mb.program` should error out, and tell the user to set `opset_version=target.iOS16`
        '''
        expected_err_str = (
            "Op topk with an out of date version coremltools.target.iOS13 is detected. Please use @mb.program\(input_specs=..., opset_version=coremltools.target.iOS16\)"
        )
        with pytest.raises(ValueError, match=expected_err_str):
            get_simple_topk_pixel_unshuffle_program()

class TestMILBuilderAPI:
    """
    Test the basic builder API.
    """
    def test_create_function(self):
        """
        Test mb.function API
        """
        @mb.function(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def func(x):
            return mb.add(x=x, y=0.0)

        assert isinstance(func, Function)
        assert len(func.operations) == 2  # add, const
        assert len(func.inputs) == 1
        assert len(func.outputs) == 1

    def test_create_program(self):
        """
        Test mb.program API
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            return mb.add(x=x, y=0.0)

        assert isinstance(prog, Program)
        func = prog.functions["main"]
        assert len(func.operations) == 2  # add, const
        assert len(func.inputs) == 1
        assert len(func.outputs) == 1

    def test_create_program_function_name(self):
        """
        If ``function_name`` is not provide, mb.program creates function with name "main" by default.
        """
        # defaults to "main"
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x0):
            return x0

        assert len(prog.functions) == 1
        assert "main" in prog.functions

        # user can also provide function_name
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))], function_name="good_function")
        def prog(x0):
            return x0

        assert len(prog.functions) == 1
        assert "good_function" in prog.functions

    def test_program_with_multiple_functions(self):
        """
        Basic creation of a program with multiple functions
        """
        @mb.function(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def func_1(x):
            return x

        @mb.function(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def func_2(x):
            return x

        @mb.function(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def func_3(x):
            return x

        prog = mil.Program()
        prog.add_function("func_1", func_1)
        prog.add_function("func_2", func_2)
        prog.add_function("func_3", func_3)

        assert set(prog.functions.keys()) == set(["func_1", "func_2", "func_3"])

    def test_error_out_incompatible_functions(self):
        """
        ``add_function`` should error out when a function with different
        opset is added to a program.
        """
        @mb.function(input_specs=[mb.TensorSpec(shape=(2, 4))], opset_version=ct.target.iOS13)
        def func_1(x):
            return x

        @mb.function(input_specs=[mb.TensorSpec(shape=(2, 4))], opset_version=ct.target.iOS17)
        def func_2(x):
            return x

        err_msg = "all functions must have the same opset_version."

        prog = mil.Program()
        prog.add_function("func_1", func_1)
        with pytest.raises(ValueError, match=err_msg):
            prog.add_function("func_2", func_2)

        prog = mil.Program()
        prog.add_function("func_2", func_2)
        with pytest.raises(ValueError, match=err_msg):
            prog.add_function("func_1", func_1)


class TestMILBasic:
    """
    Test the basic error handling / validation in pymil.
    """
    @staticmethod
    def test_type_domain_validation():
        '''
        The builder should error out early when detecting the input type violation against the defined type_domain
        '''
        expected_err_str = (
            "In op, of type rsqrt, named rsqrt_0, the named input `epsilon` must have the same data type as the named input `x`. However, epsilon has dtype int32 whereas x has dtype fp32"
        )
        with pytest.raises(ValueError, match=expected_err_str):
            @mb.program(input_specs=[mb.TensorSpec(shape=(2,), dtype=types.fp32)])
            def prog(x):
                res = mb.rsqrt(x=x, epsilon=1)
                return res

    @staticmethod
    def test_get_dialect_namespaces():
        """
        Test we can get a dict of dialect namespaces in the program.
        """
        # The pymil program is mixed of torch / complex dialect opset
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 2, 3, 4), dtype=types.fp32)])
        def prog(x):
            real_data = mb.torch_upsample_nearest_neighbor(
                x=x, output_height=10, output_width=5, name="op_1"
            )
            imag_data = mb.add(x=real_data, y=8.9, name="op_2")
            return mb.complex(real_data=real_data, imag_data=imag_data, name="op_3")

        dialect_namespaces = prog._get_dialect_namespaces()
        assert len(dialect_namespaces["torch"]) == 1
        assert dialect_namespaces["torch"][0].name == "op_1"
        assert len(dialect_namespaces["complex"]) == 1
        assert dialect_namespaces["complex"][0].name == "op_3"

        # The pymil program with only core ops returns an empty dict
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 2, 3, 4), dtype=types.fp32)])
        def prog(x):
            return mb.add(x=x, y=8.9)

        assert len(prog._get_dialect_namespaces()) == 0

    @staticmethod
    def test_invalid_dialect_namespaces_error_out():
        """
        The converter should early error out if dialect opset is detected in the pymil program.
        """
        # The pymil program of torch dialect opset cannot be lowered to backend
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 2, 3, 4), dtype=types.fp32)])
        def prog(x):
            return mb.torch_upsample_nearest_neighbor(
                x=x, output_height=10, output_width=5, name="op_1"
            )

        expected_err_str = 'Core ML only support core opset. Got unsupported op "op_1" with type "torch_upsample_nearest_neighbor" of dialect namespace "torch".'
        with pytest.raises(ValueError, match=expected_err_str):
            ct.convert(prog, convert_to="mlprogram", pass_pipeline=ct.PassPipeline.EMPTY)

    @staticmethod
    def test_rank6_tensor_early_error_out():
        '''
        The builder should error out early when detecting a rank 6 (or higher) tensor which cannot be eliminated by graph passes
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            res = mb.reshape(x=x, shape=(1, 1, 1, 1, 1, 1), name="reshape_0")
            return res

        expected_err_str = (
            "Core ML only supports tensors with rank <= 5. Layer \"reshape_0\", with type \"reshape\", outputs a rank 6 tensor"
        )
        with pytest.raises(ValueError, match=expected_err_str):
            ct.convert(
                prog,
                source="milinternal",
                convert_to="neuralnetwork",
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )

    @staticmethod
    def test_rank5_list_early_error_out():
        '''
        The builder should error out early when detecting a list of rank 5 (or higher) tensors is created
        '''
        expected_err_str = (
            "Core ML only supports list of elements with rank <= 4. Layer \"list_0\", with type \"make_list\", outputs a list of rank 5 tensors."
        )
        with pytest.raises(ValueError, match=expected_err_str):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
            def prog(x):
                ls = mb.make_list(
                    init_length=1,
                    dtype="fp32",
                    elem_shape=(1, 1, 1, 1, 1),
                    dynamic_length=True,
                    name="list_0",
                )
                return ls

    @staticmethod
    def test_invalid_const_input_early_error_out():
        """
        The following program:

        constexpr -> transpose -> linear

        will not error out during the front end conversion, even though the weight of
        linear op needs to be const / constexpr directly.

        It is going to error out after all the optimization graph passes are finished,
        and transpose remains.

        However, if transpose can be removed, the conversion goes through.
        """
        # Test a simple constexpr op fed into linear
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            constexpr = CONSTEXPR_FUNCS["constexpr_affine_dequantize"]((4, 3))
            return mb.linear(x=x, weight=constexpr)

        for compute_precision in [ct.precision.FLOAT32, ct.precision.FLOAT16]:
            mlmodel = ct.convert(
                prog,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS16,
                compute_units=ct.ComputeUnit.CPU_ONLY,
                compute_precision=compute_precision,
            )

        # Additional pattern (transpose) after constexpr will cause an early error out
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            constexpr = CONSTEXPR_FUNCS["constexpr_affine_dequantize"]((3, 4))
            constexpr = mb.transpose(x=constexpr, perm=[1, 0])
            return mb.linear(x=x, weight=constexpr)

        for compute_precision in [ct.precision.FLOAT32, ct.precision.FLOAT16]:
            with pytest.raises(ValueError, match="must be const or constexpr ops"):
                mlmodel = ct.convert(
                    prog,
                    convert_to="mlprogram",
                    minimum_deployment_target=ct.target.iOS16,
                    pass_pipeline=ct.PassPipeline.EMPTY,
                    compute_units=ct.ComputeUnit.CPU_ONLY,
                    compute_precision=compute_precision,
                )

        # If the transpose is removed by graph pass merge_affine_dequantize_with_consecutive_ops,
        # the conversion goes through
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            constexpr = CONSTEXPR_FUNCS["constexpr_affine_dequantize"]((4, 3))
            constexpr = mb.transpose(x=constexpr, perm=[0, 1])
            return mb.linear(x=x, weight=constexpr)

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=compute_precision,
        )

class TestScope:
    @staticmethod
    def test_basic_single():
        # single scope with scope_name and scope_type
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data="module_1"),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="Module1"),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["module_1"]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["Module1"]

        # single scope with scope_name and scope_type with list type
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1"]),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["module_1"]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["Module1"]

        # single scope with scope_type and no scope_name
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_1"]),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op.attributes["scopes"]

        # nested scope in a single mb.scope call. Both scope_name and scope_type provided
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1", "module_2"]
                ),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1", "Module2"]),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == [
            "module_1",
            "module_2",
        ]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "Module1",
            "Module2",
        ]

        # nested scope in a single mb.scope call. Only scope_type provided
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_1", "module_2"]
                ),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
        ]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op.attributes["scopes"]

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["", ""]),
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_1", "module_2"]
                ),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["", ""]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
        ]

    @staticmethod
    def test_basic_nested():
        # nested scope with scope_name and scope_type
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data="module_1"),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="Module1"),
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data="module_2"),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="Module2"),
                ):
                    x = mb.add(x=x, y=5.4)
                return mb.add(x=x, y=0.0)

        add_op_1 = prog.find_ops(op_type="add")[0]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == [
            "module_1",
            "module_2",
        ]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "Module1",
            "Module2",
        ]

        add_op_2 = prog.find_ops(op_type="add")[1]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["module_1"]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["Module1"]

        # nested scope with scope_name and scope_type with list type
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1"]),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_2"]),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module2"]),
                ):
                    x = mb.add(x=x, y=5.4)
                return mb.add(x=x, y=0.0)

        add_op_1 = prog.find_ops(op_type="add")[0]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == [
            "module_1",
            "module_2",
        ]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "Module1",
            "Module2",
        ]

        add_op_2 = prog.find_ops(op_type="add")[1]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["module_1"]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["Module1"]

        # nested scope with scope_name and no scope_type
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_1"]),
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_2"]),
                ):
                    x = mb.add(x=x, y=5.4)
                return mb.add(x=x, y=0.0)

        add_op_1 = prog.find_ops(op_type="add")[0]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op_1.attributes["scopes"]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
        ]

        add_op_2 = prog.find_ops(op_type="add")[1]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op_2.attributes["scopes"]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]

        # nested scope in a nested mb.scope call. Both scope_name and scope_type provided
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1", "module_2"]
                ),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1", "Module2"]),
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data="module_3"),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="Module3"),
                ):
                    x = mb.add(x=x, y=5.4)
                return mb.add(x=x, y=0.0)

        add_op_1 = prog.find_ops(op_type="add")[0]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == [
            "module_1",
            "module_2",
            "module_3",
        ]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "Module1",
            "Module2",
            "Module3",
        ]

        add_op_2 = prog.find_ops(op_type="add")[1]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == [
            "module_1",
            "module_2",
        ]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "Module1",
            "Module2",
        ]

        # nested scope in a single mb.scope call. Only scope_type provided
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_1", "module_2"]
                ),
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_3"]),
                ):
                    x = mb.add(x=x, y=5.4)
                return mb.add(x=x, y=0.0)

        add_op_1 = prog.find_ops(op_type="add")[0]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op_1.attributes["scopes"]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
            "module_3",
        ]

        add_op_2 = prog.find_ops(op_type="add")[1]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op_2.attributes["scopes"]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
        ]

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_1", "module_2"]
                ),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["", ""]),
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["module_3"]),
                ):
                    x = mb.add(x=x, y=5.4)
                return mb.add(x=x, y=0.0)

        add_op_1 = prog.find_ops(op_type="add")[0]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
            "module_3",
        ]
        assert add_op_1.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["", ""]

        add_op_2 = prog.find_ops(op_type="add")[1]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
        ]
        assert add_op_2.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["", ""]


    @staticmethod
    def test_invalid_dtype_error_out():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with pytest.raises(
                ValueError,
                match="TORCHSCRIPT_MODULE_TYPE and TORCHSCRIPT_MODULE_NAME scope must be type of List\[str\]. Got element 9 with type \<class 'int'\>.",
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["m1", 9]),
                    ScopeInfo(
                        source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1", "Module2"]
                    ),
                ):
                    return mb.add(x=x, y=5.4)

            with pytest.raises(
                ValueError,
                match="TORCHSCRIPT_MODULE_TYPE and TORCHSCRIPT_MODULE_NAME scope must be type of List\[str\]. Got element 0 with type \<class 'int'\>.",
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["m1", "m2"]),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1", 0]),
                ):
                    return mb.add(x=x, y=5.4)
            return x

    @staticmethod
    def test_empty_scope():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope():
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert ScopeSource.TORCHSCRIPT_MODULE_TYPE not in add_op.attributes["scopes"]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op.attributes["scopes"]

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope():
                with mb.scope():
                    return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert ScopeSource.TORCHSCRIPT_MODULE_TYPE not in add_op.attributes["scopes"]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op.attributes["scopes"]

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope():
                with mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="m1")):
                    with mb.scope():
                        return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["m1"]
        assert ScopeSource.TORCHSCRIPT_MODULE_NAME not in add_op.attributes["scopes"]


    @staticmethod
    def test_empty_scope_type_error_out():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with pytest.raises(
                ValueError, match="TORCHSCRIPT_MODULE_TYPE scope info cannot contains empty string."
            ):
                with mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="")):
                    with mb.scope():
                        return mb.add(x=x, y=5.4)

            with pytest.raises(
                ValueError, match="TORCHSCRIPT_MODULE_TYPE scope info cannot contains empty string."
            ):
                with mb.scope(
                    ScopeInfo(
                        source=ScopeSource.TORCHSCRIPT_MODULE_TYPE,
                        data=["a", ""],
                    )
                ):
                    with mb.scope():
                        return mb.add(x=x, y=5.4)
            return x

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_TYPE,
                    data=["module_1"],
                )
            ):
                with pytest.raises(
                    ValueError,
                    match="TORCHSCRIPT_MODULE_TYPE scope info cannot contains empty string.",
                ):
                    with mb.scope(
                        ScopeInfo(
                            source=ScopeSource.TORCHSCRIPT_MODULE_TYPE,
                            data=[""],
                        )
                    ):
                        return mb.add(x=x, y=5.4)
                with pytest.raises(
                    ValueError,
                    match="TORCHSCRIPT_MODULE_TYPE scope info cannot contains empty string.",
                ):
                    with mb.scope(
                        ScopeInfo(
                            source=ScopeSource.TORCHSCRIPT_MODULE_TYPE,
                            data=["a", "", ""],
                        )
                    ):
                        return mb.add(x=x, y=5.4)
            return x

    @staticmethod
    def test_white_space_handling():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=[" module_1  "]),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=[" Module1"]),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == [
            "module_1",
        ]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "Module1",
        ]

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=[" Module1   ", " "]),
                ScopeInfo(
                    source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=[" module_1 ", " module_2 "]
                ),
            ):
                return mb.add(x=x, y=5.4)

        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_1",
            "module_2",
        ]
        assert add_op.attributes["scopes"][ScopeSource.TORCHSCRIPT_MODULE_NAME] == ["Module1", ""]

    @staticmethod
    def test_duplicated_scope_source_error_out():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with pytest.raises(
                ValueError, match="Scope source ScopeSource.TORCHSCRIPT_MODULE_TYPE duplicated."
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="a1"),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="a2"),
                ):
                    return mb.add(x=x, y=5.4)
            return x

    @staticmethod
    def test_check_prog_has_scope_error_out():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1"]),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
            ):
                x = mb.add(x=x, y=5.4)
            x = mb.relu(x=x, name="invalid_op")
            return x

        with pytest.raises(ValueError, match="op invalid_op is missing scopes attribute."):
            prog._check_has_scope_info()

    @staticmethod
    def test_invalid_scope_source_type():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with pytest.raises(TypeError, match="'source' must be \<enum 'ScopeSource'\>"):
                with mb.scope(
                    ScopeInfo(source="invalid_source", data="a1"),
                ):
                    return mb.add(x=x, y=5.4)
            return x

    @staticmethod
    def test_invalid_scope_info_type():
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            with pytest.raises(
                ValueError,
                match="mb.scope only accepts inputs of type ScopeInfo. Got \<class 'str'\>.",
            ):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1"]),
                    "invalid",
                ):
                    return mb.add(x=x, y=5.4)
            return x
