# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import argparse
import contextlib
from distutils import spawn
from shutil import which
import json
import os
import shlex
import subprocess
import tempfile
import pathlib
from typing import List, Any, Union, Callable, Optional, Dict, Iterable, Tuple, Set
import logging
import warnings

import torch
import numpy as np

from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import xla_data_pb2, hlo_pb2
from torch_neuronx.xla_impl.hlo_conversion import xla_trace
from torch_neuronx.xla_impl.options import OptionsDefault, Options
from torch_neuronx.xla_impl.partitioner import PartitionerConfig
import torch_neuronx
from torch_neuronx.xla_impl import structure
from torch_neuronx.xla_impl import placement
from torch_neuronx.utils import get_platform_target


logger = logging.getLogger("Neuron")

XLA_DTYPE_TO_METANEFF_DTYPE = {
    xla_data_pb2.F32: metaneff_pb2.MetaTensor.DataType.FLOAT,
    xla_data_pb2.F64: metaneff_pb2.MetaTensor.DataType.DOUBLE,
    xla_data_pb2.BF16: metaneff_pb2.MetaTensor.DataType.BFLOAT16,
    xla_data_pb2.F16: metaneff_pb2.MetaTensor.DataType.FLOAT16,
    xla_data_pb2.U8: metaneff_pb2.MetaTensor.DataType.UINT8,
    xla_data_pb2.S8: metaneff_pb2.MetaTensor.DataType.INT8,
    xla_data_pb2.U16: metaneff_pb2.MetaTensor.DataType.UINT16,
    xla_data_pb2.S16: metaneff_pb2.MetaTensor.DataType.INT16,
    xla_data_pb2.U32: metaneff_pb2.MetaTensor.DataType.INT32,
    xla_data_pb2.S32: metaneff_pb2.MetaTensor.DataType.INT32,
    xla_data_pb2.U64: metaneff_pb2.MetaTensor.DataType.INT64,
    xla_data_pb2.S64: metaneff_pb2.MetaTensor.DataType.INT64,
    xla_data_pb2.C64: None,
    xla_data_pb2.C128: None,
    xla_data_pb2.PRED: metaneff_pb2.MetaTensor.DataType.BOOL,
    xla_data_pb2.F8E4M3FN: metaneff_pb2.MetaTensor.DataType.F8E4M3FN,
    xla_data_pb2.F8E5M2: metaneff_pb2.MetaTensor.DataType.F8E5M2,
}


@contextlib.contextmanager
def nullcontext():
    """
    A context which does nothing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager.
    """
    yield


class NeuronModule(torch.nn.Module):
    """
    A torch module that encapsulates a compiled Neuron graph.

    Args:
        neff: A compiled model artifact.
        metaneff: Serialized metadata regarding the NEFF interface.
        flattener: Extracts tensors from an expected hierarchical input structure.
        packer: Packs tensors into an expected hierarhcical output structure.
    """

    def __init__(
        self,
        neff: str,
        metaneff: str,
        flattener: structure.Flattener,
        packer: structure.Packer,
        initial_states: List = None,
        weights: List = None,
    ) -> None:
        super().__init__()
        self.model = torch.classes.neuron.Model(neff, metaneff)
        self.flattener = flattener
        self.packer = packer

        states = list()
        for tensor in initial_states:
            if tensor.device.type == 'xla':
                tensor = tensor.cpu()
            states.append(torch.nn.Parameter(tensor, requires_grad=False))
        self.states = torch.nn.ParameterList(states)

        self.weights = (
            torch.nn.ParameterDict(
                [
                    (module_path, torch.nn.Parameter(weight, requires_grad=False))
                    for module_path, weight in weights
                ]
            )
            if weights
            else torch.nn.ParameterDict([])
        )

    def forward(self, *tensors):
        inputs = self.flattener(tensors)
        inputs.extend(self.states)
        inputs.extend(self.weights.values())  # parameter dict is ordereddict
        result = torch.ops.neuron.forward_v2(inputs, self.model)
        return self.packer(result)


def hlo_entry_computation(hlo: hlo_pb2.HloProto) -> hlo_pb2.HloComputationProto:
    """
    Retrieve the main computation (external interface) from the HloProto

    Args:
        hlo: The HloProto to extract root computation from.

    Returns:
        The computation interface.
    """
    return get_hlo_computation_by_id(hlo, hlo.entry_computation_id)


def get_hlo_computation_by_id(hlo: hlo_pb2.HloModuleProto, id: int):
    for computation in hlo.computations:
        if computation.id == id:
            return computation


def get_hlo_root_instruction(hlo_computation: hlo_pb2.HloComputationProto) -> hlo_pb2.HloInstructionProto:
    for instr in hlo_computation.instructions:
        if instr.id == hlo_computation.root_id:
            return instr


def hlo_metaneff(
    hlo: hlo_pb2.HloProto,
    input_parameter_names: List[str],
    input_output_aliases: Dict,
    sorted_param_idxs: List[int],
    weights_param_num_to_module_path: Dict[int, str],
    buffers_param_num_to_module_path: Dict[int, str],
    output_aliased_tensor: bool = True,

) -> metaneff_pb2.MetaNeff:
    """
    Build a tensor metadata message from the root computation of an HloProto.

    This is used by the runtime to validate input tensor shapes, names, and
    types. Secondly the resulting message can be augmented to contain metadata
    about the graph being executed and the pipeline configuration.

    Args:
        hlo: The HloProto to extract the interface metadata from.
        input_parameter_names: inputs that should be added to the metaneff
        input_output_aliases: alias mapping between the inputs and outputs

    Returns:
        The MetaNeff message containing interface metadata.
    """
    # TODO: See if metaneff can be eliminated in favor of
    #       `nrt_get_model_tensor_info`. Right now this does not seem possible
    #       since the tensor info API requires that a model was successfully
    #       loaded to be able to call it. The metaneff is used during tracing
    #       prior to an `nrt_load` to emit placeholder tensors.

    computations = hlo_entry_computation(hlo)

    metaneff = metaneff_pb2.MetaNeff()
    program_shape = computations.program_shape
    metaneff.model_config.return_aliases = output_aliased_tensor

    # save hlo in metaneff
    metaneff.serialized_graph_def = hlo.SerializeToString()

    # Create mapping from parameter name to parameter metadata
    parameter_name_to_metadata = {}
    for name, metadata in zip(program_shape.parameter_names, program_shape.parameters):
        parameter_name_to_metadata[name] = metadata

    # Iterate through input parameters to ensure correct input order at runtime
    for index, name in enumerate(input_parameter_names):
        metadata = parameter_name_to_metadata[name]

        shape = list(metadata.dimensions)
        dtype = metadata.element_type

        tensor = metaneff.input_tensors.add()
        tensor.name = f"input{index}".encode(
            "utf8"
        )  # Needs to be `input#` to avoid a `ddrs_create_lookup_key` error
        tensor.shape[:] = shape
        tensor.data_type = XLA_DTYPE_TO_METANEFF_DTYPE[dtype]
        if index in buffers_param_num_to_module_path:
            tensor.checkpoint_key = buffers_param_num_to_module_path[index].encode("utf8")
            tensor.type = metaneff_pb2.MetaTensor.Type.INPUT_STATE
        else:
            tensor.type = metaneff_pb2.MetaTensor.Type.USER_INPUT

    # input weights parameter index and metaneff index need to be mapped
    start_weight_idx = len(input_parameter_names)
    num_weights = 0
    weight_indices = []
    for index in sorted_param_idxs:
        weight_indices.append((index, start_weight_idx + num_weights))
        num_weights += 1

    for parameter_idx, metaneff_idx in weight_indices:
        metadata = parameter_name_to_metadata[f"p{parameter_idx}"]

        shape = list(metadata.dimensions)
        dtype = metadata.element_type

        tensor = metaneff.input_tensors.add()
        tensor.name = f"input{metaneff_idx}".encode("utf8")
        tensor.shape[:] = shape
        tensor.data_type = XLA_DTYPE_TO_METANEFF_DTYPE[dtype]
        tensor.checkpoint_key = weights_param_num_to_module_path[parameter_idx].encode("utf8")
        tensor.type = metaneff_pb2.MetaTensor.Type.INPUT_WEIGHT

    for index, metadata in enumerate(program_shape.result.tuple_shapes):
        shape = list(metadata.dimensions)
        dtype = metadata.element_type

        tensor = metaneff.output_tensors.add()
        tensor.name = f"output{index}".encode("utf8")
        tensor.shape[:] = shape
        tensor.data_type = XLA_DTYPE_TO_METANEFF_DTYPE[dtype]

    for output_idx, input_idx in input_output_aliases.items():
        metaneff.output_aliases_to[output_idx] = input_idx

    return metaneff


def setup_compiler_dirs(
    hlo: hlo_pb2.HloProto,
    compiler_workdir: Union[str, pathlib.Path],
    constant_parameter_tensors: Optional[List[torch.Tensor]],
    inline_weights_to_neff: bool,
) -> str:
    # Create compiler directory if it does not exist
    if not os.path.exists(compiler_workdir):
        os.makedirs(compiler_workdir, exist_ok=True)

    # Set up compiler_workdir folder structure
    model_dir = os.path.join(compiler_workdir, "model")
    os.makedirs(model_dir, exist_ok=True)
    hlo_filename = os.path.join(model_dir, "graph.hlo")

    hlo_path = hlo_filename

    # Write weights to disk only if inlining weights
    if inline_weights_to_neff and constant_parameter_tensors is not None:
        weight_paths = write_params(
            model_dir,
            constant_parameter_tensors,
        )

        table = {
            "model_files": "graph.hlo",
            "version": "1.0",
            "weights": weight_paths,
        }
        with open(f"{model_dir}/metadata.json", "w") as f:
            json.dump(table, f, indent=4)

        hlo_path = model_dir

    # Write optimized HLO to disk
    with open(hlo_filename, "wb") as handle:
        handle.write(hlo.SerializeToString())

    return hlo_path


def hlo_compile(
    filename: Union[str, pathlib.Path],
    compiler_workdir: Union[str, pathlib.Path],
    compiler_args: Optional[Union[List[str], str]] = None,
) -> str:
    """
    Compiles a serialized HloProto into a NEFF using `neuronx-cc`

    Compiler CLI Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html  # noqa: E501

    Args:
        filename: The filename for the serialized HloProto or the path to the
            `model` folder
        compiler_workdir: The directory to save any compiler outputs to.
        compiler_args: Additional compiler arguments.

    Returns:
        The filename of the compiled NEFF.
    """
    # Ensure neuronx-cc is installed
    try:
        import neuronxcc
    except ImportError as e:
        raise RuntimeError(
            "neuronx-cc is not installed.\n"
            "neuronx-cc can be installed using:\n"
            "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc"
        ) from e

    # NOTE: Checking the installation above isolates of all neuronx-cc usage to
    #       this function. This is convenient for mocking/patching for tests.

    neff_filename = os.path.join(compiler_workdir, "graph.neff")
    command = get_compile_command(filename, neff_filename, compiler_args)

    # Write the command that produces the NEFF
    save_compile_command(command, compiler_workdir)

    status = subprocess.run(command).returncode
    if status != 0:
        if status == -9:
            logger.warning(
                "The neuronx-cc (neuron compiler) process was killed (SIG_KILL).  "
                "This typically happens when there is insufficient memory to compile and the linux "
                "Out Of Memory (OOM) killer terminates the compiler. "
                "Consider trying compilation on an instance with more memory"
            )
        elif status == -6:
            logger.warning(
                f'neuronx-cc aborted (status={status}) in hlo_compile: '
                'likely compiler bug. See https://github.com/aws/aws-neuron-sdk/issues'
            )
        elif status == -11:
            logger.warning(
                "The neuronx-cc (neuron compiler) crashed (SEGFAULT). "
                "This is likely due to a bug in the compiler.  "
                "Please lodge an issue at 'https://github.com/aws/aws-neuron-sdk/issues'"
            )

        raise RuntimeError(f"neuronx-cc failed with {status}")

    return neff_filename


def get_compile_command(
    filename: Union[str, pathlib.Path],
    neff_filename: Union[str, pathlib.Path],
    compiler_args: Optional[Union[List[str], str]] = None,
):
    """
    Creates a compile command.

    Args:
        filename: The filename for the serialized HloProto or the path to the
            `model` folder
        neff_filename: The filename for the output NEFF.
        compiler_args: Additional compiler arguments.

    Returns:
        The compile command as a list.
    """
    if compiler_args is None:
        compiler_args = []
    elif isinstance(compiler_args, str):
        compiler_args = shlex.split(compiler_args)

    # Merge compiler_args with default compiler arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target',
        default=get_platform_target(compiler_args=compiler_args),
        help='Instance family')
    known_flags, compiler_args = parser.parse_known_args(compiler_args)

    neuron_cc = which("neuronx-cc")
    if neuron_cc is None:
        raise RuntimeError("neuronx-cc compiler binary does not exist")
    command = [
        neuron_cc,
        "compile",
        filename,
        "--framework",
        "XLA",
        "--target",
        known_flags.target,
        "--output",
        neff_filename,
    ]
    if "--verbose" not in "".join(compiler_args):
        compiler_args.append("--verbose=35")

    command.extend(compiler_args)
    return command


def save_compile_command(command: List[str], compiler_workdir: Union[str, pathlib.Path]):
    """
    Saves the compiler command as command.txt in the compiler workdir.
    """
    command_filename = os.path.join(compiler_workdir, "command.txt")
    with open(command_filename, "w") as f:
        # Strip file paths from the command.
        stripped = []
        for item in command:
            item = str(item)
            if "=" in item:
                item_parts = item.split("=", 1)
                stripped.append(item_parts[0] + "=" + os.path.basename(item_parts[1]))
            else:
                stripped.append(os.path.basename(item))
        f.write(shlex.join(stripped))


class HloArtifacts:
    def __init__(
        self,
        hlo_module,
        flattener,
        packer,
        metaneff,
        weights,
        constant_parameter_tensors,
        weight_name_to_idx,
    ):
        self.hlo_module = hlo_module
        self.flattener = flattener
        self.packer = packer
        self.metaneff = metaneff
        self.weights = weights
        self.constant_parameter_tensors = constant_parameter_tensors
        self.weight_name_to_idx = weight_name_to_idx


def generate_hlo(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    input_output_aliases=None,
    inline_weights_to_neff: bool = True,
    return_weights: bool = True,
    output_aliased_tensor: bool = True,
    cpu_backend: bool = False,
    preserve_parameters: bool = True,
    enable_aliasing: bool = False,
    treat_inputs_as_kwargs: bool = False,
):
    with torch_neuronx.contexts.chain(
        torch_neuronx.contexts.mock_neuron_cores(),
        torch_neuronx.contexts.set_pjrt_device(cpu_backend),
    ):
        (
            hlo,
            input_parameter_names,
            constant_parameter_tensors,
            weights_param_num_to_module_path,
            buffers_param_num_to_module_path,
            flattener,
            packer,
            updated_input_output_aliases,
        ) = xla_trace(
            func,
            example_inputs,
            input_output_aliases=input_output_aliases,
            output_aliased_tensor=output_aliased_tensor,
            preserve_parameters=preserve_parameters,
            enable_aliasing=enable_aliasing,
            treat_inputs_as_kwargs=treat_inputs_as_kwargs,
        )

    # get weight dict if in weight separation mode
    weight_idxs, weights, weight_name_to_idx = get_weights_and_index(
        inline_weights_to_neff,
        constant_parameter_tensors,
        weights_param_num_to_module_path,
    )

    metaneff = hlo_metaneff(
        hlo, input_parameter_names, updated_input_output_aliases, weight_idxs,
        weights_param_num_to_module_path,
        buffers_param_num_to_module_path,
        output_aliased_tensor=output_aliased_tensor,
    )

    return HloArtifacts(
        hlo,
        flattener,
        packer,
        metaneff,
        weights if return_weights else None,
        constant_parameter_tensors if inline_weights_to_neff else None,
        weight_name_to_idx,
    )


class NeffArtifacts:
    def __init__(
        self,
        neff_filename,
    ):
        self.neff_filename = neff_filename


def generate_neff(
    hlo_artifacts: HloArtifacts,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    inline_weights_to_neff: bool = True,
):
    # setup compiler environment
    compiler_target = setup_compiler_dirs(
        hlo_artifacts.hlo_module,
        compiler_workdir,
        hlo_artifacts.constant_parameter_tensors,
        inline_weights_to_neff,
    )

    # Compile HLO to NEFF
    neff_filename = hlo_compile(
        compiler_target,
        compiler_workdir,
        compiler_args,
    )

    return NeffArtifacts(neff_filename)


def trace(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    *_,
    input_output_aliases=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    partitioner_config: Optional[PartitionerConfig] = None,
    inline_weights_to_neff=True,
    cpu_backend=True,
    enable_aliasing=False,
    **kwargs,
) -> torch.jit.ScriptModule:
    """
    Trace a torch module/function to produce a compiled Neuron ScriptModule.

    This uses torch-xla to extract the computation graph. The input `func` and
    `example_inputs` must be able to be moved to the XLA device.

    The resulting module can used with `torch.jit.save` and `torch.jit.load`

    NOTE: Please use keyword arguments for all arguments after `example_inputs`.
    Ex: `torch_neuronx.trace(func,example_inputs,compiler_workdir="some_dir") #valid`
        `torch_neuronx.trace(func,example_inputs,"some_dir") #invalid`

    Args:
        func: A module or function which defines a torch model or computation.
        example_inputs: An example set of inputs which will be passed to the
            `func` during tracing.
        input_output_aliases: alias mapping between the inputs and outputs
        compiler_workdir: The directory to save any compiler outputs to.
        compiler_args: Additional compiler arguments.
        partitioner_config: A PartitionerConfig object, which can be optionally
        supplied if there are unsupported ops in the model that need to be
        partitioned out to CPU.
        inline_weights_to_neff: A boolean indicating whether the weights should be
        inlined to the neff. The default is `True`.

    Returns:
        A Module where the HLO computation is a fused neuron::foward operation.
    """

    if input_output_aliases is None:
        input_output_aliases = dict()

    # Create a temporary directory for artifacts if none is specified
    context = nullcontext()
    if compiler_workdir is None:
        context = tempfile.TemporaryDirectory()
        compiler_workdir = context.name

    states = None
    if _:
        ValueError(
            f"Received {len(_)} positional arguments but expected 2. "
            "Use of positional arguments after `func` and `example_inputs` is "
            "deprecated. Please specify keyword arguments instead.",
        )

    if kwargs:
        warnings.warn(f'trace() ignoring unexpected kwargs: {list(kwargs.keys())}', category=UserWarning)

    if partitioner_config:
        if partitioner_config.trace_kwargs is None and (
            compiler_args or compiler_workdir
        ):
            partitioner_config.trace_kwargs = {
                "compiler_args": compiler_args,
                "compiler_workdir": compiler_workdir,
            }

        return torch_neuronx.partition(
            func, example_inputs, **(partitioner_config.__dict__)
        )

    with context:
        neff_filename, metaneff, flattener, packer, weights = _trace(
            func,
            example_inputs,
            states,
            input_output_aliases,
            compiler_workdir,
            compiler_args,
            inline_weights_to_neff=inline_weights_to_neff,
            cpu_backend=cpu_backend,
            enable_aliasing=enable_aliasing,
        )
        return create_neuron_model(
            neff_filename,
            metaneff,
            flattener,
            packer,
            example_inputs,
            input_output_aliases,
            weights,
        )


def create_neuron_model(
    neff_filename: Union[str, pathlib.Path],
    metaneff: str,
    flattener: structure.Flattener,
    packer: structure.Packer,
    example_inputs: Any,
    input_output_aliases: Dict,
    weights: List[Tuple[str, torch.Tensor]],
):

    # Handle case when NxD uses input_output_aliases=None when using automatic aliasing
    if input_output_aliases is None:
        input_output_aliases = dict()

    if weights is None:
        raise ValueError(
            "Weights is None, but expected a list of length 0 or greater."
            "create_neuron_model() is only supported with return_weights=True for generate_hlo()."
        )

    with open(neff_filename, "rb") as handle:
        neff = handle.read()

    initial_states = tuple(input_output_aliases.keys())
    result = NeuronModule(neff, metaneff, flattener, packer, initial_states, weights)
    with torch_neuronx.contexts.disable_nrt_load():
        # NOTE: Turn on strict=False by default since the only reason we
        #       trace here is to ensure that data structures are correctly
        #       represented in torchscript (input/output)
        return torch.jit.trace(result, example_inputs, strict=False)


def _trace(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    states=None,
    input_output_aliases=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    inline_weights_to_neff=True,
    cpu_backend=True,
    enable_aliasing=False,
) -> Union[str, str, structure.Flattener, structure.Packer]:

    if input_output_aliases is None:
        input_output_aliases = dict()

    # Convert the function to a HloProto message
    hlo_artifacts = generate_hlo(
        func,
        example_inputs,
        input_output_aliases=input_output_aliases,
        inline_weights_to_neff=inline_weights_to_neff,
        cpu_backend=cpu_backend,
        enable_aliasing=enable_aliasing,
    )

    # Call neuronx-cc to generate neff
    neff_artifacts = generate_neff(
        hlo_artifacts,
        compiler_workdir=compiler_workdir,
        compiler_args=compiler_args,
        inline_weights_to_neff=inline_weights_to_neff,
    )

    return (
        neff_artifacts.neff_filename,
        hlo_artifacts.metaneff.SerializeToString(),
        hlo_artifacts.flattener,
        hlo_artifacts.packer,
        hlo_artifacts.weights,
    )


def get_weights_and_index(
    inline_weights_to_neff,
    constant_parameter_tensors,
    weights_param_num_to_module_path,
):
    """
    This function will identify the weights from the
    collection of HLO Parameters. If `inline_weights_to_neff`
    is enabled, then all hlo_parameter tensors are returned and
    be classified as constants; however, if not enabled, then only
    the tensors classified as weights using the `weights_param_num_to_module_path` dictionary,
    are returned, and two index collections will be returned:
    `weight_idxs` and `constant_idxs`, which are used for inlining
    constants, and encoding weights as inputs into the neff,
    via the metaneff.

    Returns:
    1. weight_idxs: List of hlo parameter indices for weight tensors. Used
        for the purposes of mapping inputs in metaneff
    2. weights: List of weight tensors to be used for a weight separated model.
        This list is empty when `inline_weights_to_neff=True`.
    3. name_to_idx: Mapping from weight name to input idx. This is used for
        weight layout optimization across different HLO
    """
    if inline_weights_to_neff:
        return [], [], []

    sorted_param_idxs = sorted([idx for idx in constant_parameter_tensors.keys()])

    weight_idxs = []
    weights = []
    weight_name_to_idx = {}

    for idx in sorted_param_idxs:
        param_tensor = constant_parameter_tensors[idx]

        name = weights_param_num_to_module_path[idx].replace(".", "->")
        weights.append((name, param_tensor))
        weight_idxs.append(idx)
        weight_name_to_idx[name] = idx

    return weight_idxs, weights, weight_name_to_idx


def write_params(
    directory: Union[str, pathlib.Path],
    params: Dict[Union[str, int], torch.Tensor],
) -> None:
    # Create the directory to store hlo parameters
    os.makedirs(f"{directory}/weights/", exist_ok=True)
    inlined_parameter_paths = {}

    # Write tensor data to disk
    for name, param_tensor in params.items():
        # Represent bfloat16 as 2-byte void numpy type to allow serialization
        if param_tensor.dtype == torch.bfloat16:
            param_tensor = param_tensor.view(torch.int16)
            param_tensor = param_tensor.numpy()
            param_tensor = param_tensor.view("|V2")
        elif param_tensor.dtype == torch.float8_e4m3fn or param_tensor.dtype == torch.float8_e5m2:
            param_tensor = param_tensor.view(torch.int8)
            param_tensor = param_tensor.numpy()
            param_tensor = param_tensor.view("|V1")
        else:
            param_tensor = param_tensor.numpy()
        np.save(f"{directory}/weights/{name}.npy", param_tensor)

        inlined_parameter_paths[name] = f"weights/{name}.npy"

    return inlined_parameter_paths


def move_trace_to_device(trace, device_id):
    runtime = torch.classes.neuron.Runtime()
    runtime.initialize()
    runtime.set_default_neuron_cores(device_id, -1)
    # Currently we have to explicitly move the state params to device because
    # the map_location feature for privateuseone device is only added recently
    # in torch. TODO: replace this with map_location feature in jit.load once
    # this commit is pulled: https://github.com/pytorch/pytorch/commit/da322ea874a5abeb2f10f9e4c4bb8414892cb0d2
    # Ideally trace.to(device) would work(and it works when we trace and move) but it
    # fails when we load a trace module and then try to move it to device. This is because,
    # when we save the trace, the torch.save doesn't preserve the Param attribute and when
    # we try to move the loaded trace, since we don't have the param attribute to tensor,
    # it crashes in .to()
    for name, param in trace.states._parameters.items():
        trace.states._parameters[name] = param.to(f"privateuseone:{device_id}")

    for name, param in trace.weights._parameters.items():
        trace.weights._parameters[name] = param.to(f"privateuseone:{device_id}")


def get_torch_dtype(type: metaneff_pb2.MetaTensor.DataType) -> torch.dtype:
    dtypes = {
        metaneff_pb2.MetaTensor.DataType.FLOAT: torch.float32,
        metaneff_pb2.MetaTensor.DataType.INT32: torch.int32,
        metaneff_pb2.MetaTensor.DataType.BYTE: torch.uint8,
        metaneff_pb2.MetaTensor.DataType.BOOL: torch.bool,
        metaneff_pb2.MetaTensor.DataType.UINT8: torch.uint8,
        metaneff_pb2.MetaTensor.DataType.INT8: torch.int8,
        metaneff_pb2.MetaTensor.DataType.INT16: torch.int16,
        metaneff_pb2.MetaTensor.DataType.INT64: torch.int64,
        metaneff_pb2.MetaTensor.DataType.FLOAT16: torch.float16,
        metaneff_pb2.MetaTensor.DataType.DOUBLE: torch.float64,
        metaneff_pb2.MetaTensor.DataType.BFLOAT16: torch.bfloat16,
        metaneff_pb2.MetaTensor.DataType.F8E4M3FN: torch.float8_e4m3fn,
    }
    # metaneff_pb2.MetaTensor.DataType.UINT16 is not supported as of Pytorch2.1
    if type not in dtypes:
        raise KeyError(f"type: {type} not present in the get_torch_dtype map. Supported types are: {dtypes.keys()}")
    return dtypes[type]
