import numpy as np
import onnx
from onnx import helper, numpy_helper


# ============================================================
# INPUT
# ============================================================

# One integer number
input_tensor = helper.make_tensor_value_info(
    "number",
    onnx.TensorProto.INT64,
    [1]
)


# ============================================================
# OUTPUT
# ============================================================

# 1 = PRIME
# 0 = NOT PRIME
output_tensor = helper.make_tensor_value_info(
    "is_prime",
    onnx.TensorProto.INT64,
    [1]
)


# ============================================================
# CONSTANTS
# ============================================================

# Divisors we will check
divisors = numpy_helper.from_array(
    np.array([2, 3, 5, 7], dtype=np.int64),
    name="divisors"
)

# Zero
zero = numpy_helper.from_array(
    np.array([0], dtype=np.int64),
    name="zero"
)

# Number 2
two = numpy_helper.from_array(
    np.array([2], dtype=np.int64),
    name="two"
)


# ============================================================
# NODES
# ============================================================

nodes = [

    # number % [2,3,5,7]
    #
    # Example:
    # 21 % [2,3,5,7]
    #
    # => [1,0,1,0]
    helper.make_node(
        "Mod",
        inputs=["number", "divisors"],
        outputs=["remainders"],
        fmod=0
    ),


    # Check:
    #
    # remainder == 0
    #
    # Example:
    # [1,0,1,0]
    #
    # =>
    # [false,true,false,true]
    helper.make_node(
        "Equal",
        inputs=["remainders", "zero"],
        outputs=["divisible"]
    ),


    # Convert:
    #
    # false,true,false,true
    #
    # to:
    #
    # 0,1,0,1
    helper.make_node(
        "Cast",
        inputs=["divisible"],
        outputs=["divisible_int"],
        to=onnx.TensorProto.INT64
    ),


    # If ANY divisor matched, max = 1
    #
    # [0,1,0,1]
    #       ↓
    #       1
    helper.make_node(
        "ReduceMax",
        inputs=["divisible_int"],
        outputs=["has_divisor"],
        keepdims=1
    ),


    # Check number < 2
    helper.make_node(
        "Less",
        inputs=["number", "two"],
        outputs=["less_than_two"]
    ),


    # Convert boolean to integer
    helper.make_node(
        "Cast",
        inputs=["less_than_two"],
        outputs=["less_than_two_int"],
        to=onnx.TensorProto.INT64
    ),


    # If number < 2 OR has divisor
    #
    # We use Add and Clip-like logic through comparison:
    #
    # 0 + 0 = 0 -> PRIME
    # anything > 0 -> NOT PRIME
    helper.make_node(
        "Add",
        inputs=["has_divisor", "less_than_two_int"],
        outputs=["invalid"]
    ),


    # invalid == 0
    #
    # invalid = 0 -> PRIME (1)
    # invalid != 0 -> NOT PRIME (0)
    helper.make_node(
        "Equal",
        inputs=["invalid", "zero"],
        outputs=["prime_bool"]
    ),


    # Convert TRUE/FALSE to 1/0
    helper.make_node(
        "Cast",
        inputs=["prime_bool"],
        outputs=["is_prime"],
        to=onnx.TensorProto.INT64
    )
]


# ============================================================
# GRAPH
# ============================================================

graph = helper.make_graph(
    nodes,
    "PrimeMathModel",
    [input_tensor],
    [output_tensor],
    initializer=[
        divisors,
        zero,
        two
    ]
)


# ============================================================
# MODEL
# ============================================================

model = helper.make_model(
    graph,
    producer_name="prime-math-python",
    opset_imports=[
        helper.make_opsetid("", 13)
    ]
)
# Make the ONNX file compatible with older ONNX Runtime versions
model.ir_version = 13


# ============================================================
# VALIDATE
# ============================================================

onnx.checker.check_model(model)


# ============================================================
# SAVE
# ============================================================

onnx.save(
    model,
    "prime_math_model.onnx"
)

print("ONNX model created successfully!")
print("File: prime_math_model.onnx")