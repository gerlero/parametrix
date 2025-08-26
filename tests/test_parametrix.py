import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from parametrix import Param


def test_create() -> None:
    param = Param(42.0)
    assert isinstance(param, Param)
    assert isinstance(param.value, jax.Array)
    assert param.value == 42.0
    assert param.value.shape == ()
    assert param.value.dtype == jnp.float32
    assert isinstance(param.raw_value, jax.Array)
    assert param.raw_value == 42.0
    assert param.raw_value.shape == ()
    assert param.raw_value.dtype == jnp.float32


def test_ops() -> None:
    param = Param(42.0)
    assert param + 1 == 43.0
    assert param - 1 == 41.0
    assert param * 2 == 84.0
    assert param / 2 == 21.0
    assert param % 5 == 2.0
    assert param**2 == 1764.0
    assert param // 5 == 8
    assert -param == -42.0
    assert +param == 42.0
    assert abs(param) == 42.0


def test_equinox() -> None:
    class Model(eqx.Module):
        param: Param = Param(42.0)

    model = Model()
    optim = optax.adam(learning_rate=1e-2)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: Model, opt_state: optax.OptState
    ) -> tuple[Model, optax.OptState]:
        def loss_fn(model: Model) -> jax.Array:
            return model.param**2

        grads = eqx.filter_grad(loss_fn)(model)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    for _ in range(10_000):
        model, opt_state = train_step(model, opt_state)

    assert model.param.value == pytest.approx(0, abs=1e-6)


def test_array_like_behavior() -> None:
    """Test array-like behavior methods."""
    # Test with a 1D array
    param = Param([1.0, 2.0, 3.0])

    # Test __jax_array__ - it should return the value as a JAX array
    jax_array = param.__jax_array__()
    assert isinstance(jax_array, jax.Array)
    assert jnp.array_equal(jax_array, jnp.array([1.0, 2.0, 3.0]))

    # Test __getitem__
    assert param[0] == 1.0
    assert param[1] == 2.0
    assert param[-1] == 3.0

    # Test __len__
    assert len(param) == 3

    # Test __iter__
    values = list(param)
    assert values == [1.0, 2.0, 3.0]

    # Test __contains__
    assert 1.0 in param
    assert 2.0 in param
    assert 4.0 not in param


def test_param_to_param_operations() -> None:
    """Test operations between two Param objects."""
    param1 = Param(10.0)
    param2 = Param(3.0)

    # Test basic operations with Param objects
    assert param1 + param2 == 13.0
    assert param1 - param2 == 7.0
    assert param1 * param2 == 30.0
    assert param1 / param2 == pytest.approx(3.333333, abs=1e-5)
    assert param1 % param2 == 1.0
    assert param1 ** param2 == 1000.0
    assert param1 // param2 == 3.0


def test_matrix_operations() -> None:
    """Test matrix multiplication operations."""
    # Test 2x2 matrices
    param1 = Param([[1.0, 2.0], [3.0, 4.0]])
    param2 = Param([[5.0, 6.0], [7.0, 8.0]])

    # Test matrix multiplication
    result = param1 @ param2
    expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
    assert jnp.allclose(result, expected)

    # Test with regular array
    array = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    result2 = param1 @ array
    expected2 = jnp.array([[2.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(result2, expected2)


def test_divmod_operations() -> None:
    """Test divmod operations."""
    param = Param(17.0)

    # Test divmod with scalar
    quot, rem = divmod(param, 5.0)
    assert quot == 3.0
    assert rem == 2.0

    # Test with another Param
    param2 = Param(5.0)
    quot2, rem2 = divmod(param, param2)
    assert quot2 == 3.0
    assert rem2 == 2.0


def test_reverse_operations() -> None:
    """Test reverse (right-hand side) operations."""
    param = Param(5.0)

    # Test reverse operations
    assert 10 + param == 15.0  # __radd__
    assert 10 - param == 5.0   # __rsub__
    assert 3 * param == 15.0   # __rmul__
    assert 20 / param == 4.0   # __rtruediv__
    assert 17 // param == 3.0  # __rfloordiv__
    assert 17 % param == 2.0   # __rmod__
    assert 2 ** param == 32.0  # __rpow__

    # Test reverse divmod
    quot, rem = divmod(17, param)
    assert quot == 3.0
    assert rem == 2.0

    # Test reverse matrix multiplication
    param_matrix = Param([[1.0, 2.0], [3.0, 4.0]])
    array = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    result = array @ param_matrix
    expected = jnp.array([[2.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(result, expected)


def test_type_conversions() -> None:
    """Test type conversion methods."""
    # Test complex conversion
    param_complex = Param(3.0 + 4.0j)
    assert complex(param_complex) == 3.0 + 4.0j

    # Test int conversion
    param_int = Param(42.7)
    assert int(param_int) == 42

    # Test float conversion
    param_float = Param(42.5)
    assert float(param_float) == 42.5

    # Test index conversion (for use as array index)
    param_index = Param(3)
    assert param_index.__index__() == 3


def test_additional_operations() -> None:
    """Test additional operations and methods."""
    # Test invert operation (bitwise NOT)
    param_int = Param(5)  # binary: 101
    inverted = ~param_int
    assert inverted == -6  # ~5 = -6 in two's complement

    # Test round operation
    param_float = Param(3.7)
    rounded = round(param_float, 0)
    assert rounded == 4.0

    param_precise = Param(3.14159)
    rounded_precise = round(param_precise, 2)
    assert rounded_precise == pytest.approx(3.14, abs=1e-5)


def test_reverse_param_to_param_operations() -> None:
    """Test reverse operations when both operands are Param objects."""
    param1 = Param(10.0)
    param2 = Param(3.0)

    # Test reverse operations with Param objects on both sides
    # These test the isinstance(other, Param) branches in reverse operations
    assert param2.__radd__(param1) == 13.0
    assert param2.__rsub__(param1) == 7.0
    assert param2.__rmul__(param1) == 30.0
    assert param2.__rtruediv__(param1) == pytest.approx(3.333333, abs=1e-5)
    assert param2.__rfloordiv__(param1) == 3.0
    assert param2.__rmod__(param1) == 1.0
    assert param2.__rpow__(param1) == 1000.0

    # Test reverse divmod with Param objects
    quot, rem = param2.__rdivmod__(param1)
    assert quot == 3.0
    assert rem == 1.0

    # Test reverse matrix multiplication with Param objects
    param_matrix1 = Param([[1.0, 2.0], [3.0, 4.0]])
    param_matrix2 = Param([[2.0, 0.0], [0.0, 2.0]])
    result = param_matrix1.__rmatmul__(param_matrix2)
    expected = jnp.array([[2.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(result, expected)
