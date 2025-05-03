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
