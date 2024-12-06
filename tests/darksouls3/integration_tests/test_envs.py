from functools import cache
import warnings

import pytest
import gymnasium
import soulsgym  # noqa: F401, need to import to register environments
from soulsgym.core.utils import get_pid
from soulsgym.games.darksouls3 import DarkSoulsIII


@cache
def game_not_open() -> bool:
    try:
        get_pid("DarkSoulsIII.exe")
        e = DarkSoulsIII()
        if not e.is_ingame:
            warnings.warn("DarkSoulsIII: Player is not in-game.")
            return True
        return False
    except RuntimeError:
        return True


def run_env(env_name: str, kwargs: dict | None = None):
    env = gymnasium.make(env_name, **kwargs if kwargs is not None else {})
    try:
        env.reset()
        terminated, truncated = False, False
        while not terminated or truncated:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    finally:
        env.close()
        env.unwrapped.game.reload()


@pytest.mark.integration
@pytest.mark.iudex
@pytest.mark.skipif(game_not_open(), reason="Dark Souls III is not running.")
@pytest.mark.parametrize("kwargs", [{}, {"game_speed": 2, "random_player_pose": True}])
def test_iudex(kwargs: dict | None):
    env = "SoulsGymIudex-v0"
    run_env(env, kwargs)


@pytest.mark.integration
@pytest.mark.vordt
@pytest.mark.skipif(game_not_open(), reason="Dark Souls III is not running.")
@pytest.mark.parametrize("kwargs", [{}, {"game_speed": 2}])
def test_vordt(kwargs: dict | None):
    env = "SoulsGymVordt-v0"
    run_env(env, kwargs)
