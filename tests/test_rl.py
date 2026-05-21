"""Unit tests for RL agent module."""

import pytest
import numpy as np
from src.models.rl_agent.collection_env import CollectionEnv

def test_collection_env_step():
    """Test that the environment step() function returns correct shapes."""
    env = CollectionEnv()
    obs = env.reset()
    
    assert obs.shape == (20,)
    assert env.action_space.n == 9
    
    # take a step
    next_obs, reward, done, info = env.step(0)
    
    assert next_obs.shape == (20,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
