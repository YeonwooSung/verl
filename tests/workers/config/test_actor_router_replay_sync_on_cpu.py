# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.workers.config.actor import McoreActorConfig, RouterReplayConfig, VeOmniActorConfig
from verl.workers.config.engine import EngineRouterReplayConfig, McoreEngineConfig, VeOmniEngineConfig
from verl.workers.config.optimizer import OptimizerConfig


def test_mcore_copies_actor_router_replay_onto_disabled_engine():
    cfg = McoreActorConfig(
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        router_replay=RouterReplayConfig(mode="R3", record_file="rec.json"),
        megatron=McoreEngineConfig(router_replay=EngineRouterReplayConfig(mode="disabled")),
        optim=OptimizerConfig(lr=1e-6),
    )
    assert cfg.megatron.router_replay.mode == "R3"
    assert cfg.megatron.router_replay.record_file == "rec.json"


def test_veomni_copies_actor_router_replay_onto_disabled_engine():
    cfg = VeOmniActorConfig(
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        use_remove_padding=True,
        router_replay=RouterReplayConfig(mode="R2"),
        veomni=VeOmniEngineConfig(router_replay=EngineRouterReplayConfig(mode="disabled")),
        optim=OptimizerConfig(lr=1e-6),
    )
    assert cfg.veomni.router_replay.mode == "R2"


def test_conflicting_router_replay_modes_raise():
    with pytest.raises(ValueError, match="Conflicting router_replay modes"):
        McoreActorConfig(
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            router_replay=RouterReplayConfig(mode="R2"),
            megatron=McoreEngineConfig(router_replay=EngineRouterReplayConfig(mode="R3")),
            optim=OptimizerConfig(lr=1e-6),
        )


def test_engine_mode_kept_when_actor_is_disabled():
    cfg = McoreActorConfig(
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        router_replay=RouterReplayConfig(mode="disabled"),
        megatron=McoreEngineConfig(router_replay=EngineRouterReplayConfig(mode="R3")),
        optim=OptimizerConfig(lr=1e-6),
    )
    assert cfg.megatron.router_replay.mode == "R3"
