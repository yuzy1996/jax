# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax._src.lib import xla_extension
from jax._src import distributed

manager = None

def initialize():
  """Initialize preemption sync manager.

  Enable multiple hosts to agree on a checkpoint step when one of the hosts
  receives a preemption notice.

  Raises:
    RuntimeError: If `initialize` is called more than once, or the distributed
    runtime client has not been initialized or connected.

  Example:

  This should primarily be integrated with checkpoint manager libraries.

  class CheckpointManager:
    def should_save(self, step: int) -> bool:
      # Should save preemption checkpoint
      if preemption_sync_manager.reached_sync_point(step):
        return True

      # Should save regular checkpoint
      return step - self._last_saved_step >= self._save_interval_steps

  """
  global manager
  if manager is not None:
    raise RuntimeError(
        'preemption_sync_manager should be initialized only once.')
  if distributed.global_state.client is None:
    raise RuntimeError('distributed runtime client must be initialized and '
                       'connected before initializing preemption_sync_manager')
  manager = xla_extension.create_preemption_sync_manager()
  manager.initialize(distributed.global_state.client)

def reached_sync_point(step_id: int) -> bool:
  if manager is None:
    raise RuntimeError('preemption_sync.initialize must be called before '
                       'preemption_sync.reached_sync_point')
  return manager.reached_sync_point(step_id)
