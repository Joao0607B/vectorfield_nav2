# Copyright 2026 Héctor Azpúrua
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

"""wait_until_active against an in-process get_state service."""

import threading
import time

import pytest
import rclpy
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from vectorfield_bringup.lifecycle import wait_until_active


class FakeNode:
    """A get_state service for ``name``, answered from a thread of its own.

    ``held`` requests (the first ones) are answered only after ``hold`` seconds, longer than the
    callers' call timeout: to the caller, a dropped response.
    """

    def __init__(self, name, state=State.PRIMARY_STATE_ACTIVE, held=0, hold=3.0):
        self.state = state
        self.requests = 0
        self._held = held
        self._hold = hold
        self.node = rclpy.create_node(f'fake_{name}')
        group = ReentrantCallbackGroup()
        service = f'/{name}/get_state'
        self.node.create_service(GetState, service, self._get_state, callback_group=group)
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _get_state(self, request, response):
        self.requests += 1
        if self.requests <= self._held:
            time.sleep(self._hold)
        response.current_state.id = self.state
        return response

    def close(self):
        """Stop answering and destroy the node."""
        self._executor.shutdown()
        self._thread.join(timeout=5.0)
        self.node.destroy_node()


@pytest.fixture
def ros():
    """Yield the waiting node and a list whose fake nodes are closed at the end."""
    rclpy.init()
    client = rclpy.create_node('waiter')
    fakes = []
    yield client, fakes
    for fake in fakes:
        fake.close()
    client.destroy_node()
    rclpy.shutdown()


def test_active_nodes(ros):
    """Active nodes: True at once."""
    client, fakes = ros
    fakes += [FakeNode('controller_server'), FakeNode('bt_navigator')]
    assert wait_until_active(client, ['controller_server', 'bt_navigator'], timeout=20.0)


def test_a_dropped_response_is_asked_again(ros):
    """A call without a response times out and is sent again; BasicNavigator would hang."""
    client, fakes = ros
    fake = FakeNode('controller_server', held=1, hold=3.0)
    fakes.append(fake)
    start = time.monotonic()
    assert wait_until_active(client, ['controller_server'], timeout=20.0, call_timeout=0.5)
    # Answered by the second request, before the first one's late response.
    assert fake.requests >= 2
    assert time.monotonic() - start < 3.0


def test_waits_for_the_activation(ros):
    """An inactive node is asked again until it is active."""
    client, fakes = ros
    fake = FakeNode('bt_navigator', state=State.PRIMARY_STATE_INACTIVE)
    fakes.append(fake)
    threading.Timer(1.5, lambda: setattr(fake, 'state', State.PRIMARY_STATE_ACTIVE)).start()
    start = time.monotonic()
    assert wait_until_active(client, ['bt_navigator'], timeout=20.0)
    assert time.monotonic() - start >= 1.5


def test_gives_up_at_the_timeout(ros):
    """A node that stays inactive, has no service or never answers ends the wait at the timeout."""
    client, fakes = ros
    fakes.append(FakeNode('controller_server', state=State.PRIMARY_STATE_INACTIVE))
    start = time.monotonic()
    assert not wait_until_active(client, ['controller_server', 'missing_node'], timeout=2.0)
    assert time.monotonic() - start < 3.0
    start = time.monotonic()
    assert not wait_until_active(client, ['missing_node'], timeout=2.0)
    assert time.monotonic() - start < 3.0
    # A call in flight ends at the timeout too, not at its own call timeout.
    fakes.append(FakeNode('silent', held=100, hold=4.0))
    start = time.monotonic()
    assert not wait_until_active(client, ['silent'], timeout=2.0, call_timeout=10.0)
    assert time.monotonic() - start < 3.0
