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

"""Wait for lifecycle nodes to become active, with a timeout on every ``get_state`` call.

``BasicNavigator.waitUntilNav2Active`` (nav2_simple_commander) waits for each ``get_state``
response without a timeout. Under load a response can be dropped: the server logs ``failed to
send response to /<node>/get_state (timeout)`` when the request arrives before it has discovered
the client's response reader, and the caller then waits forever. Here each call times out and is
retried.
"""

import time

import rclpy
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState

#: Wall seconds a get_state call waits for its response before it is sent again.
CALL_TIMEOUT = 5.0

#: Wall seconds between two get_state calls to a node that is not active yet.
RETRY_PERIOD = 0.5


def wait_until_active(node, names, timeout=None, call_timeout=CALL_TIMEOUT):
    """Return whether the lifecycle nodes ``names`` all became active within ``timeout``.

    The nodes are checked in order. ``timeout`` is in wall seconds; None waits for as long as it
    takes. ``node`` makes the calls and is spun here, so no executor may be spinning it.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    return all(_wait_active(node, name, deadline, call_timeout) for name in names)


def _wait_active(node, name, deadline, call_timeout):
    def remaining(limit):
        # Never negative: rclpy's spin_until_future_complete waits forever on a negative timeout.
        return limit if deadline is None else max(0.0, min(limit, deadline - time.monotonic()))

    client = node.create_client(GetState, f'{name}/get_state')
    logged = False
    try:
        while remaining(1.0) > 0.0:
            if client.wait_for_service(timeout_sec=remaining(1.0)):
                future = client.call_async(GetState.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=remaining(call_timeout))
                response = future.result() if future.done() else None
                if response is None:
                    # No response in time (it may have been dropped): forget it and ask again.
                    client.remove_pending_request(future)
                elif response.current_state.id == State.PRIMARY_STATE_ACTIVE:
                    return True
            if not logged:
                node.get_logger().info(f'waiting for {name} to become active')
                logged = True
            time.sleep(remaining(RETRY_PERIOD))
        return False
    finally:
        node.destroy_client(client)
