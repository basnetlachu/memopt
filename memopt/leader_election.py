"""
Phase 3: Leader Election

Implements distributed leader election for control plane coordination.
Uses etcd-style lease-based election with automatic failover.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from threading import Lock, Event
from Memopt.distributed_state import DistributedStateBackend, NodeRole


@dataclass
class LeadershipLease:
    """Leadership lease information."""
    leader_id: str
    lease_id: str
    acquired_at: float
    expires_at: float
    term: int  # Election term number

    @property
    def is_expired(self) -> bool:
        """Check if lease has expired."""
        return time.time() >= self.expires_at

    @property
    def remaining_seconds(self) -> float:
        """Get remaining lease time in seconds."""
        return max(0, self.expires_at - time.time())


class LeaderElection:
    """
    Phase 3: Distributed leader election.

    Implements Raft-style leader election with automatic failover.
    Only one node in the cluster is the leader at any time.

    The leader is responsible for:
    - Global scheduling decisions
    - Resource allocation across nodes
    - Cluster-wide rate limiting
    - Coordination of distributed operations
    """

    def __init__(
        self,
        node_id: str,
        backend: DistributedStateBackend,
        lease_duration: int = 10,
        renew_interval: int = 3
    ):
        """
        Args:
            node_id: This node's unique identifier
            backend: Distributed state backend
            lease_duration: Lease duration in seconds
            renew_interval: Seconds between lease renewals
        """
        self.node_id = node_id
        self.backend = backend
        self.lease_duration = lease_duration
        self.renew_interval = renew_interval

        # Election state
        self._lock = Lock()
        self._is_leader = False
        self._current_term = 0
        self._lease: Optional[LeadershipLease] = None

        # Background threads
        self._running = False
        self._election_thread: Optional[threading.Thread] = None
        self._stop_event = Event()

        # Callbacks
        self._on_elected: Optional[Callable[[], None]] = None
        self._on_lost_leadership: Optional[Callable[[], None]] = None

    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        with self._lock:
            return self._is_leader

    @property
    def current_term(self) -> int:
        """Get current election term."""
        with self._lock:
            return self._current_term

    def start(self):
        """Start leader election process."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        self._election_thread = threading.Thread(
            target=self._election_loop,
            daemon=True
        )
        self._election_thread.start()

    def stop(self):
        """Stop leader election process."""
        self._running = False
        self._stop_event.set()

        # Resign leadership if we're the leader
        if self._is_leader:
            self._resign_leadership()

        if self._election_thread:
            self._election_thread.join(timeout=5)

    def on_elected(self, callback: Callable[[], None]):
        """Register callback for when this node becomes leader."""
        self._on_elected = callback

    def on_lost_leadership(self, callback: Callable[[], None]):
        """Register callback for when this node loses leadership."""
        self._on_lost_leadership = callback

    def _election_loop(self):
        """Main election loop."""
        while self._running:
            try:
                if self._is_leader:
                    # We're the leader - renew lease
                    self._renew_lease()
                else:
                    # We're a follower - try to become leader
                    self._attempt_election()

                # Sleep until next iteration
                self._stop_event.wait(self.renew_interval)

            except Exception as e:
                # Log error but continue
                print(f"Election error: {e}")
                time.sleep(1)

    def _attempt_election(self):
        """Attempt to become the leader."""
        # Check current leader
        current_lease = self._get_current_lease()

        if current_lease is None or current_lease.is_expired:
            # No leader or lease expired - try to become leader
            new_term = (current_lease.term + 1) if current_lease else 1

            lease = LeadershipLease(
                leader_id=self.node_id,
                lease_id=f"lease_{self.node_id}_{new_term}",
                acquired_at=time.time(),
                expires_at=time.time() + self.lease_duration,
                term=new_term
            )

            # Try to acquire leadership via compare-and-swap
            success = self._acquire_leadership(lease, current_lease)

            if success:
                with self._lock:
                    self._is_leader = True
                    self._current_term = new_term
                    self._lease = lease

                # Trigger callback
                if self._on_elected:
                    try:
                        self._on_elected()
                    except Exception:
                        pass

    def _renew_lease(self):
        """Renew leadership lease."""
        if self._lease is None:
            return

        # Update lease expiry
        self._lease.expires_at = time.time() + self.lease_duration

        # Write to backend
        key = "/Memopt/leader/lease"
        value = self._serialize_lease(self._lease)

        success = self.backend.set(key, value, ttl=self.lease_duration)

        if not success:
            # Failed to renew - lost leadership
            self._handle_leadership_lost()

    def _acquire_leadership(
        self,
        new_lease: LeadershipLease,
        old_lease: Optional[LeadershipLease]
    ) -> bool:
        """
        Attempt to acquire leadership via atomic compare-and-swap.

        Args:
            new_lease: New lease to acquire
            old_lease: Current lease (or None)

        Returns:
            True if leadership acquired
        """
        key = "/Memopt/leader/lease"
        new_value = self._serialize_lease(new_lease)

        if old_lease is None:
            # No current leader - try to set
            return self.backend.set(key, new_value, ttl=self.lease_duration)
        else:
            # Current leader exists - use compare-and-swap
            old_value = self._serialize_lease(old_lease)
            return self.backend.compare_and_swap(key, old_value, new_value)

    def _resign_leadership(self):
        """Resign from leadership."""
        with self._lock:
            if not self._is_leader:
                return

            self._is_leader = False
            self._lease = None

        # Delete lease from backend
        self.backend.delete("/Memopt/leader/lease")

        # Trigger callback
        if self._on_lost_leadership:
            try:
                self._on_lost_leadership()
            except Exception:
                pass

    def _handle_leadership_lost(self):
        """Handle losing leadership."""
        with self._lock:
            was_leader = self._is_leader
            self._is_leader = False
            self._lease = None

        if was_leader and self._on_lost_leadership:
            try:
                self._on_lost_leadership()
            except Exception:
                pass

    def _get_current_lease(self) -> Optional[LeadershipLease]:
        """Get current leadership lease from backend."""
        value = self.backend.get("/Memopt/leader/lease")
        if value:
            return self._deserialize_lease(value)
        return None

    def _serialize_lease(self, lease: LeadershipLease) -> str:
        """Serialize lease to string."""
        import json
        return json.dumps({
            "leader_id": lease.leader_id,
            "lease_id": lease.lease_id,
            "acquired_at": lease.acquired_at,
            "expires_at": lease.expires_at,
            "term": lease.term
        })

    def _deserialize_lease(self, value: str) -> LeadershipLease:
        """Deserialize lease from string."""
        import json
        data = json.loads(value)
        return LeadershipLease(**data)

    def get_leader_info(self) -> Optional[dict]:
        """
        Get information about current leader.

        Returns:
            Dict with leader info or None if no leader
        """
        lease = self._get_current_lease()
        if lease and not lease.is_expired:
            return {
                "leader_id": lease.leader_id,
                "term": lease.term,
                "remaining_seconds": lease.remaining_seconds,
                "is_self": lease.leader_id == self.node_id
            }
        return None


# ============================================================================
# Phase 3: Leadership Coordinator
# ============================================================================

class LeadershipCoordinator:
    """
    Phase 3: High-level coordinator for leader-specific tasks.

    The leader performs cluster-wide coordination:
    - Global request scheduling
    - Cross-node load balancing
    - Cluster-wide rate limiting
    - Resource allocation
    """

    def __init__(self, election: LeaderElection):
        """
        Args:
            election: Leader election instance
        """
        self.election = election
        self._tasks: list[Callable] = []

        # Register callbacks
        election.on_elected(self._on_became_leader)
        election.on_lost_leadership(self._on_lost_leadership)

    def _on_became_leader(self):
        """Called when this node becomes leader."""
        print(f"Node became leader (term {self.election.current_term})")

        # Start leader-specific tasks
        for task in self._tasks:
            try:
                task()
            except Exception as e:
                print(f"Leader task error: {e}")

    def _on_lost_leadership(self):
        """Called when this node loses leadership."""
        print("Node lost leadership")
        # Stop leader-specific tasks (handled by election stop)

    def register_leader_task(self, task: Callable):
        """
        Register a task to run when this node is leader.

        Args:
            task: Function to execute when leader
        """
        self._tasks.append(task)

    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self.election.is_leader
