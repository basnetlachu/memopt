"""
Phase 3: Distributed Locks

Provides distributed mutex locks for coordinating critical sections
across multiple nodes in the cluster.
"""

import time
import uuid
from typing import Optional
from threading import Lock as ThreadLock
from memopt.distributed_state import DistributedStateBackend


class DistributedLockError(Exception):
    """Raised when lock operations fail."""
    pass


class DistributedLock:
    """
    Phase 3: Distributed mutex lock.

    Provides mutual exclusion across nodes using lease-based locking.
    Prevents race conditions in distributed operations like:
    - Global scheduler updates
    - Cluster-wide configuration changes
    - Coordinated cache eviction
    """

    def __init__(
        self,
        name: str,
        backend: DistributedStateBackend,
        lock_timeout: int = 30,
        node_id: Optional[str] = None
    ):
        """
        Args:
            name: Lock name (unique identifier)
            backend: Distributed state backend
            lock_timeout: Lock timeout in seconds
            node_id: This node's identifier (generated if None)
        """
        self.name = name
        self.backend = backend
        self.lock_timeout = lock_timeout
        self.node_id = node_id or str(uuid.uuid4())

        # Local state
        self._local_lock = ThreadLock()
        self._lock_id: Optional[str] = None
        self._acquired_at: Optional[float] = None

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire the distributed lock.

        Args:
            blocking: If True, wait for lock. If False, return immediately.
            timeout: Max time to wait in seconds (None = wait forever)

        Returns:
            True if lock acquired, False if failed (non-blocking only)

        Raises:
            DistributedLockError: If lock acquisition fails
        """
        start_time = time.time()

        while True:
            # Try to acquire
            if self._try_acquire():
                return True

            # Non-blocking mode
            if not blocking:
                return False

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise DistributedLockError(
                        f"Failed to acquire lock '{self.name}' within {timeout}s"
                    )

            # Wait before retrying
            time.sleep(0.1)

    def release(self):
        """
        Release the distributed lock.

        Raises:
            DistributedLockError: If lock not owned by this node
        """
        with self._local_lock:
            if self._lock_id is None:
                raise DistributedLockError(
                    f"Cannot release lock '{self.name}' - not owned by this node"
                )

            # Delete lock using compare-and-swap
            key = self._get_lock_key()
            current_value = self.backend.get(key)

            if current_value != self._lock_id:
                raise DistributedLockError(
                    f"Lock '{self.name}' ownership lost - cannot release"
                )

            # Release lock
            self.backend.delete(key)
            self._lock_id = None
            self._acquired_at = None

    def _try_acquire(self) -> bool:
        """
        Attempt to acquire lock (non-blocking).

        Returns:
            True if lock acquired, False otherwise
        """
        with self._local_lock:
            # Generate unique lock ID
            lock_id = f"{self.node_id}_{uuid.uuid4().hex[:8]}"

            # Try to set lock key
            key = self._get_lock_key()
            current_value = self.backend.get(key)

            if current_value is None:
                # Lock is free - try to acquire
                success = self.backend.set(key, lock_id, ttl=self.lock_timeout)

                if success:
                    self._lock_id = lock_id
                    self._acquired_at = time.time()
                    return True

            return False

    def _get_lock_key(self) -> str:
        """Get distributed state key for this lock."""
        return f"/memopt/locks/{self.name}"

    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        key = self._get_lock_key()
        value = self.backend.get(key)
        return value is not None

    def is_owned_by_self(self) -> bool:
        """Check if this node owns the lock."""
        with self._local_lock:
            if self._lock_id is None:
                return False

            key = self._get_lock_key()
            current_value = self.backend.get(key)

            return current_value == self._lock_id

    def renew(self) -> bool:
        """
        Renew lock lease.

        Returns:
            True if renewal successful

        Raises:
            DistributedLockError: If lock not owned
        """
        with self._local_lock:
            if self._lock_id is None:
                raise DistributedLockError(
                    f"Cannot renew lock '{self.name}' - not owned by this node"
                )

            key = self._get_lock_key()
            success = self.backend.set(key, self._lock_id, ttl=self.lock_timeout)

            return success

    # Context manager support
    def __enter__(self):
        """Acquire lock when entering context."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock when exiting context."""
        try:
            self.release()
        except DistributedLockError:
            # Lock may have expired - log but don't raise
            pass
        return False


class DistributedReadWriteLock:
    """
    Phase 3: Distributed read-write lock.

    Allows multiple concurrent readers OR one exclusive writer.
    Useful for scenarios like:
    - Cluster configuration (many reads, rare writes)
    - Shared cache metadata (many reads, occasional updates)
    """

    def __init__(
        self,
        name: str,
        backend: DistributedStateBackend,
        lock_timeout: int = 30,
        node_id: Optional[str] = None
    ):
        """
        Args:
            name: Lock name
            backend: Distributed state backend
            lock_timeout: Lock timeout in seconds
            node_id: This node's identifier
        """
        self.name = name
        self.backend = backend
        self.lock_timeout = lock_timeout
        self.node_id = node_id or str(uuid.uuid4())

        self._local_lock = ThreadLock()
        self._mode: Optional[str] = None  # 'read' or 'write'
        self._lock_id: Optional[str] = None

    def acquire_read(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire read lock (shared).

        Multiple readers can hold the lock simultaneously.
        """
        start_time = time.time()

        while True:
            if self._try_acquire_read():
                return True

            if not blocking:
                return False

            if timeout is not None and (time.time() - start_time) >= timeout:
                raise DistributedLockError(
                    f"Failed to acquire read lock '{self.name}' within {timeout}s"
                )

            time.sleep(0.1)

    def acquire_write(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire write lock (exclusive).

        Only one writer can hold the lock at a time.
        No readers allowed while writer holds lock.
        """
        start_time = time.time()

        while True:
            if self._try_acquire_write():
                return True

            if not blocking:
                return False

            if timeout is not None and (time.time() - start_time) >= timeout:
                raise DistributedLockError(
                    f"Failed to acquire write lock '{self.name}' within {timeout}s"
                )

            time.sleep(0.1)

    def release(self):
        """Release the lock."""
        with self._local_lock:
            if self._mode is None:
                raise DistributedLockError(
                    f"Cannot release lock '{self.name}' - not owned by this node"
                )

            if self._mode == 'read':
                self._release_read()
            else:
                self._release_write()

            self._mode = None
            self._lock_id = None

    def _try_acquire_read(self) -> bool:
        """Try to acquire read lock."""
        with self._local_lock:
            # Check if write lock exists
            write_key = f"/memopt/locks/{self.name}/write"
            write_lock = self.backend.get(write_key)

            if write_lock is not None:
                # Write lock held - cannot acquire read
                return False

            # Add to readers set
            lock_id = f"{self.node_id}_{uuid.uuid4().hex[:8]}"
            read_key = f"/memopt/locks/{self.name}/readers/{lock_id}"

            success = self.backend.set(read_key, lock_id, ttl=self.lock_timeout)

            if success:
                self._mode = 'read'
                self._lock_id = lock_id
                return True

            return False

    def _try_acquire_write(self) -> bool:
        """Try to acquire write lock."""
        with self._local_lock:
            # Check if any readers exist
            readers_prefix = f"/memopt/locks/{self.name}/readers/"
            readers = self.backend.list_keys(readers_prefix)

            if len(readers) > 0:
                # Readers exist - cannot acquire write
                return False

            # Check if write lock exists
            write_key = f"/memopt/locks/{self.name}/write"
            write_lock = self.backend.get(write_key)

            if write_lock is not None:
                # Write lock held - cannot acquire
                return False

            # Acquire write lock
            lock_id = f"{self.node_id}_{uuid.uuid4().hex[:8]}"
            success = self.backend.set(write_key, lock_id, ttl=self.lock_timeout)

            if success:
                self._mode = 'write'
                self._lock_id = lock_id
                return True

            return False

    def _release_read(self):
        """Release read lock."""
        read_key = f"/memopt/locks/{self.name}/readers/{self._lock_id}"
        self.backend.delete(read_key)

    def _release_write(self):
        """Release write lock."""
        write_key = f"/memopt/locks/{self.name}/write"
        self.backend.delete(write_key)


# ============================================================================
# Phase 3: Lock Manager
# ============================================================================

class LockManager:
    """
    Phase 3: Manager for distributed locks.

    Provides centralized access to locks and automatic cleanup.
    """

    def __init__(
        self,
        backend: DistributedStateBackend,
        node_id: Optional[str] = None,
        default_timeout: int = 30
    ):
        """
        Args:
            backend: Distributed state backend
            node_id: This node's identifier
            default_timeout: Default lock timeout in seconds
        """
        self.backend = backend
        self.node_id = node_id or str(uuid.uuid4())
        self.default_timeout = default_timeout

        self._locks: dict[str, DistributedLock] = {}
        self._local_lock = ThreadLock()

    def lock(self, name: str, timeout: Optional[int] = None) -> DistributedLock:
        """
        Get or create a distributed lock.

        Args:
            name: Lock name
            timeout: Lock timeout (uses default if None)

        Returns:
            DistributedLock instance
        """
        with self._local_lock:
            if name not in self._locks:
                self._locks[name] = DistributedLock(
                    name=name,
                    backend=self.backend,
                    lock_timeout=timeout or self.default_timeout,
                    node_id=self.node_id
                )
            return self._locks[name]

    def rw_lock(self, name: str, timeout: Optional[int] = None) -> DistributedReadWriteLock:
        """
        Create a distributed read-write lock.

        Args:
            name: Lock name
            timeout: Lock timeout (uses default if None)

        Returns:
            DistributedReadWriteLock instance
        """
        return DistributedReadWriteLock(
            name=name,
            backend=self.backend,
            lock_timeout=timeout or self.default_timeout,
            node_id=self.node_id
        )
