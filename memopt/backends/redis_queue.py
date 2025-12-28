"""
Production Redis Streams-Based Request Queue

Provides distributed, persistent request queue with:
- Consumer groups for multi-worker consumption
- At-least-once delivery semantics
- Automatic retry on failure
- Dead-letter queue for failed requests
- Backpressure via queue depth limits

Architecture:
- Redis Streams for message storage
- Consumer groups for load distribution
- ACK mechanism for reliability
- MAXLEN for memory limits
"""

import time
import json
import uuid
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    if TYPE_CHECKING:
        import redis
    else:
        redis = None  # type: ignore
    RedisError = Exception

from memopt.scheduler import InferenceRequest
from memopt.exceptions import QueueFullError


logger = logging.getLogger(__name__)


@dataclass
class RedisQueueConfig:
    """Redis queue configuration."""
    stream_name: str = "memopt:requests"
    consumer_group: str = "memopt-workers"
    consumer_id: str = None  # Auto-generated if None
    dead_letter_stream: str = "memopt:requests:dlq"
    max_len: int = 100000  # Max messages in stream (backpressure)
    max_retries: int = 3
    retry_delay_ms: int = 5000
    claim_timeout_ms: int = 60000  # Claim messages idle > 1 min
    block_ms: int = 5000  # Block time when reading
    batch_size: int = 10  # Read batch size


class RedisRequestQueue:
    """
    Production distributed request queue using Redis Streams.

    Features:
    - Distributed: Multiple workers consume from same stream
    - Persistent: Survives worker crashes
    - Reliable: At-least-once delivery with ACK
    - Backpressure: Queue depth limits
    - DLQ: Failed messages after max retries

    Redis Streams Primer:
    - XADD: Add message to stream
    - XREADGROUP: Read messages as consumer group
    - XACK: Acknowledge message processed
    - XPENDING: List unacknowledged messages
    - XCLAIM: Claim abandoned messages
    """

    def __init__(
        self,
        redis_client: "redis.Redis",
        config: Optional[RedisQueueConfig] = None
    ):
        """
        Initialize Redis queue.

        Args:
            redis_client: Connected Redis client
            config: Queue configuration

        Raises:
            ImportError: If redis-py not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis-py required. Install: pip install redis>=4.5.0")

        self._client = redis_client
        self.config = config or RedisQueueConfig()

        # Auto-generate consumer ID if not provided
        if self.config.consumer_id is None:
            self.config.consumer_id = f"worker-{uuid.uuid4().hex[:8]}"

        # Create consumer group if doesn't exist
        self._ensure_consumer_group()

        logger.info(
            f"RedisRequestQueue initialized: stream={self.config.stream_name}, "
            f"group={self.config.consumer_group}, consumer={self.config.consumer_id}"
        )

    def _ensure_consumer_group(self):
        """Create consumer group if it doesn't exist."""
        try:
            # Try to create group from beginning of stream
            self._client.xgroup_create(
                name=self.config.stream_name,
                groupname=self.config.consumer_group,
                id='0',
                mkstream=True
            )
            logger.info(f"Created consumer group: {self.config.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists, that's fine
                logger.debug(f"Consumer group already exists: {self.config.consumer_group}")
            else:
                raise

    def enqueue(self, request: InferenceRequest, max_retries: int = 3) -> str:
        """
        Add request to queue.

        Args:
            request: Inference request to queue
            max_retries: Max retry attempts

        Returns:
            Message ID in stream

        Raises:
            QueueFullError: If stream at max length
            RedisError: On Redis failure
        """
        # Check queue depth
        stream_len = self._client.xlen(self.config.stream_name)
        if stream_len >= self.config.max_len:
            raise QueueFullError(
                f"Queue full: {stream_len}/{self.config.max_len} messages"
            )

        # Serialize request
        message = {
            'request_id': request.request_id,
            'request_data': json.dumps({
                'prompt': request.prompt,
                'max_tokens': request.max_tokens,
                'temperature': request.temperature,
                'priority': request.priority,
                'tenant_id': request.tenant_id,
                'deadline': request.deadline,
                'created_at': request.created_at
            }),
            'retry_count': '0',
            'max_retries': str(max_retries),
            'enqueued_at': str(time.time())
        }

        # Add to stream with MAXLEN for backpressure
        # ~ means approximate trimming (more efficient)
        message_id = self._client.xadd(
            name=self.config.stream_name,
            fields=message,
            maxlen=self.config.max_len,
            approximate=True
        )

        logger.debug(f"Enqueued request {request.request_id} as {message_id}")
        return message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id

    def dequeue(self, block: bool = True) -> Optional[tuple]:
        """
        Dequeue next request from stream.

        Returns:
            Tuple of (message_id, InferenceRequest) or None

        This uses XREADGROUP which:
        - Reads messages not yet delivered to this consumer group
        - Automatically marks message as pending (unacknowledged)
        - Supports blocking with timeout
        """
        # First, try to claim abandoned messages
        claimed = self._claim_abandoned_messages()
        if claimed:
            return claimed[0]

        # Read new messages from stream
        block_ms = self.config.block_ms if block else 0

        try:
            messages = self._client.xreadgroup(
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_id,
                streams={self.config.stream_name: '>'},  # '>' means only new messages
                count=1,
                block=block_ms
            )

            if not messages:
                return None

            # Parse response: [(stream_name, [(message_id, fields)])]
            stream_name, message_list = messages[0]
            if not message_list:
                return None

            message_id, fields = message_list[0]
            return self._parse_message(message_id, fields)

        except RedisError as e:
            logger.error(f"Failed to dequeue: {e}")
            return None

    def _claim_abandoned_messages(self) -> List[tuple]:
        """
        Claim messages that were delivered but not ACKed (worker crashed).

        Returns:
            List of (message_id, InferenceRequest) tuples
        """
        try:
            # Get pending messages idle > claim_timeout
            pending = self._client.xpending_range(
                name=self.config.stream_name,
                groupname=self.config.consumer_group,
                min='-',
                max='+',
                count=self.config.batch_size,
                idle=self.config.claim_timeout_ms
            )

            if not pending:
                return []

            # Claim messages for this consumer
            message_ids = [p['message_id'] for p in pending]
            claimed = self._client.xclaim(
                name=self.config.stream_name,
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_id,
                min_idle_time=self.config.claim_timeout_ms,
                message_ids=message_ids
            )

            result = []
            for message_id, fields in claimed:
                parsed = self._parse_message(message_id, fields)
                if parsed:
                    result.append(parsed)
                    logger.info(f"Claimed abandoned message: {message_id}")

            return result

        except RedisError as e:
            logger.error(f"Failed to claim messages: {e}")
            return []

    def _parse_message(
        self,
        message_id: bytes,
        fields: Dict[bytes, bytes]
    ) -> Optional[tuple]:
        """Parse Redis Stream message into InferenceRequest."""
        try:
            # Decode bytes to strings
            fields_str = {
                k.decode('utf-8'): v.decode('utf-8')
                for k, v in fields.items()
            }

            request_data = json.loads(fields_str['request_data'])

            request = InferenceRequest(
                request_id=fields_str['request_id'],
                prompt=request_data['prompt'],
                max_tokens=request_data['max_tokens'],
                temperature=request_data.get('temperature', 0.7),
                priority=request_data.get('priority', 0),
                tenant_id=request_data.get('tenant_id'),
                deadline=request_data.get('deadline'),
                created_at=request_data['created_at']
            )

            msg_id = message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id
            return (msg_id, request)

        except Exception as e:
            logger.error(f"Failed to parse message {message_id}: {e}")
            return None

    def ack(self, message_id: str) -> bool:
        """
        Acknowledge message processed successfully.

        Args:
            message_id: Stream message ID

        Returns:
            True if ACKed
        """
        try:
            result = self._client.xack(
                self.config.stream_name,
                self.config.consumer_group,
                message_id
            )
            logger.debug(f"ACKed message: {message_id}")
            return result > 0
        except RedisError as e:
            logger.error(f"Failed to ACK {message_id}: {e}")
            return False

    def nack(self, message_id: str, request: InferenceRequest) -> bool:
        """
        Negative acknowledge - message processing failed.

        Will retry if under max_retries, else move to DLQ.

        Args:
            message_id: Stream message ID
            request: Original request

        Returns:
            True if requeued or moved to DLQ
        """
        try:
            # Get message metadata
            messages = self._client.xrange(
                self.config.stream_name,
                min=message_id,
                max=message_id,
                count=1
            )

            if not messages:
                logger.warning(f"Message {message_id} not found for NACK")
                return False

            _, fields = messages[0]
            fields_str = {k.decode('utf-8'): v.decode('utf-8') for k, v in fields.items()}

            retry_count = int(fields_str.get('retry_count', 0))
            max_retries = int(fields_str.get('max_retries', self.config.max_retries))

            # ACK original message
            self._client.xack(
                self.config.stream_name,
                self.config.consumer_group,
                message_id
            )

            if retry_count < max_retries:
                # Retry: re-enqueue with incremented retry count
                retry_message = dict(fields_str)
                retry_message['retry_count'] = str(retry_count + 1)
                retry_message['last_retry_at'] = str(time.time())

                self._client.xadd(
                    name=self.config.stream_name,
                    fields=retry_message,
                    maxlen=self.config.max_len,
                    approximate=True
                )

                logger.info(
                    f"Retrying request {request.request_id} "
                    f"(attempt {retry_count + 1}/{max_retries})"
                )
                return True
            else:
                # Max retries exceeded: move to DLQ
                dlq_message = dict(fields_str)
                dlq_message['failed_at'] = str(time.time())
                dlq_message['original_message_id'] = message_id

                self._client.xadd(
                    name=self.config.dead_letter_stream,
                    fields=dlq_message
                )

                logger.error(
                    f"Request {request.request_id} moved to DLQ after "
                    f"{max_retries} failed attempts"
                )
                return True

        except RedisError as e:
            logger.error(f"Failed to NACK {message_id}: {e}")
            return False

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self._client.xlen(self.config.stream_name)

    def get_pending_count(self) -> int:
        """Get count of pending (unacknowledged) messages."""
        try:
            info = self._client.xpending(
                self.config.stream_name,
                self.config.consumer_group
            )
            return info['pending']
        except RedisError:
            return 0

    def get_dlq_depth(self) -> int:
        """Get dead-letter queue depth."""
        return self._client.xlen(self.config.dead_letter_stream)

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'stream_name': self.config.stream_name,
            'consumer_group': self.config.consumer_group,
            'consumer_id': self.config.consumer_id,
            'queue_depth': self.get_queue_depth(),
            'pending_count': self.get_pending_count(),
            'dlq_depth': self.get_dlq_depth(),
            'max_len': self.config.max_len
        }

    def close(self):
        """Cleanup (Redis client managed externally)."""
        logger.info(f"RedisRequestQueue consumer {self.config.consumer_id} closed")


# Worker consumption loop
def consume_requests(
    queue: RedisRequestQueue,
    process_fn,
    shutdown_event=None
):
    """
    Worker loop for consuming and processing requests.

    Args:
        queue: RedisRequestQueue instance
        process_fn: Function to process requests: fn(request) -> bool
        shutdown_event: Threading event to signal shutdown

    Example:
        def process_request(request: InferenceRequest) -> bool:
            try:
                # Execute inference
                result = model.generate(request.prompt)
                return True
            except Exception:
                return False

        consume_requests(queue, process_request)
    """
    logger.info("Worker started consuming requests")

    while True:
        if shutdown_event and shutdown_event.is_set():
            logger.info("Shutdown signal received, stopping worker")
            break

        try:
            # Dequeue next request
            result = queue.dequeue(block=True)

            if result is None:
                continue

            message_id, request = result

            # Process request
            success = process_fn(request)

            # ACK or NACK
            if success:
                queue.ack(message_id)
            else:
                queue.nack(message_id, request)

        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(1)  # Brief pause on error

    logger.info("Worker stopped")
