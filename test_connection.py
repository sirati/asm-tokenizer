#!/usr/bin/env python3
"""Test script for primary-secondary connection via SSH port forwarding"""

import asyncio
import logging
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_primary():
    """Test primary coordinator listening"""
    logger.info("=" * 60)
    logger.info("TEST PRIMARY")
    logger.info("=" * 60)

    async def handle_client(reader, writer):
        addr = writer.get_extra_info("peername")
        logger.info(f"Client connected from {addr}")

        try:
            # Read length prefix (4 bytes)
            length_bytes = await reader.readexactly(4)
            msg_length = int.from_bytes(length_bytes, "big")
            logger.info(f"Receiving message of length {msg_length}")

            # Read message
            msg_bytes = await reader.readexactly(msg_length)
            msg_str = msg_bytes.decode("utf-8")
            logger.info(f"Received: {msg_str}")

            # Send response
            response = '{"type":"entropy","entropy_hex":"test123"}'
            response_bytes = response.encode("utf-8")
            length_bytes = len(response_bytes).to_bytes(4, "big")
            writer.write(length_bytes)
            writer.write(response_bytes)
            await writer.drain()
            logger.info(f"Sent response: {response}")

        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    # Start server
    server = await asyncio.start_server(handle_client, "0.0.0.0", 5000)
    logger.info("Primary listening on 0.0.0.0:5000")
    logger.info("Use SSH port forwarding: ssh -R 6000:localhost:5000 gateway")
    logger.info("")

    async with server:
        await server.serve_forever()


async def test_secondary(gateway_host: str, gateway_port: int):
    """Test secondary connecting to gateway"""
    logger.info("=" * 60)
    logger.info("TEST SECONDARY")
    logger.info("=" * 60)
    logger.info(f"Connecting to {gateway_host}:{gateway_port}")

    try:
        reader, writer = await asyncio.open_connection(gateway_host, gateway_port)
        logger.info("Connected successfully")

        # Send test message
        msg = '{"type":"secondary_welcome","secondary_id":"test-0","ram_bytes":8589934592,"worker_count":4,"hostname":"test-node"}'
        msg_bytes = msg.encode("utf-8")
        length_bytes = len(msg_bytes).to_bytes(4, "big")

        writer.write(length_bytes)
        writer.write(msg_bytes)
        await writer.drain()
        logger.info(f"Sent welcome message: {msg}")

        # Wait for response
        logger.info("Waiting for response...")
        length_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=10.0)
        msg_length = int.from_bytes(length_bytes, "big")
        logger.info(f"Response length: {msg_length}")

        response_bytes = await reader.readexactly(msg_length)
        response_str = response_bytes.decode("utf-8")
        logger.info(f"Received response: {response_str}")

        writer.close()
        await writer.wait_closed()
        logger.info("Connection closed")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Primary: python test_connection.py primary")
        print("  Secondary: python test_connection.py secondary <gateway_host> <gateway_port>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "primary":
        try:
            asyncio.run(test_primary())
        except KeyboardInterrupt:
            logger.info("Shutting down...")

    elif mode == "secondary":
        if len(sys.argv) < 4:
            print("Secondary mode requires gateway_host and gateway_port")
            print("Usage: python test_connection.py secondary <gateway_host> <gateway_port>")
            sys.exit(1)

        gateway_host = sys.argv[2]
        gateway_port = int(sys.argv[3])

        asyncio.run(test_secondary(gateway_host, gateway_port))

    else:
        print(f"Unknown mode: {mode}")
        print("Use 'primary' or 'secondary'")
        sys.exit(1)
