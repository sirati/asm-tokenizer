#!/usr/bin/env python3
"""Integration test for primary-secondary connection without SLURM

This script tests the full connection flow:
1. Primary coordinator starts and listens
2. Secondary mode connects and sends welcome
3. Messages are exchanged
4. Connection is validated

Usage:
    # Test without SSH (local only):
    python test_integration.py local

    # Test with SSH port forwarding to gateway:
    python test_integration.py ssh
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s,%(msecs)03d | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_mock_secondary(gateway_host: str, gateway_port: int, secondary_id: str):
    """Run a mock secondary that mimics the container behavior"""
    logger.info("=" * 60)
    logger.info(f"MOCK SECONDARY: {secondary_id}")
    logger.info("=" * 60)

    # Simulate what happens in the container
    from dynamic_batch.slurm.secondary_mode import SecondaryMode

    # Mock paths (would be /app/* in container)
    src_tmp = Path("/tmp/mock-src-tmp")
    out_tmp = Path("/tmp/mock-out-tmp")
    log_tmp = Path("/tmp/mock-log-tmp")
    src_network = Path("/tmp/mock-src-network")
    out_network = Path("/tmp/mock-out-network")
    log_network = Path("/tmp/mock-log-network")
    socket_dir = Path("/tmp/mock-sockets")

    # Create directories
    for p in [src_tmp, out_tmp, log_tmp, src_network, out_network, log_network, socket_dir]:
        p.mkdir(parents=True, exist_ok=True)

    try:
        # Create secondary mode
        secondary = SecondaryMode(
            primary_url=f"tcp://{gateway_host}:{gateway_port}",
            secondary_id=secondary_id,
            num_workers=4,
            ram_bytes=8 * 1024 * 1024 * 1024,  # 8GB
            src_tmp=src_tmp,
            out_tmp=out_tmp,
            log_tmp=log_tmp,
            src_network=src_network,
            out_network=out_network,
            log_network=log_network,
            socket_dir=socket_dir,
        )

        # Run for limited time
        logger.info("Starting secondary mode...")
        await asyncio.wait_for(secondary._run_async(), timeout=15.0)

    except asyncio.TimeoutError:
        logger.info("Secondary test completed (timeout)")
    except KeyboardInterrupt:
        logger.info("Secondary interrupted")
    except Exception as e:
        logger.error(f"Secondary error: {e}", exc_info=True)
    finally:
        # Cleanup
        import shutil

        for p in [src_tmp, out_tmp, log_tmp, src_network, out_network, log_network, socket_dir]:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)


async def run_mock_primary(num_secondaries: int = 1):
    """Run a mock primary coordinator"""
    logger.info("=" * 60)
    logger.info("MOCK PRIMARY COORDINATOR")
    logger.info("=" * 60)

    from dynamic_batch.gateway.local_gateway import LocalGateway
    from dynamic_batch.slurm import SlurmConfig
    from dynamic_batch.slurm.coordinator import PrimaryCoordinator

    # Create mock dependencies
    binaries = []  # No actual binaries for this test

    slurm_config = SlurmConfig(
        root_folder=Path("/tmp/mock-slurm"),
        image_subfolder="image_bin",
        output_subfolder="out",
        log_subfolder="log",
        notify_email=None,
    )

    # Create directories
    slurm_config.root_folder.mkdir(parents=True, exist_ok=True)
    (slurm_config.root_folder / "image_bin" / "srcbins").mkdir(parents=True, exist_ok=True)
    (slurm_config.root_folder / "out").mkdir(parents=True, exist_ok=True)
    (slurm_config.root_folder / "log").mkdir(parents=True, exist_ok=True)

    # Use local gateway for testing
    gateway = LocalGateway()
    gateway.connect()

    # Mock job manager that doesn't actually submit jobs
    class MockJobManager:
        def __init__(self):
            self.jobs = []

        def generate_wrapper_script(self, **kwargs):
            return "#!/bin/bash\necho 'mock job'"

        def submit_job(self, wrapper, job_name):
            job_id = f"mock-{len(self.jobs)}"
            self.jobs.append({"id": job_id, "name": job_name})
            logger.info(f"MOCK: Would submit job {job_name} (ID: {job_id})")
            return job_id

    job_manager = MockJobManager()

    # Create coordinator
    coordinator = PrimaryCoordinator(binaries, slurm_config, job_manager, gateway)

    # Override _submit_slurm_jobs to not actually submit
    original_submit = coordinator._submit_slurm_jobs

    def mock_submit(num_secondaries, base_port):
        logger.info(f"MOCK: Skipping actual SLURM job submission")
        logger.info(f"MOCK: In real scenario, {num_secondaries} jobs would be submitted")
        # Don't actually call original_submit
        pass

    coordinator._submit_slurm_jobs = mock_submit

    try:
        # Run coordinator with short timeout
        logger.info("Starting coordinator...")
        await asyncio.wait_for(coordinator._run_async(num_secondaries, 5000), timeout=20.0)

    except asyncio.TimeoutError:
        logger.info("Primary test completed (timeout)")
    except KeyboardInterrupt:
        logger.info("Primary interrupted")
    except Exception as e:
        logger.error(f"Primary error: {e}", exc_info=True)
    finally:
        await coordinator._cleanup()
        gateway.disconnect()
        # Cleanup
        import shutil

        if slurm_config.root_folder.exists():
            shutil.rmtree(slurm_config.root_folder, ignore_errors=True)


async def test_local():
    """Test with local connection (no SSH)"""
    logger.info("=" * 60)
    logger.info("INTEGRATION TEST: LOCAL MODE")
    logger.info("=" * 60)
    logger.info("")

    # Start primary in background
    primary_task = asyncio.create_task(run_mock_primary(num_secondaries=1))

    # Wait for primary to start listening
    await asyncio.sleep(3)

    # Start secondary
    secondary_task = asyncio.create_task(run_mock_secondary("localhost", 5000, "test-secondary-0"))

    # Wait for both to complete or timeout
    try:
        await asyncio.gather(primary_task, secondary_task)
    except Exception as e:
        logger.error(f"Test error: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("LOCAL TEST COMPLETED")
    logger.info("=" * 60)


async def test_ssh():
    """Test with SSH port forwarding"""
    logger.info("=" * 60)
    logger.info("INTEGRATION TEST: SSH MODE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("NOTE: This requires manual SSH port forwarding setup:")
    logger.info("  ssh -R 6000:localhost:5000 lmu")
    logger.info("")
    logger.info("Press Ctrl+C to skip or wait 10 seconds to continue...")

    # Give user time to cancel if SSH isn't set up
    try:
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        logger.info("SSH test skipped")
        return

    # Start primary in background
    primary_task = asyncio.create_task(run_mock_primary(num_secondaries=1))

    # Wait for primary to start
    await asyncio.sleep(2)

    # For SSH test, secondary would need to run on gateway
    # This is just a placeholder to show the structure
    logger.info("To test SSH mode:")
    logger.info("1. Ensure SSH port forwarding is active: ssh -R 6000:localhost:5000 lmu")
    logger.info("2. On gateway, run:")
    logger.info("   python test_integration.py secondary lmu 6000 test-secondary-0")
    logger.info("")
    logger.info("Waiting for primary to timeout...")

    # Just let primary run
    try:
        await primary_task
    except Exception as e:
        logger.error(f"Test error: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SSH TEST COMPLETED")
    logger.info("=" * 60)


async def test_secondary_only(gateway_host: str, gateway_port: int, secondary_id: str):
    """Run only the secondary (for testing from gateway)"""
    logger.info("Running secondary only mode")
    await run_mock_secondary(gateway_host, gateway_port, secondary_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Local test:     python test_integration.py local")
        print("  SSH test:       python test_integration.py ssh")
        print("  Secondary only: python test_integration.py secondary <host> <port> <id>")
        sys.exit(1)

    mode = sys.argv[1]

    try:
        if mode == "local":
            asyncio.run(test_local())

        elif mode == "ssh":
            asyncio.run(test_ssh())

        elif mode == "secondary":
            if len(sys.argv) < 5:
                print("Secondary mode requires: host port id")
                print("Usage: python test_integration.py secondary <host> <port> <id>")
                sys.exit(1)

            gateway_host = sys.argv[2]
            gateway_port = int(sys.argv[3])
            secondary_id = sys.argv[4]

            asyncio.run(test_secondary_only(gateway_host, gateway_port, secondary_id))

        else:
            print(f"Unknown mode: {mode}")
            print("Use 'local', 'ssh', or 'secondary'")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
