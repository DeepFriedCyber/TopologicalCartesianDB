import subprocess
import sys
import os

def run_command(cmd, cwd=None):
    """Runs a command and exits if it fails."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {' '.join(e.cmd)}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Command '{cmd[0]}' not found. Make sure it's installed and in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    # 1. Start Docker Compose
    compose_file = "docker-compose.dynamic.yml" if os.path.exists("docker-compose.dynamic.yml") else "docker-compose.yml"
    
    try:
        run_command(["docker", "compose", "-f", compose_file, "up", "-d"])

        # 2. Run the benchmark script
        # Use sys.executable to ensure we're using the same python interpreter
        benchmark_script = os.path.join("vectordb-benchmark", "run_comprehensive_benchmark.py")
        run_command([sys.executable, benchmark_script])

        print("\nAll services started and benchmark completed.\n")

    finally:
        print("To stop all services, run: docker compose -f {} down".format(compose_file))
