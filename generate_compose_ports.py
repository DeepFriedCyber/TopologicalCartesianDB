import socket
import yaml

# Default ports for each service
services = {
    "qdrant": 6333,
    "weaviate": 8080,
    "milvus": [19530, 9091],
    "redis": 6379,
    "elasticsearch": 9200,
}

def find_free_port(start_port):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1

# Load your existing docker-compose.yml
with open("docker-compose.yml", "r") as f:
    compose = yaml.safe_load(f)

for name, default in services.items():
    if name in compose["services"]:
        if isinstance(default, list):
            new_ports = []
            for p in default:
                free = find_free_port(p)
                new_ports.append(f"{free}:{p}")
            compose["services"][name]["ports"] = new_ports
        else:
            free = find_free_port(default)
            compose["services"][name]["ports"] = [f"{free}:{default}"]

# Save the new docker-compose file
with open("docker-compose.dynamic.yml", "w") as f:
    yaml.dump(compose, f)

print("Generated docker-compose.dynamic.yml with available ports.")
