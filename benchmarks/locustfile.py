import random
from locust import HttpUser, task, between

# Sample queries representing frequent questions and their semantic paraphrases
PROMPTS = [
    "How do I reverse a string in Python?",
    "What is the time complexity of quicksort?",
    "Explain the difference between TCP and UDP",
    "How does asyncio work in Python?",
    "What is Docker and how does containerization work?",
    "Python method to reverse string",
    "Average and worst case time complexity of quick sort",
    "Compare TCP vs UDP protocols",
    "How does Python asyncio event loop work?",
    "Explain containerization with Docker",
]


class ScacheHitUser(HttpUser):
    # Rapid fire requests simulating concurrent cached lookups
    wait_time = between(0.005, 0.02)

    @task(8)
    def query_cached_prompts(self):
        """Simulates queries against cached items."""
        prompt = random.choice(PROMPTS)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        self.client.post("/v1/chat/completions", json=payload, headers={"X-Tenant-ID": "bench"})

    @task(1)
    def check_health(self):
        self.client.get("/health")

    @task(1)
    def check_metrics(self):
        self.client.get("/metrics")
