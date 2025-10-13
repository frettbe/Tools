import os
import logging
from functools import wraps

# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_ENDPOINT = "https://api.example.com"

# Decorator
def retry_on_failure(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
            return None
        return wrapper
    return decorator

class APIClient:
    """A sample API client to test class analysis."""
    
    def __init__(self, base_url):
        self.base_url = base_url
    
    @retry_on_failure(max_attempts=MAX_RETRIES)
    def fetch_data(self, endpoint):
        """Fetch data from the API."""
        try:
            # Simulate API call
            result = f"Data from {self.base_url}/{endpoint}"
            return result
        except ConnectionError as e:
            logging.error(f"Connection failed: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise

def process_item(item_id: int) -> str:
    """Process a single item."""
    if item_id <= 0:
        raise ValueError("Item ID must be positive")
    
    return f"Processed item {item_id}"

async def async_process_items(items):
    """Asynchronously process multiple items."""
    results = []
    for item in items:
        result = await process_item_async(item)
        results.append(result)
    return results

async def process_item_async(item):
    """Async version of item processing."""
    return f"Processed async item {item}"

if __name__ == "__main__":
    print("Running advanced test")
    client = APIClient(API_ENDPOINT)
    try:
        data = client.fetch_data("test")
        print(data)
    except Exception as e:
        print(f"Error: {e}")