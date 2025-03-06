def log_message(message):
    print(f"[LOG] {message}")

def benchmark(func, *args, **kwargs):
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    return result
