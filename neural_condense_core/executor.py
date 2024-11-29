from concurrent.futures import ThreadPoolExecutor

THREAD_POOL_SIZE: int = 8
THREAD_POOL: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
