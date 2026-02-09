"""
Parallel Processing Utilities

Provides standardized parallel processing for the PRIMVS pipeline.

Author: Niall Miller
Date: 2025-10-21
"""

import multiprocessing as mp
from typing import Callable, List, Any, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def parallel_process(
    func: Callable,
    items: List[Any],
    n_processes: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process items in parallel using multiprocessing.
    
    Provides a clean interface for parallel processing with progress tracking.
    Automatically handles the number of processes and error handling.
    
    Args:
        func: Function to apply to each item. Must be picklable.
        items: List of items to process
        n_processes: Number of processes to use. If None, uses all available cores - 1.
        show_progress: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of results in same order as input items
        
    Example:
        >>> def process_source(source_id):
        ...     # Process source
        ...     return result
        >>> 
        >>> source_ids = [1, 2, 3, 4, 5]
        >>> results = parallel_process(process_source, source_ids, n_processes=4)
        
    Notes:
        - Uses multiprocessing.Pool for true parallelism
        - Preserves order of results
        - Handles errors gracefully
        - Shows progress bar by default
    """
    # Determine number of processes
    if n_processes is None or n_processes == -1:
        n_processes = max(1, mp.cpu_count() - 1)
    
    n_processes = min(n_processes, len(items))  # Don't use more processes than items
    
    logger.info(f"Processing {len(items)} items using {n_processes} processes")
    
    # Handle edge case of single process
    if n_processes == 1:
        if show_progress:
            results = [func(item) for item in tqdm(items, desc=desc)]
        else:
            results = [func(item) for item in items]
        return results
    
    # Parallel processing
    try:
        with mp.Pool(processes=n_processes) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(func, items),
                    total=len(items),
                    desc=desc
                ))
            else:
                results = pool.map(func, items)
        
        logger.info(f"Successfully processed {len(results)} items")
        return results
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise


def parallel_process_chunks(
    func: Callable,
    items: List[Any],
    chunk_size: int = 100,
    n_processes: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process items in parallel with chunking for better performance.
    
    Useful when processing many small items - chunking reduces overhead.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        chunk_size: Number of items per chunk
        n_processes: Number of processes to use
        show_progress: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of results in same order as input items
        
    Example:
        >>> def process_source(source_id):
        ...     return calculate_features(source_id)
        >>> 
        >>> results = parallel_process_chunks(
        ...     process_source,
        ...     source_ids,
        ...     chunk_size=100,
        ...     n_processes=16
        ... )
    """
    # Determine number of processes
    if n_processes is None or n_processes == -1:
        n_processes = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Processing {len(items)} items in chunks of {chunk_size} using {n_processes} processes")
    
    # Handle edge case of single process
    if n_processes == 1:
        if show_progress:
            results = [func(item) for item in tqdm(items, desc=desc)]
        else:
            results = [func(item) for item in items]
        return results
    
    # Parallel processing with chunking
    try:
        with mp.Pool(processes=n_processes) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(func, items, chunksize=chunk_size),
                    total=len(items),
                    desc=desc
                ))
            else:
                results = pool.map(func, items, chunksize=chunk_size)
        
        logger.info(f"Successfully processed {len(results)} items")
        return results
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise


def get_optimal_n_processes(n_items: int, max_processes: Optional[int] = None) -> int:
    """
    Determine optimal number of processes for a given number of items.
    
    Args:
        n_items: Number of items to process
        max_processes: Maximum number of processes to use
        
    Returns:
        Optimal number of processes
        
    Example:
        >>> n_proc = get_optimal_n_processes(1000, max_processes=32)
    """
    # Get available cores
    available_cores = mp.cpu_count()
    
    # Use all cores - 1 by default
    optimal = max(1, available_cores - 1)
    
    # Don't use more processes than items
    optimal = min(optimal, n_items)
    
    # Respect max_processes limit
    if max_processes is not None:
        optimal = min(optimal, max_processes)
    
    return optimal

