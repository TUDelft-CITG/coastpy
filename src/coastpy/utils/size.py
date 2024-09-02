def size_to_bytes(size: str | float | int) -> int:
    """
    Convert human-readable size string or number to bytes.

    Args:
        size (Union[str, float, int]): Size either in a human-readable format like "100MB", "5GB", or as a raw number.

    Returns:
        int: Size in bytes.

    Example:
        >>> size_to_bytes("5GB")
        5368709120
    """
    if isinstance(size, float | int):
        return int(size)

    size_str = size.strip().lower()
    size_units = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}

    for unit, multiplier in size_units.items():
        if size_str.endswith(unit):
            try:
                size_num = float(size_str[: -len(unit)])
                return int(size_num * multiplier)
            except ValueError as err:
                msg = f"Invalid size specification: {size}"
                raise ValueError(msg) from err

    msg = "Invalid size specification."
    raise ValueError(msg)


def readable_bytes(num_bytes: int | float) -> str:
    """
    Convert bytes to a readable format (B, KB, MB, GB, TB).

    Args:
        num_bytes (int): Number of bytes

    Returns:
        str: Readable format string.

    Example:
        >>> bytes_to_readable_size(5368709120)
        '5.00 GB'
    """
    if num_bytes < 0:
        msg = "num_bytes should be a non-negative integer."
        raise ValueError(msg)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0

    return f"{num_bytes:.2f} PB"  # Handle very large sizes
